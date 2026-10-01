from __future__ import annotations

import re
import json
import os
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import getaddresses
from io import BytesIO
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.config import AppConfig
from app.schemas import (
    EmailIntelligenceRequest,
    EmailIntelligenceResponse,
    ExtractedEntities,
    PriorityAssignment
)
# Sentiment analysis is handled by separate MCP server

import extract_msg
from horizon_langchain import HorizonLlmChat, TokenManager
from langchain_core.messages import HumanMessage, SystemMessage
from app.agent_logger import log_step
import time


def process_uploaded_email(
    file_name: str | None,
    content_type: str | None,
    file_bytes: bytes,
    config: AppConfig,
) -> EmailIntelligenceResponse:
    """Route uploaded bytes to the appropriate parser."""

    if _looks_like_msg(file_name, content_type):
        return process_msg_file(file_bytes, config)

    # Default to RFC822/.eml parsing
    return process_eml_file(file_bytes, config)


def _looks_like_msg(file_name: str | None, content_type: str | None) -> bool:
    """Infer whether the upload is a .msg file."""

    if file_name and file_name.lower().endswith(".msg"):
        return True

    if content_type and "application/vnd.ms-outlook" in content_type.lower():
        return True

    return False


def process_msg_file(msg_bytes: bytes, config: AppConfig) -> EmailIntelligenceResponse:
    """Parse an Outlook .msg file and delegate to the main analyzer."""

    with BytesIO(msg_bytes) as buffer:
        message = extract_msg.Message(buffer)
        email_request = _convert_msg_to_email_request(message)
    return process_email_intelligence(email_request, config)


def _convert_msg_to_email_request(message: extract_msg.Message) -> EmailIntelligenceRequest:
    """Convert an extract_msg Message into EmailIntelligenceRequest."""

    subject = message.subject or ""
    body = message.body or message.bodyRTF or ""
    from_email = getattr(message, "sender_email", None) or getattr(message, "sender", None)
    to_emails = _split_recipient_string(getattr(message, "to", None))
    cc_emails = _split_recipient_string(getattr(message, "cc", None))

    sent_at = getattr(message, "date", None)

    return EmailIntelligenceRequest(
        subject=subject,
        body=body,
        from_email=from_email,
        to_emails=to_emails or None,
        cc_emails=cc_emails or None,
        sent_at=sent_at,
    )


def _split_recipient_string(raw: str | None) -> List[str]:
    if not raw:
        return []
    candidates = re.split(r"[;,]", raw)
    return [item.strip() for item in candidates if item and item.strip()]


def process_eml_file(eml_bytes: bytes, config: AppConfig) -> EmailIntelligenceResponse:
    """Parse a raw .eml file and delegate to the main email processor."""

    message = BytesParser(policy=policy.default).parsebytes(eml_bytes)
    email_request = _convert_message_to_email_request(message)
    return process_email_intelligence(email_request, config)


def _convert_message_to_email_request(message: EmailMessage, client_id: str, client_secret: str) -> EmailIntelligenceRequest:
    """Convert an EmailMessage parsed from .eml bytes into EmailIntelligenceRequest."""

    subject = message.get("subject", "") or ""
    body = _get_email_body(message)

    from_email = None
    raw_from = message.get("from")
    if raw_from:
        addresses = getaddresses([raw_from])
        if addresses:
            _, addr = addresses[0]
            from_email = addr or None

    to_emails: List[str] = []
    cc_emails: List[str] = []

    for header, target in (("to", to_emails), ("cc", cc_emails)):
        raw_value = message.get(header)
        if not raw_value:
            continue
        for _, addr in getaddresses([raw_value]):
            if addr:
                target.append(addr)

    return EmailIntelligenceRequest(
        subject=subject,
        body=body,
        from_email=from_email,
        to_emails=to_emails or None,
        cc_emails=cc_emails or None,
        client_id=client_id,
        client_secret=client_secret,
    )


def _get_email_body(message: EmailMessage) -> str:
    """Extract a reasonable text body from an EmailMessage."""

    if message.is_multipart():
        parts: List[str] = []
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                try:
                    parts.append(part.get_content())
                except Exception:
                    # If a part cannot be decoded, skip it but continue.
                    continue
        return "\n".join(parts).strip()

    try:
        return (message.get_content() or "").strip()
    except Exception:
        return ""


def process_email_intelligence(request: EmailIntelligenceRequest, config: AppConfig, request_id: str = None, user_id: str = None) -> EmailIntelligenceResponse:
    """Analyze an email and return summary, entities, and priority assignment."""

    # Generate request_id if not provided
    if request_id is None:
        import uuid
        request_id = str(uuid.uuid4())

    # Log: Email input received
    log_step(
        request_id=request_id,
        step_name="email_input_received",
        input_data={
            "subject": request.subject,
            "body_length": len(request.body) if request.body else 0,
            "from_email": request.from_email
        },
        status="success",
        user_id=user_id
    )

    if config.include_debug_info:
        print("Debug process_email_intelligence:", request)

    # Extract credentials from request
    client_id = request.client_id
    client_secret = request.client_secret

    # Generate summary and classify intent using LLM
    start_time = time.time()
    summary, intent = _generate_email_summary(request.subject, request.body, config, client_id, client_secret)
    summary_duration = int((time.time() - start_time) * 1000)

    # Log: Summarization and intent classification complete
    log_step(
        request_id=request_id,
        step_name="summarization_complete",
        input_data={"subject": request.subject, "body_length": len(request.body) if request.body else 0},
        output_data={"summary": summary, "intent": intent},
        status="success",
        duration_ms=summary_duration,
        user_id=user_id
    )

    # Extract entities using LLM
    start_time = time.time()
    entities = _extract_entities(request, summary, config, client_id, client_secret)
    entities_duration = int((time.time() - start_time) * 1000)

    # Log: Entity extraction complete
    log_step(
        request_id=request_id,
        step_name="entity_extraction_complete",
        input_data={"summary": summary},
        output_data=entities.dict(),
        status="success",
        duration_ms=entities_duration,
        user_id=user_id
    )

    # Assign priority using LLM
    start_time = time.time()
    priority = _assign_priority(request.subject, request.body, summary, config, client_id, client_secret)
    priority_duration = int((time.time() - start_time) * 1000)

    # Log: Priority assignment complete
    log_step(
        request_id=request_id,
        step_name="priority_assignment_complete",
        input_data={"summary": summary},
        output_data=priority.dict(),
        status="success",
        duration_ms=priority_duration,
        user_id=user_id
    )

    if config.include_debug_info:
        print("Debug process_email_intelligence results:", summary, entities, priority)

    # Create initial response
    response = EmailIntelligenceResponse(
        summary=summary,
        intent=intent,
        extracted_entities=entities,
        priority=priority,
        original_subject=request.subject,
        original_body=request.body
    )

    # Sentiment analysis is handled by separate MCP server
    # Agent focuses on: summary, entity extraction (HCID), priority

    return response


def _generate_email_summary(subject: str | None, body: str | None, config: AppConfig, client_id: str, client_secret: str) -> tuple[str, str]:
    """Generate a concise summary of the email and classify intent using LLM.

    Returns:
        tuple[str, str]: (summary, intent)
        - summary: 3-4 line summary of email
        - intent: One of "ID Card Change Request", "Address Change Request", "General Enquiry", "Other"
    """

    print("Debug: Initializing TokenManager with user credentials...")
    token_manager = TokenManager(client_id=client_id, client_secret=client_secret)
    print("Debug: Using user-provided credentials")

    llm = HorizonLlmChat(
        api_endpoint="https://api.horizon.elevancehealth.com/v2/text/chats",
        token_manager=token_manager
    )

    SUMMARY_PROMPT_TEMPLATE = """
You are an experienced email intelligence assistant for a health insurance company, tasked with creating concise, accurate summaries of member emails and classifying their intent.

Read the following email:
Subject: {subject}
Body: {body}

Strictly follow these rules:
1. Create a summary limited to 3-4 lines maximum
2. Focus on the latest meaningful customer request, ignoring forwarded history
3. Ignore email signatures, disclaimers, and boilerplate content
4. If available (meaning EXPLICITLY present in the text), capture important details including:
   * Member information (names, IDs) - only if directly stated
   * Policy details - only if directly stated
   * Case details - only if directly stated
   * Specific requests or issues - only if directly stated
5. If the email lacks relevant content, return "No relevant content to summarize"
6. Do not generate any information beyond what is explicitly present
7. Do not include code, newline characters, or tab characters

**Intent Classification:**
You MUST classify the email into EXACTLY ONE of these four categories:

1. "ID Card Change Request" - Email is requesting a new ID card, replacement card, or changes to ID card information
2. "Address Change Request" - Email is requesting to update mailing address, home address, or contact address information
3. "General Enquiry" - Email is asking questions, requesting information, or seeking clarification (not requesting a physical change)
4. "Other" - Email does not fit into the above categories (complaints, claims, prescriptions, appointments, billing, etc.)

**Classification Rules:**
- You MUST return EXACTLY one of these four values: "ID Card Change Request", "Address Change Request", "General Enquiry", "Other"
- Do NOT create new categories or modify the category names
- If multiple intents are present, choose the PRIMARY intent
- If unclear, default to "General Enquiry" for questions or "Other" for everything else

Return the summary and intent in the following JSON format with no backticks:
{{"summary": "Your summarized text here", "intent": "ID Card Change Request"}}

**Valid intent values ONLY:**
- "ID Card Change Request"
- "Address Change Request"
- "General Enquiry"
- "Other"
"""

    # Handle null values
    subject_text = subject or ""
    body_text = body or ""

    prompt = SUMMARY_PROMPT_TEMPLATE.format(subject=subject_text, body=body_text)

    response = llm.invoke(prompt)

    if config.include_debug_info:
        print("debug-llm summary response-content", response.content)

    # Valid intent categories
    VALID_INTENTS = [
        "ID Card Change Request",
        "Address Change Request",
        "General Enquiry",
        "Other"
    ]

    try:
        parsed = json.loads(response.content)
        summary = parsed.get("summary", "No relevant content to summarize")
        intent = parsed.get("intent", "Other")

        # Validate intent - must be one of the four categories
        if intent not in VALID_INTENTS:
            print(f"Warning: Invalid intent '{intent}' returned by LLM, defaulting to 'Other'")
            intent = "Other"

        return summary, intent

    except json.JSONDecodeError:
        return "No relevant content to summarize", "Other"


def _extract_entities(request: EmailIntelligenceRequest, summary: str, config: AppConfig, client_id: str, client_secret: str) -> ExtractedEntities:
    """Extract relevant entities from the email using LLM."""

    token_manager = TokenManager(client_id=client_id, client_secret=client_secret)

    llm = HorizonLlmChat(
        api_endpoint="https://api.horizon.elevancehealth.com/v2/text/chats",
        token_manager=token_manager
    )

    ENTITY_EXTRACTION_PROMPT = """
You are an entity extraction assistant for a health insurance company. Extract relevant entities from the following email content.

Email Subject: {subject}
Email Body: {body}
Summary: {summary}

Extract the following entities ONLY if they are explicitly present in the email content. Return null for any entity not found:

1. sender_email: Email address of the sender
2. recipient_emails: List of recipient email addresses
3. dates: List of dates mentioned (format as strings)
4. amounts: List of monetary amounts or financial figures
5. company_names: List of company or organization names
6. hcid: List of healthcare identification numbers (HCID)
7. policy_numbers: List of policy or contract numbers
8. case_numbers: List of case or ticket numbers
9. phone_numbers: List of phone numbers
10. addresses: List of physical addresses
11. action_items: List of specific actions requested or required

Rules:
- Only extract information that is EXPLICITLY stated in the email
- Do not guess or fabricate any information
- Return empty arrays for lists with no items
- Return null for single values not found

Return the extracted entities in the following JSON format with no backticks:
{{
    "sender_email": "email@example.com" or null,
    "recipient_emails": ["email1@example.com", "email2@example.com"] or [],
    "dates": ["2024-01-15", "January 15, 2024"] or [],
    "amounts": ["$100.00", "500 dollars"] or [],
    "company_names": ["Company Name"] or [],
    "hcid": ["HCID123456"] or [],
    "policy_numbers": ["POL789012"] or [],
    "case_numbers": ["CASE345678"] or [],
    "phone_numbers": ["555-123-4567"] or [],
    "addresses": ["123 Main St, City, State"] or [],
    "action_items": ["Update member information", "Process refund"] or []
}}
"""

    # Handle null values
    subject_text = request.subject or ""
    body_text = request.body or ""

    prompt = ENTITY_EXTRACTION_PROMPT.format(
        subject=subject_text,
        body=body_text,
        summary=summary
    )

    response = llm.invoke(prompt)

    if config.include_debug_info:
        print("debug-llm entity response-content", response.content)

    try:
        parsed = json.loads(response.content)
        return ExtractedEntities(
            sender_email=parsed.get("sender_email"),
            recipient_emails=parsed.get("recipient_emails", []),
            dates=parsed.get("dates", []),
            amounts=parsed.get("amounts", []),
            company_names=parsed.get("company_names", []),
            hcid=parsed.get("hcid", []),
            policy_numbers=parsed.get("policy_numbers", []),
            case_numbers=parsed.get("case_numbers", []),
            phone_numbers=parsed.get("phone_numbers", []),
            addresses=parsed.get("addresses", []),
            action_items=parsed.get("action_items", [])
        )
    except json.JSONDecodeError:
        return ExtractedEntities()


def _assign_priority(subject: str | None, body: str | None, summary: str, config: AppConfig, client_id: str, client_secret: str) -> PriorityAssignment:
    """Assign priority level based on email content using LLM."""

    token_manager = TokenManager(client_id=client_id, client_secret=client_secret)

    llm = HorizonLlmChat(
        api_endpoint="https://api.horizon.elevancehealth.com/v2/text/chats",
        token_manager=token_manager
    )

    PRIORITY_ASSIGNMENT_PROMPT = """
You are a priority assignment assistant for a healthcare operations system. Classify the email content based on its urgency and nature.

Email Subject: {subject}
Email Body: {body}
Summary: {summary}

**Priority Labels:**
1. "Standard"
2. "Access to Care"
3. "Urgent"

**Classification Rules:**

1. **Standard**:
   - Assign "Standard" if the email contains any of the following keywords:
     ["White Glove", "Internal Audit", "Peer to Peer", "P2P Audit", "MTM Audit", "Inline Audit",
      "routine", "inquiry", "question", "information request", "status update"]

2. **Access to Care**:
   - Assign "Access to Care" if the email contains any of the following keywords:
     ["Access to Care", "Immediate Access to Care", "Doctor Appointment", "RX", "Immediate RX",
      "Code Blue", "RX needed", "Immediate", "Dr. Office", "Surgery", "Procedure",
      "medical appointment", "prescription", "medication", "treatment needed"]

3. **Urgent**:
   - Assign "Urgent" if the email contains any of the following keywords:
     ["URGENT", "ASAP", "Emergency", "Critical", "Immediate attention required",
      "time-sensitive", "deadline", "compliance issue", "escalation"]

**Priority Rules:**
1. If none of the keywords match, always default to "Standard"
2. You must respond with exactly one of: "Standard", "Access to Care", or "Urgent"
3. In case multiple keywords are found, prioritize in this order: "Urgent" > "Access to Care" > "Standard"
4. Provide a brief explanation mentioning which keywords were found

Return the priority assignment in the following JSON format with no backticks:
{{
    "level": "Standard" or "Access to Care" or "Urgent",
    "explanation": "Brief explanation mentioning keywords found"
}}
"""

    # Handle null values
    subject_text = subject or ""
    body_text = body or ""

    prompt = PRIORITY_ASSIGNMENT_PROMPT.format(
        subject=subject_text,
        body=body_text,
        summary=summary
    )

    response = llm.invoke(prompt)
    print("debug-llm priority response-content", response.content)
    print("debug-llm priority response-content-type", type(response.content))

    if config.include_debug_info:
        print("debug-llm priority response-content", response.content)

    try:
        parsed = json.loads(response.content)
        level = parsed.get("level", "Standard")
        explanation = parsed.get("explanation", "Priority assigned based on email content analysis")

        # Validate priority level using e360 system
        valid_priorities = ["Standard", "Access to Care", "Urgent"]
        if level not in valid_priorities:
            level = "Standard"

        return PriorityAssignment(level=level, explanation=explanation)
    except json.JSONDecodeError:
        return PriorityAssignment(
            level="Standard",
            explanation="Unable to determine priority frpm email content"
        )


def process_email_request(request: EmailIntelligenceRequest, config: AppConfig, request_id: str = None, user_id: str = None) -> EmailIntelligenceResponse:
    """Process an email intelligence request directly from structured input."""
     return Planner(config, request_id=request_id, user_id=user_id).run(request) 