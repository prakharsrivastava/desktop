"""
util.py — Business Logic Utilities for e360 Inquiry Categorization
===================================================================
LLM prompts, category classification, entity extraction — core logic.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Tuple

from horizon_langchain_obj import HorizonLlmChat, TokenManager
from get_util import (
    safe_parse_json,
    validate_category,
    validate_priority,
    get_sla_hours,
    get_pega_queue,
    truncate_text,
)
from models import (
    InquiryCategoryRequest,
    InquiryCategory,
    ExtractedInquiryEntities,
    InquiryPriority,
    InquiryCategoryResponse,
)
from logger import get_context_logger

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# LLM PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

CATEGORIZATION_PROMPT = """
You are an inquiry categorization assistant for a health insurance company (e360 system).

Read the following inquiry and classify it into exactly one category.

Subject: {subject}
Body: {body}

CATEGORIES (choose exactly one):
1. "ID Card Change Request" — Member wants new/replacement ID card
2. "Address Change Request" — Member wants to update mailing/home address
3. "General Enquiry"        — Questions, information requests, status updates
4. "Billing Inquiry"        — Questions about bills, payments, premium
5. "Claims Inquiry"         — Questions about claims, EOB, reimbursement
6. "Benefits Inquiry"       — Questions about coverage, benefits, plan details
7. "Provider Inquiry"       — Questions about doctors, network, referrals
8. "Other"                  — Does not fit above categories

Rules:
- Return EXACTLY one category name from the list above
- If multiple apply, choose the PRIMARY intent
- Provide confidence score 0.0–1.0
- Provide brief reasoning

Return JSON only, no backticks:
{{"category": "General Enquiry", "confidence": 0.85, "reasoning": "Member is asking about..."}}
"""

ENTITY_EXTRACTION_PROMPT = """
You are an entity extraction assistant for a health insurance company.

Extract entities from this inquiry. Return ONLY what is explicitly present.

Subject: {subject}
Body: {body}

Extract:
- hcid: Healthcare ID numbers
- policy_numbers: Policy/contract numbers
- case_numbers: Case/ticket numbers
- dates: Date strings mentioned
- amounts: Dollar amounts or financial figures
- phone_numbers: Phone numbers
- addresses: Physical addresses
- action_items: Specific actions requested
- provider_names: Doctor/hospital names

Return JSON only, no backticks:
{{
  "hcid": [],
  "policy_numbers": [],
  "case_numbers": [],
  "dates": [],
  "amounts": [],
  "phone_numbers": [],
  "addresses": [],
  "action_items": [],
  "provider_names": []
}}
"""

PRIORITY_PROMPT = """
You are a priority assignment assistant for a healthcare operations system.

Classify this inquiry's priority level.

Subject: {subject}
Body: {body}
Category: {category}

PRIORITY LEVELS:
1. "Urgent" — Keywords: URGENT, ASAP, Emergency, Critical, time-sensitive, escalation
2. "Access to Care" — Keywords: doctor appointment, surgery, prescription, medication, RX, treatment
3. "Standard" — Routine inquiries, information requests, status updates

Rules:
- Priority order: Urgent > Access to Care > Standard
- Default to "Standard" if no keywords match
- Provide brief explanation

Return JSON only, no backticks:
{{"level": "Standard", "explanation": "Routine inquiry with no urgency keywords"}}
"""

SUMMARY_PROMPT = """
You are a summary assistant for a health insurance company.

Create a concise 2-3 line summary of this inquiry.
Focus on: what the member wants, any key details (HCID, case number, etc.)
Do NOT include signatures, disclaimers, or boilerplate.

Subject: {subject}
Body: {body}

Return JSON only, no backticks:
{{"summary": "Member is requesting..."}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# CORE PROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _get_llm(client_id: str, client_secret: str) -> HorizonLlmChat:
    """HorizonLlmChat instance banao."""
    token_manager = TokenManager(client_id=client_id, client_secret=client_secret)
    return HorizonLlmChat(
        api_endpoint="https://api.horizon.elevancehealth.com/v2/text/chats",
        token_manager=token_manager,
    )


def categorize_inquiry(
    subject: str,
    body: str,
    client_id: str,
    client_secret: str,
) -> InquiryCategory:
    """Inquiry ko category assign karo using LLM."""
    llm    = _get_llm(client_id, client_secret)
    prompt = CATEGORIZATION_PROMPT.format(
        subject=subject or "",
        body=truncate_text(body or "", 2000),
    )
    response = llm.invoke(prompt)
    parsed   = safe_parse_json(response.content, fallback={})

    return InquiryCategory(
        primary=validate_category(parsed.get("category", "Other")),
        confidence=float(parsed.get("confidence", 0.5)),
        reasoning=parsed.get("reasoning", "Unable to determine reasoning"),
    )


def extract_inquiry_entities(
    subject: str,
    body: str,
    client_id: str,
    client_secret: str,
) -> ExtractedInquiryEntities:
    """Inquiry se entities extract karo."""
    llm    = _get_llm(client_id, client_secret)
    prompt = ENTITY_EXTRACTION_PROMPT.format(
        subject=subject or "",
        body=truncate_text(body or "", 2000),
    )
    response = llm.invoke(prompt)
    parsed   = safe_parse_json(response.content, fallback={})

    return ExtractedInquiryEntities(
        hcid=parsed.get("hcid", []),
        policy_numbers=parsed.get("policy_numbers", []),
        case_numbers=parsed.get("case_numbers", []),
        dates=parsed.get("dates", []),
        amounts=parsed.get("amounts", []),
        phone_numbers=parsed.get("phone_numbers", []),
        addresses=parsed.get("addresses", []),
        action_items=parsed.get("action_items", []),
        provider_names=parsed.get("provider_names", []),
    )


def assign_inquiry_priority(
    subject: str,
    body: str,
    category: str,
    client_id: str,
    client_secret: str,
) -> InquiryPriority:
    """Priority assign karo."""
    llm    = _get_llm(client_id, client_secret)
    prompt = PRIORITY_PROMPT.format(
        subject=subject or "",
        body=truncate_text(body or "", 2000),
        category=category,
    )
    response = llm.invoke(prompt)
    parsed   = safe_parse_json(response.content, fallback={})
    level    = validate_priority(parsed.get("level", "Standard"))

    return InquiryPriority(
        level=level,
        explanation=parsed.get("explanation", "Priority assigned based on content"),
        sla_hours=get_sla_hours(level),
    )


def summarize_inquiry(
    subject: str,
    body: str,
    client_id: str,
    client_secret: str,
) -> str:
    """Inquiry ka summary banao."""
    llm    = _get_llm(client_id, client_secret)
    prompt = SUMMARY_PROMPT.format(
        subject=subject or "",
        body=truncate_text(body or "", 2000),
    )
    response = llm.invoke(prompt)
    parsed   = safe_parse_json(response.content, fallback={})
    return parsed.get("summary", "No relevant content to summarize")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_inquiry(
    request: InquiryCategoryRequest,
    request_id: Optional[str] = None,
) -> InquiryCategoryResponse:
    """
    Full inquiry processing pipeline.

    Steps:
        1. Summarize
        2. Categorize
        3. Extract entities
        4. Assign priority
        5. Build response with PEGA routing
    """
    import uuid
    if not request_id:
        request_id = str(uuid.uuid4())

    log = get_context_logger(
        __name__,
        request_id=request_id,
        inquiry_id=request.inquiry_id,
    )

    subject = request.subject or ""
    body    = request.body    or ""
    cid     = request.client_id
    csec    = request.client_secret

    total_start = time.time()

    # ── Step 1: Summary ───────────────────────────────────────────────────────
    t = time.time()
    summary = summarize_inquiry(subject, body, cid, csec)
    log.step("summarize_inquiry", duration_ms=int((time.time()-t)*1000))

    # ── Step 2: Category ──────────────────────────────────────────────────────
    t = time.time()
    category = categorize_inquiry(subject, body, cid, csec)
    log.step("categorize_inquiry", duration_ms=int((time.time()-t)*1000))

    # ── Step 3: Entities ──────────────────────────────────────────────────────
    t = time.time()
    entities = extract_inquiry_entities(subject, body, cid, csec)
    log.step("extract_entities", duration_ms=int((time.time()-t)*1000))

    # ── Step 4: Priority ──────────────────────────────────────────────────────
    t = time.time()
    priority = assign_inquiry_priority(subject, body, category.primary, cid, csec)
    log.step("assign_priority", duration_ms=int((time.time()-t)*1000))

    # ── Step 5: Routing + Human review ───────────────────────────────────────
    pega_queue = get_pega_queue(category.primary)
    human_review = (
        priority.level in ("Urgent", "Access to Care")
        or category.primary == "Other"
        or category.confidence < 0.6
    )

    total_ms = int((time.time() - total_start) * 1000)

    log.step("inquiry_processing_complete", duration_ms=total_ms,
             category=category.primary, priority=priority.level)

    return InquiryCategoryResponse(
        inquiry_id=request.inquiry_id,
        summary=summary,
        category=category,
        entities=entities,
        priority=priority,
        suggested_routing=pega_queue,
        human_review_required=human_review,
        processing_time_ms=total_ms,
        original_subject=subject or None,
        original_body=body or None,
    )
