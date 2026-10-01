"""
main.py — FastAPI Application for Email Intelligence Agent
===========================================================
Routes:
    GET  /                             → Scalar UI (custom docs)
    POST /login                        → JWT token
    POST /analyze-email                → Email intelligence (no sentiment)
    POST /analyze-email-upload         → Upload .eml/.msg (no sentiment)
    POST /analyze-sentiment            → MCP sentiment only
    POST /analyze-sentiment-upload     → Upload + MCP sentiment only
    POST /analyze-email-with-sentiment → Full intelligence + MCP sentiment
    POST /analyze-email-with-sentiment-upload → Upload + full + sentiment
    GET  /get-logs                     → Agent step logs
"""

from __future__ import annotations

import json
import time
import uuid
import traceback
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer

from app.config import AppConfig, get_config
from app.schemas import (
    EmailIntelligenceRequest,
    EmailIntelligenceResponse,
    HealthResponse,
)
from app.services.email_intelligence import process_email_request
from app.agent_logger import log_step, get_logger
from horizon_langchain import HorizonLlmChat, TokenManager
from app.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    User,
    Token,
    LoginRequest,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)


app = FastAPI(
    title="Email Assist Agent",
    version="0.1.0",
    description="Email Assist API with OAuth 2.0 Bearer Token Authentication",
    docs_url=None,    # Disable default Swagger UI
    redoc_url=None,   # Disable ReDoc
)

security = HTTPBearer()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def prepare_email_text(subject: str, body: str) -> str:
    """Combine subject and body into single text for sentiment analysis."""
    parts = []
    if subject:
        parts.append(f"Subject: {subject}")
    if body:
        parts.append(f"Body: {body}")
    return "\n".join(parts)


def should_analyze_sentiment(
    email_text: str,
    summary: str,
    priority: str,
    client_id: str,
    client_secret: str
) -> dict:
    """
    Use LLM to intelligently decide if sentiment analysis is needed.

    Returns:
        dict with keys:
        - decision: "call_mcp" or "skip_mcp"
        - reason: Brief explanation
        - confidence: 0.0 to 1.0
    """
    token_manager = TokenManager(client_id=client_id, client_secret=client_secret)
    llm = HorizonLlmChat(
        api_endpoint="https://api.horizon.elevancehealth.com/v2/text/chats",
        token_manager=token_manager
    )

    DECISION_PROMPT = """
You are a decision-making assistant that determines if sentiment analysis is needed for an email.

Email Summary: {summary}
Priority Level: {priority}
Email Preview (first 200 chars): {preview}

**Decision Criteria:**

CALL MCP SENTIMENT TOOL if the email contains:
1. Emotional language (frustrated, angry, upset, happy, grateful, disappointed)
2. Complaints or dissatisfaction
3. Urgency indicators (ASAP, urgent, emergency, immediate)
4. Strong opinions or feedback
5. Customer service issues
6. Healthcare-related concerns (pain, anxiety, medical urgency)

SKIP MCP SENTIMENT TOOL if the email is:
1. Purely informational (status updates, confirmations)
2. Automated/system-generated messages
3. Structured data only (forms, tables, lists)
4. Administrative/routine inquiries
5. Already classified as "Standard" priority with no emotional indicators

**Instructions:**
- Analyze the email content carefully
- Make a clear decision: "call_mcp" or "skip_mcp"
- Provide a brief reason (1-2 sentences)
- Assign confidence (0.0 to 1.0)

Return ONLY valid JSON in this exact format with no backticks:
{{
    "decision": "call_mcp" or "skip_mcp",
    "reason": "Brief explanation here",
    "confidence": 0.85
}}
"""

    preview = email_text[:200] if len(email_text) > 200 else email_text
    prompt = DECISION_PROMPT.format(
        summary=summary,
        priority=priority,
        preview=preview
    )

    try:
        response = llm.invoke(prompt)
        decision_data = json.loads(response.content)

        # Validate response
        if decision_data.get("decision") not in ["call_mcp", "skip_mcp"]:
            return {
                "decision": "call_mcp",
                "reason": "Invalid LLM decision format, defaulting to sentiment analysis",
                "confidence": 0.5
            }

        return decision_data

    except Exception as e:
        return {
            "decision": "call_mcp",
            "reason": f"Decision-making error: {str(e)}, defaulting to sentiment analysis",
            "confidence": 0.5
        }


def get_default_sentiment() -> dict:
    """Return default neutral sentiment when MCP is skipped."""
    return {
        "sentiment": "neutral",
        "confidence": 0.5,
        "emotional_tone": "neutral",
        "urgency_level": "low",
        "key_indicators": [],
        "healthcare_flags": {
            "medical_urgency": False,
            "insurance_frustration": False,
            "health_anxiety": False,
            "service_appreciation": False
        },
        "source": "skipped_mcp"
    }


def get_fallback_sentiment(email_text: str) -> dict:
    """Simple keyword-based fallback sentiment when MCP fails."""
    text_lower = email_text.lower()

    if any(word in text_lower for word in ["frustrated", "angry", "upset", "disappointed", "urgent", "emergency",
                                            "asap", "immediate", "critical", "unacceptable"]):
        sentiment = "negative"
    elif any(word in text_lower for word in ["thank", "grateful", "excellent", "appreciate", "satisfied",
                                              "wonderful", "great", "happy", "pleased"]):
        sentiment = "positive"
    else:
        sentiment = "neutral"

    return {"sentiment": sentiment, "source": "fallback"}


# ══════════════════════════════════════════════════════════════════════════════
# SCALAR UI (Custom API docs)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Public health check endpoint (no authentication required)."""
    return HealthResponse(status="ok")


@app.get("/docs", include_in_schema=False, response_class=HTMLResponse)
async def scalar_html():
    """Serve Scalar API documentation UI."""
    return """
<!doctype html>
<html>
  <head>
    <title>Email Intelligence Agent API</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
  </head>
  <body>
    <script
      id="api-reference"
      data-url="/openapi.json"
      data-configuration='{"theme": "purple"}'
    ></script>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
  </body>
</html>
""", media_type="text/html"


# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/login", response_model=Token)
def login(login_request: LoginRequest) -> Token:
    """Login endpoint to obtain OAuth 2.0 Bearer token."""
    user = authenticate_user(login_request.username, login_request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


# ══════════════════════════════════════════════════════════════════════════════
# CORE EMAIL ROUTES (no sentiment)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/analyze-email", response_model=EmailIntelligenceResponse)
def analyze_email(
    request: EmailIntelligenceRequest,
    config: AppConfig = Depends(get_config),
    current_user: User = Depends(get_current_user),
) -> EmailIntelligenceResponse:
    """
    Analyze email content provided as structured fields or raw content.
    Returns: summary, intent, entities, priority (no sentiment).
    Requires OAuth 2.0 authentication.
    """
    request_id = str(uuid.uuid4())
    user_id = current_user.username

    result = process_email_request(request, config, request_id=request_id, user_id=user_id)

    log_step(
        request_id=request_id,
        step_name="final_response_ready",
        output_data={"summary": result.summary, "priority": result.priority.level},
        status="success",
        user_id=user_id
    )

    return result


@app.post("/analyze-email-upload", response_model=EmailIntelligenceResponse)
async def analyze_email_upload(
    file: UploadFile = File(...),
    client_id: str = Form(..., description="Horizon API Client ID (REQUIRED)"),
    client_secret: str = Form(..., description="Horizon API Client Secret (REQUIRED)"),
    config: AppConfig = Depends(get_config),
    current_user: User = Depends(get_current_user),
) -> EmailIntelligenceResponse:
    """
    Analyze an uploaded email file (.eml or .msg format).
    Returns: summary, intent, entities, priority (no sentiment).
    Requires OAuth 2.0 authentication.
    """
    from app.services.email_intelligence import _convert_message_to_email_request, process_email_intelligence
    from email.parser import BytesParser
    from email import policy

    request_id = str(uuid.uuid4())
    user_id = current_user.username

    file_bytes = await file.read()

    message = BytesParser(policy=policy.default).parsebytes(file_bytes)
    email_request = _convert_message_to_email_request(message, client_id, client_secret)

    result = process_email_intelligence(email_request, config, request_id=request_id, user_id=user_id)

    log_step(
        request_id=request_id,
        step_name="final_response_ready",
        output_data={"summary": result.summary, "priority": result.priority.level},
        status="success",
        user_id=user_id
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT-ONLY ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/analyze-sentiment")
def analyze_sentiment_only(
    request: EmailIntelligenceRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Analyze only the sentiment of email content using MCP server.
    Returns: sentiment analysis only (no intelligence).
    Requires OAuth 2.0 authentication.
    """
    from app.services.mcp_helper import analyze_sentiment_sync

    request_id = str(uuid.uuid4())
    user_id = current_user.username

    email_text = prepare_email_text(request.subject or "", request.body or "")

    log_step(
        request_id=request_id,
        step_name="mcp_sentiment_request_initiated",
        input_data={"email_text_length": len(email_text), "subject": request.subject},
        status="success",
        user_id=user_id
    )

    start_time = time.time()
    try:
        sentiment_data = analyze_sentiment_sync(
            text=email_text,
            context="email",
            fallback_on_error=True
        )

        sentiment_duration = int((time.time() - start_time) * 1000)

        log_step(
            request_id=request_id,
            step_name="mcp_sentiment_analysis_complete",
            input_data={"email_text_length": len(email_text)},
            output_data=sentiment_data,
            status="success",
            duration_ms=sentiment_duration,
            user_id=user_id
        )

        return sentiment_data

    except Exception as e:
        log_step(
            request_id=request_id,
            step_name="mcp_sentiment_analysis_complete",
            status="error",
            error_message=f"MCP server error: {str(e)}",
            user_id=user_id
        )

        return get_fallback_sentiment(email_text)


@app.post("/analyze-sentiment-upload")
async def analyze_sentiment_upload(
    file: UploadFile = File(...),
    client_id: str = Form(..., description="Horizon API Client ID (REQUIRED)"),
    client_secret: str = Form(..., description="Horizon API Client Secret (REQUIRED)"),
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Analyze an uploaded email file AND get sentiment from MCP server.
    Returns: full intelligence + sentiment analysis (if needed).
    Requires OAuth 2.0 authentication.
    """
    from app.services.email_intelligence import _get_email_body
    from app.services.mcp_helper import analyze_sentiment_async
    from email.parser import BytesParser
    from email import policy

    request_id = str(uuid.uuid4())
    user_id = current_user.username

    file_bytes = await file.read()
    message = BytesParser(policy=policy.default).parsebytes(file_bytes)

    subject = message.get("subject", "") or ""
    body = _get_email_body(message)
    email_text = prepare_email_text(subject, body)

    log_step(
        request_id=request_id,
        step_name="mcp_sentiment_request_initiated",
        input_data={"email_text_length": len(email_text), "subject": subject, "file_name": file.filename},
        status="success",
        user_id=user_id
    )

    start_time = time.time()
    try:
        import asyncio
        sentiment_data = asyncio.run(analyze_sentiment_async(
            text=email_text,
            context="email"
        ))

        sentiment_duration = int((time.time() - start_time) * 1000)

        log_step(
            request_id=request_id,
            step_name="mcp_sentiment_analysis_complete",
            input_data={"email_text_length": len(email_text)},
            output_data=sentiment_data,
            status="success",
            duration_ms=sentiment_duration,
            user_id=user_id
        )

        return sentiment_data

    except Exception as e:
        log_step(
            request_id=request_id,
            step_name="mcp_sentiment_analysis_complete",
            status="error",
            error_message=f"MCP server error: {str(e)}",
            user_id=user_id
        )

        return get_fallback_sentiment(email_text)


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED ROUTES (intelligence + sentiment)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/analyze-email-with-sentiment")
def analyze_email_with_sentiment(
    request: EmailIntelligenceRequest,
    config: AppConfig = Depends(get_config),
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Analyze email content AND intelligently decide if sentiment is needed.
    Returns: full intelligence + sentiment analysis (if needed).
    Requires OAuth 2.0 authentication.
    """
    from app.services.mcp_helper import analyze_sentiment_sync

    request_id = str(uuid.uuid4())
    user_id = current_user.username

    # Get email intelligence
    email_analysis = process_email_request(request, config, request_id=request_id, user_id=user_id)

    # Prepare email text for sentiment
    email_text = prepare_email_text(request.subject or "", request.body or "")

    # STEP 1: Decide if sentiment analysis is needed
    decision_start = time.time()
    decision = should_analyze_sentiment(
        email_text=email_text,
        summary=email_analysis.summary,
        priority=email_analysis.priority.level,
        client_id=request.client_id,
        client_secret=request.client_secret,
    )
    decision_duration = int((time.time() - decision_start) * 1000)

    log_step(
        request_id=request_id,
        step_name="sentiment_decision_made",
        input_data={"email_text_length": len(email_text)},
        output_data=decision,
        status="success",
        duration_ms=decision_duration,
        user_id=user_id
    )

    # STEP 2: Call MCP or skip based on decision
    if decision.get("decision") == "call_mcp":
        start_time = time.time()
        try:
            sentiment_data = analyze_sentiment_sync(
                text=email_text,
                context="email",
                fallback_on_error=True
            )
            sentiment_data["decision_reason"] = decision.get("reason", "")

            sentiment_duration = int((time.time() - start_time) * 1000)

            log_step(
                request_id=request_id,
                step_name="mcp_sentiment_analysis_complete",
                input_data={"email_text_length": len(email_text)},
                output_data=sentiment_data,
                status="success",
                duration_ms=sentiment_duration,
                user_id=user_id
            )

        except Exception as e:
            sentiment_data = get_default_sentiment()
            sentiment_data["decision_reason"] = decision.get("reason", "")
            log_step(
                request_id=request_id,
                step_name="mcp_sentiment_analysis_complete",
                status="error",
                error_message=f"MCP server error: {str(e)}",
                user_id=user_id
            )

    else:
        # Skip MCP
        sentiment_data = get_default_sentiment()
        sentiment_data["decision_reason"] = decision.get("reason", "Skipped MCP per decision")
        log_step(
            request_id=request_id,
            step_name="mcp_sentiment_skipped",
            output_data={"reason": decision.get("reason")},
            status="success",
            user_id=user_id
        )

    # Combine results
    result = email_analysis.dict()
    result["sentiment_analysis"] = sentiment_data
    result["sentiment_decision"] = {
        "decision": decision.get("decision"),
        "reason": decision.get("reason"),
        "confidence": decision.get("confidence"),
    }

    return result


@app.post("/analyze-email-with-sentiment-upload")
async def analyze_email_with_sentiment_upload(
    file: UploadFile = File(...),
    client_id: str = Form(..., description="Horizon API Client ID (REQUIRED)"),
    client_secret: str = Form(..., description="Horizon API Client Secret (REQUIRED)"),
    config: AppConfig = Depends(get_config),
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Analyze an uploaded email file AND get sentiment from MCP server.
    Returns: full intelligence + sentiment analysis (if needed).
    Requires OAuth 2.0 authentication.
    """
    from app.services.mcp_helper import analyze_sentiment_async
    from app.services.email_intelligence import _convert_message_to_email_request, process_email_intelligence
    from email.parser import BytesParser
    from email import policy

    request_id = str(uuid.uuid4())
    user_id = current_user.username

    file_bytes = await file.read()
    message = BytesParser(policy=policy.default).parsebytes(file_bytes)
    email_request = _convert_message_to_email_request(message, client_id, client_secret)

    # Get email intelligence
    email_analysis = process_email_intelligence(email_request, config, request_id=request_id, user_id=user_id)

    email_text = prepare_email_text(email_request.subject or "", email_request.body or "")

    # STEP 1: Decide if sentiment needed
    decision = should_analyze_sentiment(
        email_text=email_text,
        summary=email_analysis.summary,
        priority=email_analysis.priority.level,
        client_id=client_id,
        client_secret=client_secret,
    )

    # STEP 2: Call MCP or skip
    if decision.get("decision") == "call_mcp":
        start_time = time.time()
        try:
            import asyncio
            sentiment_data = asyncio.run(analyze_sentiment_async(
                text=email_text,
                context="email"
            ))
            sentiment_data["decision_reason"] = decision.get("reason", "")

            sentiment_duration = int((time.time() - start_time) * 1000)

            log_step(
                request_id=request_id,
                step_name="mcp_sentiment_analysis_complete",
                input_data={"email_text_length": len(email_text)},
                output_data=sentiment_data,
                status="success",
                duration_ms=sentiment_duration,
                user_id=user_id
            )

        except Exception as e:
            sentiment_data = get_default_sentiment()
            sentiment_data["decision_reason"] = decision.get("reason", "")
            log_step(
                request_id=request_id,
                step_name="sentiment_analysis_complete",
                status="error",
                error_message=str(e),
                user_id=user_id
            )
    else:
        sentiment_data = get_default_sentiment()
        sentiment_data["decision_reason"] = decision.get("reason", "")

    result = email_analysis.dict()
    result["sentiment_analysis"] = sentiment_data
    result["sentiment_decision"] = {
        "decision": decision.get("decision"),
        "reason": decision.get("reason"),
        "confidence": decision.get("confidence"),
    }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# LOGS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/get-logs")
def get_logs(
    request_id: Optional[str] = None,
    step_name: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Retrieve agent processing logs with optional filtering."""
    logger = get_logger()

    if request_id:
        logs = logger.get_request_logs(request_id)
    else:
        logs = logger.get_logs(
            step_name=step_name,
            status=status_filter,
            limit=limit,
            offset=offset,
        )

    return {"logs": logs, "total": len(logs), "limit": limit, "offset": offset}


@app.get("/logs/{request_id}")
def get_request_logs(
    request_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Get all logs for a specific request ID.
    Requires OAuth 2.0 authentication.
    """
    logger = get_logger()
    logs = logger.get_request_logs(request_id)

    return {
        "request_id": request_id,
        "total_steps": len(logs),
        "logs": logs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL EXCEPTION HANDLER
# ══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler to ensure consistent error responses."""
    import traceback
    error_details = traceback.format_exc()
    print(f"Error occurred: {error_details}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An error occurred while processing the email intelligence request",
            "details": str(exc)
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
