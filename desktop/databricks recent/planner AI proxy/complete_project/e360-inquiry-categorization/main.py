"""
main.py — FastAPI App for e360 Inquiry Categorization
======================================================
Architecture mein:
    PEGA → Kafka → Orchestrator → [THIS SERVICE] → Response Agent → PEGA

Routes:
    POST /categorize          — Structured inquiry categorize karo
    POST /upload              — .eml / .msg file upload karo
    GET  /health              — Health check
    POST /token               — JWT login
"""

from __future__ import annotations

import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query, status
from fastapi.middleware.cors import CORSMiddleware

from models import (
    InquiryCategoryRequest,
    InquiryCategoryResponse,
    EmailIntelligenceRequest,
    HealthResponse,
)
from util import process_inquiry
from logger import setup_logging, get_context_logger
from set_ssl_env import configure_ssl
from get_util import generate_request_id

# Auth reuse from project A pattern
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import timedelta
from jose import jwt, JWTError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hashlib

SECRET_KEY  = os.environ.get("JWT_SECRET", "e360-secret-change-in-prod")
ALGORITHM   = "HS256"
security    = HTTPBearer()

log = logging.getLogger(__name__)


# ── Startup ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        json_format=os.environ.get("LOG_FORMAT", "text") == "json",
    )
    configure_ssl()
    log.info("e360 Inquiry Categorization Service starting...")
    yield
    log.info("e360 Service shutting down.")


app = FastAPI(
    title="e360 Inquiry Categorization",
    description="AI-powered inquiry classification for health insurance operations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ── Simple auth ───────────────────────────────────────────────────────────────

USERS = {
    "admin": hashlib.sha256(b"admin123").hexdigest(),
    "e360":  hashlib.sha256(b"e360pass").hexdigest(),
}


async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    try:
        payload  = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")


@app.post("/token", tags=["Auth"])
async def login(username: str, password: str):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    if USERS.get(username) != hashed:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    import datetime
    token = jwt.encode(
        {"sub": username, "exp": datetime.datetime.utcnow() + timedelta(hours=8)},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    return {"access_token": token, "token_type": "bearer"}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(status="ok")


# ── Core routes ───────────────────────────────────────────────────────────────

@app.post(
    "/categorize",
    response_model=InquiryCategoryResponse,
    tags=["Inquiry"],
)
async def categorize(
    request: InquiryCategoryRequest,
    current_user: str = Depends(get_current_user),
):
    """
    Inquiry categorize karo.

    Input: inquiry_id, subject, body, client_id, client_secret
    Output: category, entities, priority, PEGA routing
    """
    request_id = generate_request_id("e360")
    clog = get_context_logger(__name__, request_id=request_id,
                              inquiry_id=request.inquiry_id, user_id=current_user)
    clog.info("Categorize request received")

    try:
        response = process_inquiry(request, request_id=request_id)
        clog.info(f"Categorized as: {response.category.primary} | Priority: {response.priority.level}")
        return response
    except Exception as exc:
        clog.error(f"Categorization failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Categorization failed: {str(exc)}",
        )


@app.post(
    "/upload",
    response_model=InquiryCategoryResponse,
    tags=["Inquiry"],
)
async def upload_email(
    file: UploadFile = File(...),
    client_id: str = Query(...),
    client_secret: str = Query(...),
    current_user: str = Depends(get_current_user),
):
    """
    .eml ya .msg file upload karke categorize karo.
    """
    request_id = generate_request_id("e360-upload")
    clog = get_context_logger(__name__, request_id=request_id, user_id=current_user)
    clog.info(f"Upload request: {file.filename}")

    if not client_id or not client_secret:
        raise HTTPException(status_code=400, detail="client_id and client_secret required")

    try:
        file_bytes = await file.read()

        # Parse email file — reuse Project A logic
        from email import policy as email_policy
        from email.parser import BytesParser

        message = BytesParser(policy=email_policy.default).parsebytes(file_bytes)
        subject = message.get("subject", "") or ""
        body    = ""
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body += part.get_content()
                    except Exception:
                        pass
        else:
            try:
                body = message.get_content() or ""
            except Exception:
                pass

        inquiry_request = InquiryCategoryRequest(
            inquiry_id=request_id,
            subject=subject,
            body=body,
            channel="email_upload",
            client_id=client_id,
            client_secret=client_secret,
        )
        response = process_inquiry(inquiry_request, request_id=request_id)
        return response

    except Exception as exc:
        clog.error(f"Upload processing failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload processing failed: {str(exc)}",
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
