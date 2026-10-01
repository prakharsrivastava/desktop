"""
models.py — Pydantic models for e360 Inquiry Categorization
============================================================
Project B: e360-inquiry-categorization

Architecture diagram ke hisaab se:
    Kafka → Orchestrator → Inquiry Agent → SOA endpoints → Response → PEGA
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ══════════════════════════════════════════════════════════════════════════════

class InquiryCategoryRequest(BaseModel):
    """
    Inquiry categorization ka input.
    Kafka se aata hai via Orchestrator.
    """
    inquiry_id: str = Field(..., description="Unique inquiry identifier")
    subject: Optional[str] = Field(None, description="Email/inquiry subject")
    body: Optional[str] = Field(None, description="Full inquiry text")
    from_email: Optional[str] = Field(None, description="Sender email")
    channel: Optional[str] = Field(
        "email", description="Ingestion channel: email, portal, phone"
    )
    hcid: Optional[str] = Field(None, description="Healthcare ID if known")
    case_number: Optional[str] = Field(None, description="Existing case number")
    client_id: str = Field(..., description="Horizon API client ID (REQUIRED)")
    client_secret: str = Field(..., description="Horizon API client secret (REQUIRED)")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Extra fields from Kafka message"
    )


class EmailIntelligenceRequest(BaseModel):
    """
    Reused from Project A — same shape, used for email uploads.
    """
    subject: Optional[str] = None
    body: Optional[str] = None
    from_email: Optional[str] = None
    to_emails: Optional[List[str]] = None
    cc_emails: Optional[List[str]] = None
    sent_at: Optional[datetime] = None
    client_id: str
    client_secret: str


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class InquiryCategory(BaseModel):
    """Classified inquiry category."""
    primary: Literal[
        "ID Card Change Request",
        "Address Change Request",
        "General Enquiry",
        "Billing Inquiry",
        "Claims Inquiry",
        "Benefits Inquiry",
        "Provider Inquiry",
        "Other",
    ]
    secondary: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class ExtractedInquiryEntities(BaseModel):
    """Entities extracted from inquiry."""
    hcid: List[str] = Field(default_factory=list)
    policy_numbers: List[str] = Field(default_factory=list)
    case_numbers: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    amounts: List[str] = Field(default_factory=list)
    phone_numbers: List[str] = Field(default_factory=list)
    addresses: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    provider_names: List[str] = Field(default_factory=list)


class InquiryPriority(BaseModel):
    """Priority assignment."""
    level: Literal["Standard", "Access to Care", "Urgent"]
    explanation: str
    sla_hours: int = Field(
        description="SLA hours for this priority level"
    )


class InquiryCategoryResponse(BaseModel):
    """
    Full response — jaata hai Response Agent → PEGA.
    """
    inquiry_id: str
    summary: str
    category: InquiryCategory
    entities: ExtractedInquiryEntities
    priority: InquiryPriority
    suggested_routing: str = Field(
        description="Which PEGA queue/workflow to route to"
    )
    human_review_required: bool = False
    processing_time_ms: Optional[int] = None
    original_subject: Optional[str] = None
    original_body: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str = "e360-inquiry-categorization"
    version: str = "1.0.0"
