from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, EmailStr, Field


class EmailIntelligenceRequest(BaseModel):
    """Request model for email intelligence analysis."""

    subject: Optional[str] = Field(None, description="Email subject line")
    body: Optional[str] = Field(None, description="Full email body text")
    from_email: Optional[EmailStr] = Field(
        None, description="Sender email address"
    )
    to_emails: Optional[List[EmailStr]] = Field(
        None, description="Recipient email addresses"
    )
    cc_emails: Optional[List[EmailStr]] = Field(
        None, description="CC email addresses"
    )
    sent_at: Optional[datetime] = Field(
        None, description="Datetime when the email was sent"
    )
    thread_id: Optional[str] = Field(
        None, description="Conversation or thread identifier"
    )
    raw_content: Optional[str] = Field(
        None, description="Raw email content in .eml format"
    )
    attachments: Optional[List[str]] = Field(
        None, description="List of attachment names"
    )
    client_id: str = Field(
        ..., description="Horizon API client ID for authentication (REQUIRED)"
    )
    client_secret: str = Field(
        ..., description="Horizon API client secret for authentication (REQUIRED)"
    )


class ExtractedEntities(BaseModel):
    """Structured entities extracted from an email."""

    sender_email: Optional[EmailStr] = None
    recipient_emails: List[EmailStr] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    amounts: List[str] = Field(default_factory=list)
    company_names: List[str] = Field(default_factory=list)
    hcid: List[str] = Field(default_factory=list)
    policy_numbers: List[str] = Field(default_factory=list)
    case_numbers: List[str] = Field(default_factory=list)
    phone_numbers: List[str] = Field(default_factory=list)
    addresses: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)


class PriorityAssignment(BaseModel):
    """Priority assignment with explanation."""

    level: Literal["Standard", "Access to Care", "Urgent"]
    explanation: str


class EmailIntelligenceResponse(BaseModel):
    """Response returned after email intelligence analysis."""

    summary: str
    intent: Literal["ID Card Change Request", "Address Change Request", "General Enquiry", "Other"]
    extracted_entities: ExtractedEntities
    priority: PriorityAssignment
    original_subject: Optional[str] = None
    original_body: Optional[str] = None


class HealthResponse(BaseModel):
    """Simple health check response."""

    status: str
