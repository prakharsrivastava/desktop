"""
agentic_workflow.workflow_state

Reconstructed from screen recording 2026-06-11_12-53-02.mp4 (seen only briefly,
not at full zoom). Confidence: MEDIUM/LOW for exact field set & types. The class
name `class WorkflowState(BaseModel):` and the presence of the fields below were
visible; precise defaults/types should be verified against the real file.

This is the LangGraph-style shared state object threaded through the agentic
workflow (mirrors the WorkflowState seen in the e360-marketplace / inquiry
categorization projects).
"""

from datetime import datetime
from typing import Annotated, Any, Optional

from pydantic import BaseModel

# from langgraph.graph.message import add_messages  # seen usage: Annotated[..., add_messages]


def add_messages(left, right):  # placeholder if langgraph isn't importable
    return (left or []) + (right or [])


class WorkflowState(BaseModel):
    # --- fields visible in the recording (types/defaults approximate) ---
    inquiry_categorization_inputs: Optional[dict[str, Any]] = None
    email_analysis_data: Optional[dict[str, Any]] = None
    intent: Optional[str] = None
    preprocessor_status: Optional[str] = None
    entity_extraction_status: Optional[str] = None
    agents: list[str] = []
    unsupported_entities: list[str] = []

    task: Optional["TaskState"] = None  # noqa: F821 - TaskState defined elsewhere
    auto_close: bool = False

    # --- timing fields ---
    initial_query_received_at: Optional[datetime] = None
    human_feedback_received_at: Optional[datetime] = None

    # --- audit data ---
    audit_data: Optional[dict[str, Any]] = None

    # --- conversation history ---
    messages: Annotated[list[str], add_messages] = []
