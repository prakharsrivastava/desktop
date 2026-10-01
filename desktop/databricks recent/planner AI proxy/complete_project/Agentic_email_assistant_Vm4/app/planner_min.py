"""Minimal Planner: runs the email pipeline as ordered steps, then routes."""

import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

from app.agent_logger import log_step

ROUTING_MAP = {
    "ID Card Change Request": "PEGA_ID_CARD_QUEUE",
    "Address Change Request": "PEGA_ADDRESS_QUEUE",
    "General Enquiry": "PEGA_INQUIRY_QUEUE",
}
SLA_MAP = {"Urgent": 4, "Access to Care": 8, "Standard": 24}


@dataclass
class PlanStep:
    name: str
    handler: Callable          # takes ctx (dict), returns output dict
    required: bool = True


class Planner:
    def __init__(self, config, request_id=None, user_id=None):
        self.config = config
        self.request_id = request_id or str(uuid.uuid4())
        self.user_id = user_id

    def run(self, request):
        ctx = {"request": request}
        steps = [
            PlanStep("summarization_complete", self._summarize),
            PlanStep("entity_extraction_complete", self._extract_entities),
            PlanStep("priority_assignment_complete", self._assign_priority),
            PlanStep("routing_complete", self._route),
        ]
        for step in steps:
            start = time.time()
            try:
                out = step.handler(ctx) or {}
                log_step(self.request_id, step.name, output_data=out,
                         status="success",
                         duration_ms=int((time.time() - start) * 1000),
                         user_id=self.user_id)
            except Exception as e:
                log_step(self.request_id, step.name, status="error",
                         error_message=str(e), user_id=self.user_id)
                if step.required:
                    raise
        return self._build_response(ctx)

    # --- steps ---
    def _summarize(self, ctx):
        from app.services.email_intelligence import _generate_email_summary
        r = ctx["request"]
        ctx["summary"], ctx["intent"] = _generate_email_summary(
            r.subject, r.body, self.config, r.client_id, r.client_secret)
        return {"summary": ctx["summary"], "intent": ctx["intent"]}

    def _extract_entities(self, ctx):
        from app.services.email_intelligence import _extract_entities
        r = ctx["request"]
        ctx["entities"] = _extract_entities(
            r, ctx["summary"], self.config, r.client_id, r.client_secret)
        return ctx["entities"].dict()

    def _assign_priority(self, ctx):
        from app.services.email_intelligence import _assign_priority
        r = ctx["request"]
        ctx["priority"] = _assign_priority(
            r.subject, r.body, ctx["summary"], self.config,
            r.client_id, r.client_secret)
        return ctx["priority"].dict()

    def _route(self, ctx):
        intent = ctx.get("intent")
        level = ctx["priority"].level if ctx.get("priority") else None
        ctx["routing"] = {
            "queue": ROUTING_MAP.get(intent, "PEGA_GENERAL_QUEUE"),
            "human_review": level in ("Urgent", "Access to Care") or intent in (None, "Other"),
            "sla_hours": SLA_MAP.get(level, 24),
        }
        return ctx["routing"]

    def _build_response(self, ctx):
        from app.schemas import EmailIntelligenceResponse
        r = ctx["request"]
        return EmailIntelligenceResponse(
            summary=ctx["summary"],
            intent=ctx["intent"],
            extracted_entities=ctx["entities"],
            priority=ctx["priority"],
            original_subject=r.subject,
            original_body=r.body,
        )
