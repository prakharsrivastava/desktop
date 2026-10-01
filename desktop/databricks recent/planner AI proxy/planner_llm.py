"""LLM-prompt-based Planner: ek LLM decide karta hai kaunse executor agent chalein.
Minimum change from the rule-based version — sirf planning step ab prompt-based hai."""

import json
import time
import uuid

from horizon_langchain import HorizonLlmChat, TokenManager

from app.agent_logger import log_step
from app.schemas import EmailIntelligenceResponse
from app.services.email_intelligence import (
    _generate_email_summary, _extract_entities, _assign_priority,
)

# Architecture diagram ke "config" block ke executor agents
AVAILABLE_AGENTS = ["id_card_agent", "address_change_agent", "inquiry_agent"]

PLANNER_PROMPT = """You are the Planner agent in an email-processing system.
Choose which executor agent(s) should handle the email.

Available agents:
- id_card_agent: new or replacement ID card requests
- address_change_agent: member address changes
- inquiry_agent: general questions / anything else

Email summary: {summary}
Detected intent: {intent}
Priority: {priority}

Return ONLY a JSON object (no markdown, no extra text), exactly like:
{{"agents": ["address_change_agent"], "human_review": false, "reason": "short reason"}}
Set "human_review" to true if priority is Urgent or Access to Care, or if you are unsure."""


class Planner:
    def __init__(self, config, request_id=None, user_id=None):
        self.config, self.user_id = config, user_id
        self.request_id = request_id or str(uuid.uuid4())

    def _log(self, name, start, out):
        log_step(self.request_id, name, output_data=out, status="success",
                 duration_ms=int((time.time() - start) * 1000), user_id=self.user_id)

    def run(self, r):
        t = time.time()
        summary, intent = _generate_email_summary(r.subject, r.body, self.config, r.client_id, r.client_secret)
        self._log("summarization_complete", t, {"intent": intent})

        t = time.time()
        entities = _extract_entities(r, summary, self.config, r.client_id, r.client_secret)
        self._log("entity_extraction_complete", t, entities.dict())

        t = time.time()
        priority = _assign_priority(r.subject, r.body, summary, self.config, r.client_id, r.client_secret)
        self._log("priority_assignment_complete", t, priority.dict())

        t = time.time()
        self.plan = self._plan(summary, intent, priority.level, r.client_id, r.client_secret)
        self._log("planning_complete", t, self.plan)

        return EmailIntelligenceResponse(
            summary=summary, intent=intent, extracted_entities=entities,
            priority=priority, original_subject=r.subject, original_body=r.body)

    def _plan(self, summary, intent, priority, client_id, client_secret):
        """LLM se poochо kaunse executor agent chalein (rule-based routing ki jagah)."""
        llm = HorizonLlmChat(
            api_endpoint="https://api.horizon.elevancehealth.com/v2/text/chats",
            token_manager=TokenManager(client_id=client_id, client_secret=client_secret),
        )
        prompt = PLANNER_PROMPT.format(summary=summary, intent=intent, priority=priority)
        raw = llm.invoke(prompt).content.strip()
        if raw.startswith("```"):                      # ```json ... ``` saaf karo
            raw = raw.strip("`").replace("json", "", 1).strip()
        try:
            plan = json.loads(raw)
        except Exception:
            plan = {"agents": ["inquiry_agent"], "human_review": True,
                    "reason": "fallback: LLM plan parse nahi hua"}
        plan["agents"] = [a for a in plan.get("agents", []) if a in AVAILABLE_AGENTS] or ["inquiry_agent"]
        return plan
