"""
Agent Planner Module

Turns the email-intelligence pipeline into a declarative, ordered plan of
steps that the agent executes one at a time.

Previously the processing sequence (summarize + classify intent, extract
entities, assign priority, optional sentiment) was hard-coded inside
``process_email_intelligence``.  This module hoists that flow into a
``Planner`` so steps become:

* declarative      - described as data (:class:`PlanStep`) rather than inline code
* ordered          - executed in the order they are added to the plan
* conditional       - a step can be skipped at runtime via a ``condition`` predicate
* observable        - every step is timed and persisted through ``app.agent_logger``
* fault-tolerant    - a failing *optional* step is logged and skipped; a failing
                      *required* step aborts the plan

The planner keeps the exact step names already used by the service
(``email_input_received``, ``summarization_complete``,
``entity_extraction_complete``, ``priority_assignment_complete``) so existing
log consumers and dashboards keep working unchanged.

Planning / routing decision
---------------------------
On top of the original pipeline, the planner now performs the job the
architecture diagram assigns to the *Planner agent*: once the intent and
priority are known it decides **where the email should go next**.  This mirrors
the routing logic of the sibling ``e360-inquiry-categorization`` service:

* ``suggested_routing`` - the downstream PEGA queue / executor agent chosen from
  the intent via :data:`ROUTING_MAP` (e.g. ``"ID Card Change Request"`` ->
  ``"PEGA_ID_CARD_QUEUE"``).
* ``human_review_required`` - ``True`` for high-touch cases (``Urgent`` /
  ``Access to Care`` priority, or an ``Other`` / unknown intent).
* ``sla_hours`` - turnaround target derived from the priority via :data:`SLA_MAP`.

The decision is emitted as a new ``routing_complete`` log step and exposed on the
context/result.  The returned :class:`EmailIntelligenceResponse` is unchanged, so
existing callers keep working; callers that want the routing decision use
:meth:`Planner.run_plan` (or read :attr:`Planner.routing`).

Typical use::

    from app.planner import Planner

    planner = Planner(config, user_id="alice")
    response = planner.run(request)            # same response as before
    routing = planner.routing                  # new: where to send it next

    # or get both together:
    result = planner.run_plan(request)
    result.response, result.routing

That is a drop-in replacement for ``process_email_intelligence`` and emits the
same per-step logs, plus the new ``routing_complete`` step.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from app.agent_logger import log_step
from app.config import AppConfig

if TYPE_CHECKING:  # avoid importing pydantic / heavy LLM deps at module load
    from app.schemas import (
        EmailIntelligenceRequest,
        EmailIntelligenceResponse,
        ExtractedEntities,
        PriorityAssignment,
    )


# Status strings mirror those already understood by AgentLogger.
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"


# --------------------------------------------------------------------------- #
# Routing configuration
#
# Mirrors the ROUTING_MAP / SLA_MAP of the sibling e360-inquiry-categorization
# service, but keyed to the intents this assistant actually emits
# (see EmailIntelligenceResponse.intent in app.schemas).
# --------------------------------------------------------------------------- #
DEFAULT_QUEUE = "PEGA_GENERAL_QUEUE"

ROUTING_MAP: Dict[str, str] = {
    "ID Card Change Request": "PEGA_ID_CARD_QUEUE",
    "Address Change Request": "PEGA_ADDRESS_QUEUE",
    "General Enquiry": "PEGA_INQUIRY_QUEUE",
    "Other": DEFAULT_QUEUE,
}

# Turnaround target (hours) per priority level.
DEFAULT_SLA_HOURS = 24
SLA_MAP: Dict[str, int] = {
    "Urgent": 4,
    "Access to Care": 8,
    "Standard": 24,
}

# Priority levels that always warrant a human in the loop.
HUMAN_REVIEW_PRIORITIES = ("Urgent", "Access to Care")


def get_pega_queue(intent: Optional[str]) -> str:
    """Map an intent to its downstream PEGA queue / executor agent."""
    return ROUTING_MAP.get(intent or "", DEFAULT_QUEUE)


def get_sla_hours(priority_level: Optional[str]) -> int:
    """Return the SLA turnaround (hours) for a priority level."""
    return SLA_MAP.get(priority_level or "", DEFAULT_SLA_HOURS)


def needs_human_review(intent: Optional[str], priority_level: Optional[str]) -> bool:
    """Decide whether the case must be reviewed by a human.

    High-priority cases and unknown / catch-all intents are routed to a person.
    """
    return priority_level in HUMAN_REVIEW_PRIORITIES or intent in (None, "Other")


@dataclass
class RoutingDecision:
    """The planner's decision about where an analyzed email goes next."""

    suggested_routing: str
    human_review_required: bool
    sla_hours: int
    intent: Optional[str] = None
    priority_level: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "suggested_routing": self.suggested_routing,
            "human_review_required": self.human_review_required,
            "sla_hours": self.sla_hours,
            "intent": self.intent,
            "priority_level": self.priority_level,
        }


@dataclass
class PlanContext:
    """Mutable state shared between plan steps.

    A single context object is threaded through every step of a run.  Step
    handlers read the request/config and write their results back here so that
    later steps (and the final response assembly) can consume them.
    """

    request: "EmailIntelligenceRequest"
    config: AppConfig
    request_id: str
    user_id: Optional[str] = None

    # Filled in by steps as the plan progresses.
    summary: Optional[str] = None
    intent: Optional[str] = None
    entities: Optional["ExtractedEntities"] = None
    priority: Optional["PriorityAssignment"] = None
    sentiment: Optional[Dict[str, Any]] = None
    routing: Optional["RoutingDecision"] = None

    # Free-form scratch space for anything a custom step wants to stash.
    scratch: Dict[str, Any] = field(default_factory=dict)

    @property
    def client_id(self) -> str:
        return self.request.client_id

    @property
    def client_secret(self) -> str:
        return self.request.client_secret


# A step handler does the work and returns a dict used as the step's
# ``output_data`` in the log.  It receives and mutates the shared context.
StepHandler = Callable[[PlanContext], Dict[str, Any]]
# Optional helpers that derive log ``input_data`` and the skip ``condition``.
InputBuilder = Callable[[PlanContext], Dict[str, Any]]
Condition = Callable[[PlanContext], bool]


@dataclass
class PlanStep:
    """A single unit of work in a plan.

    Attributes:
        name: Logged ``step_name``. Keep stable - log consumers key off it.
        handler: Callable that performs the work and returns the log ``output_data``.
        input_builder: Optional callable returning the log ``input_data`` snapshot.
        condition: Optional predicate; when it returns ``False`` the step is skipped.
        required: If ``True`` an exception aborts the run; if ``False`` it is
            logged and the plan continues.
        description: Human-readable note describing what the step does.
    """

    name: str
    handler: StepHandler
    input_builder: Optional[InputBuilder] = None
    condition: Optional[Condition] = None
    required: bool = True
    description: str = ""


@dataclass
class StepOutcome:
    """Result of executing one step (returned for inspection/testing)."""

    name: str
    status: str
    duration_ms: int
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class PlanResult:
    """Full output of a planned run: the response plus the routing decision."""

    response: "EmailIntelligenceResponse"
    routing: Optional[RoutingDecision]
    outcomes: List[StepOutcome] = field(default_factory=list)


class Planner:
    """Builds and executes an ordered plan of :class:`PlanStep` objects.

    The default plan reproduces the existing email-intelligence pipeline.  The
    plan is data, so callers can inspect it, reorder it, drop steps, or append
    their own before calling :meth:`run`.
    """

    def __init__(
        self,
        config: AppConfig,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        self.config = config
        self.request_id = request_id or str(uuid.uuid4())
        self.user_id = user_id
        self.steps: List[PlanStep] = []
        self.outcomes: List[StepOutcome] = []
        self.routing: Optional[RoutingDecision] = None
        self._last_context: Optional[PlanContext] = None

    # ------------------------------------------------------------------ #
    # Plan construction
    # ------------------------------------------------------------------ #
    def add_step(self, step: PlanStep) -> "Planner":
        """Append a step. Returns ``self`` so calls can be chained."""
        self.steps.append(step)
        return self

    def build_default_plan(self) -> "Planner":
        """Populate ``self.steps`` with the standard email-intelligence flow.

        Steps mirror ``process_email_intelligence`` exactly, with the optional
        sentiment step guarded by ``config.enable_sentiment_integration``.
        Heavy handlers are imported lazily inside :meth:`run` to keep this
        module import-light.
        """
        self.steps = []

        self.add_step(
            PlanStep(
                name="email_input_received",
                description="Snapshot the inbound email request.",
                handler=self._step_input_received,
                input_builder=lambda ctx: {
                    "subject": ctx.request.subject,
                    "body_length": len(ctx.request.body) if ctx.request.body else 0,
                    "from_email": ctx.request.from_email,
                },
            )
        )
        self.add_step(
            PlanStep(
                name="summarization_complete",
                description="Generate a summary and classify the email intent via LLM.",
                handler=self._step_summarize,
                input_builder=lambda ctx: {
                    "subject": ctx.request.subject,
                    "body_length": len(ctx.request.body) if ctx.request.body else 0,
                },
            )
        )
        self.add_step(
            PlanStep(
                name="entity_extraction_complete",
                description="Extract structured entities (HCID, dates, amounts, ...) via LLM.",
                handler=self._step_extract_entities,
                input_builder=lambda ctx: {"summary": ctx.summary},
            )
        )
        self.add_step(
            PlanStep(
                name="priority_assignment_complete",
                description="Assign a priority level with explanation via LLM.",
                handler=self._step_assign_priority,
                input_builder=lambda ctx: {"summary": ctx.summary},
            )
        )
        self.add_step(
            PlanStep(
                name="routing_complete",
                description="Plan the downstream route (queue/agent), human review and SLA.",
                handler=self._step_route,
                input_builder=lambda ctx: {
                    "intent": ctx.intent,
                    "priority_level": ctx.priority.level if ctx.priority else None,
                },
            )
        )
        self.add_step(
            PlanStep(
                name="sentiment_analysis_complete",
                description="Optional sentiment analysis via the MCP server.",
                handler=self._step_sentiment,
                input_builder=lambda ctx: {
                    "subject": ctx.request.subject,
                    "body_length": len(ctx.request.body) if ctx.request.body else 0,
                },
                # Conditional + non-fatal: skipped unless enabled, never aborts the run.
                condition=lambda ctx: ctx.config.enable_sentiment_integration,
                required=False,
            )
        )
        return self

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def run(self, request: "EmailIntelligenceRequest") -> "EmailIntelligenceResponse":
        """Execute the plan for ``request`` and assemble the response.

        If no steps have been added yet, the default plan is built first.
        """
        if not self.steps:
            self.build_default_plan()

        ctx = PlanContext(
            request=request,
            config=self.config,
            request_id=self.request_id,
            user_id=self.user_id,
        )

        if self.config.include_debug_info:
            print("Debug Planner.run start:", request)

        for step in self.steps:
            self._execute_step(step, ctx)

        self._last_context = ctx
        self.routing = ctx.routing

        if self.config.include_debug_info:
            print(
                "Debug Planner.run results:",
                ctx.summary,
                ctx.entities,
                ctx.priority,
                ctx.routing,
            )

        return self._assemble_response(ctx)

    def run_plan(self, request: "EmailIntelligenceRequest") -> "PlanResult":
        """Execute the plan and return the response *plus* the routing decision.

        Use this when you need the planner's routing/human-review/SLA output;
        :meth:`run` stays a drop-in returning only the response.
        """
        response = self.run(request)
        return PlanResult(
            response=response,
            routing=self.routing,
            outcomes=list(self.outcomes),
        )

    def _execute_step(self, step: PlanStep, ctx: PlanContext) -> StepOutcome:
        """Run a single step with conditioning, timing, logging and error policy."""
        # Conditional skip.
        if step.condition is not None and not step.condition(ctx):
            outcome = StepOutcome(name=step.name, status=STATUS_SKIPPED, duration_ms=0)
            self.outcomes.append(outcome)
            return outcome

        input_data = step.input_builder(ctx) if step.input_builder else None

        start = time.time()
        try:
            output_data = step.handler(ctx) or {}
            duration_ms = int((time.time() - start) * 1000)
            log_step(
                request_id=ctx.request_id,
                step_name=step.name,
                input_data=input_data,
                output_data=output_data,
                status=STATUS_SUCCESS,
                duration_ms=duration_ms,
                user_id=ctx.user_id,
            )
            outcome = StepOutcome(
                name=step.name,
                status=STATUS_SUCCESS,
                duration_ms=duration_ms,
                output_data=output_data,
            )
            self.outcomes.append(outcome)
            return outcome

        except Exception as exc:  # noqa: BLE001 - we deliberately log and decide policy
            duration_ms = int((time.time() - start) * 1000)
            log_step(
                request_id=ctx.request_id,
                step_name=step.name,
                input_data=input_data,
                status=STATUS_ERROR,
                error_message=str(exc),
                duration_ms=duration_ms,
                user_id=ctx.user_id,
            )
            outcome = StepOutcome(
                name=step.name,
                status=STATUS_ERROR,
                duration_ms=duration_ms,
                error_message=str(exc),
            )
            self.outcomes.append(outcome)
            if step.required:
                raise
            return outcome

    # ------------------------------------------------------------------ #
    # Default step handlers (lazy-import heavy deps so the module stays light)
    # ------------------------------------------------------------------ #
    def _step_input_received(self, ctx: PlanContext) -> Dict[str, Any]:
        return {
            "subject": ctx.request.subject,
            "body_length": len(ctx.request.body) if ctx.request.body else 0,
            "from_email": ctx.request.from_email,
        }

    def _step_summarize(self, ctx: PlanContext) -> Dict[str, Any]:
        from app.services.email_intelligence import _generate_email_summary

        summary, intent = _generate_email_summary(
            ctx.request.subject,
            ctx.request.body,
            ctx.config,
            ctx.client_id,
            ctx.client_secret,
        )
        ctx.summary = summary
        ctx.intent = intent
        return {"summary": summary, "intent": intent}

    def _step_extract_entities(self, ctx: PlanContext) -> Dict[str, Any]:
        from app.services.email_intelligence import _extract_entities

        entities = _extract_entities(
            ctx.request,
            ctx.summary,
            ctx.config,
            ctx.client_id,
            ctx.client_secret,
        )
        ctx.entities = entities
        return entities.dict()

    def _step_assign_priority(self, ctx: PlanContext) -> Dict[str, Any]:
        from app.services.email_intelligence import _assign_priority

        priority = _assign_priority(
            ctx.request.subject,
            ctx.request.body,
            ctx.summary,
            ctx.config,
            ctx.client_id,
            ctx.client_secret,
        )
        ctx.priority = priority
        return priority.dict()

    def _step_route(self, ctx: PlanContext) -> Dict[str, Any]:
        """Decide the downstream route, human-review flag and SLA from intent+priority."""
        priority_level = ctx.priority.level if ctx.priority else None
        decision = RoutingDecision(
            suggested_routing=get_pega_queue(ctx.intent),
            human_review_required=needs_human_review(ctx.intent, priority_level),
            sla_hours=get_sla_hours(priority_level),
            intent=ctx.intent,
            priority_level=priority_level,
        )
        ctx.routing = decision
        return decision.as_dict()

    def _step_sentiment(self, ctx: PlanContext) -> Dict[str, Any]:
        from app.services.mcp_helper import analyze_email_sentiment_sync

        sentiment = analyze_email_sentiment_sync(
            subject=ctx.request.subject or "",
            body=ctx.request.body or "",
            fallback_on_error=True,
        )
        ctx.sentiment = sentiment
        return sentiment

    # ------------------------------------------------------------------ #
    # Response assembly
    # ------------------------------------------------------------------ #
    def _assemble_response(self, ctx: PlanContext) -> "EmailIntelligenceResponse":
        from app.schemas import EmailIntelligenceResponse

        if ctx.summary is None or ctx.intent is None:
            raise RuntimeError(
                "Planner finished without a summary/intent - the "
                "'summarization_complete' step did not run."
            )
        if ctx.entities is None:
            raise RuntimeError(
                "Planner finished without entities - the "
                "'entity_extraction_complete' step did not run."
            )
        if ctx.priority is None:
            raise RuntimeError(
                "Planner finished without a priority - the "
                "'priority_assignment_complete' step did not run."
            )

        return EmailIntelligenceResponse(
            summary=ctx.summary,
            intent=ctx.intent,
            extracted_entities=ctx.entities,
            priority=ctx.priority,
            original_subject=ctx.request.subject,
            original_body=ctx.request.body,
        )


def plan_and_process(
    request: "EmailIntelligenceRequest",
    config: AppConfig,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> "EmailIntelligenceResponse":
    """Drop-in replacement for ``process_email_intelligence``.

    Builds a :class:`Planner` with the default plan and runs it, producing the
    same per-step logs and the same ``EmailIntelligenceResponse``.
    """
    return Planner(config, request_id=request_id, user_id=user_id).run(request)


def plan_and_route(
    request: "EmailIntelligenceRequest",
    config: AppConfig,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> "PlanResult":
    """Like :func:`plan_and_process` but also returns the routing decision."""
    return Planner(config, request_id=request_id, user_id=user_id).run_plan(request)
