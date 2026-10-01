"""
usecases.base  -  E360UseCaseAgent

Reconstructed from screen recording 2026-06-11_13-04-14.mp4.
Confidence: MEDIUM-HIGH for the attributes and methods shown (agent_name,
steps_completed, logger, log_step, requires_human_approval). The full method
set of this ABC base was partly scrolled; faithful to what was visible.
"""

from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import TypedDict

# Seen import in the recording:
# from agentic_workflow.workflow_state import GroupEntity, MemberEntity, TaskState


class E360UseCaseAgent(ABC):
    """Base class for all use case specific agents."""

    agent_name: str
    steps_completed: list[str]
    logger: Any

    def __init__(self, logger):
        self.logger = logger
        self.steps_completed: list[str] = []

    def log_step(self, message: str, error: bool = False) -> None:
        message = message + " " + "=" * 4  # (decorative separator seen in frame)
        self.steps_completed.append(message)
        if error:
            self.logger.critical(message)
        else:
            self.logger.info(message)

    def requires_human_approval(self) -> bool:
        """True if human approval is required for this flow. Decided based on
        config."""
        # exact body scrolled past; returns a config-driven boolean in the real code
        raise NotImplementedError

    @abstractmethod
    async def pre_approval(self, *args, **kwargs):
        """Entry point invoked by main.process_*_request before human approval.

        Signature inferred from main.py usage (agent.pre_approval(entities=...)).
        Concrete agents (e.g. AddressChangeAgent) implement this.
        """
        ...
