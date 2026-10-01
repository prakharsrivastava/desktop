"""
main.py  -  e360-address-change-agent FastAPI service entrypoint

Reconstructed from screen recording 2026-06-11_13-04-14.mp4.
Confidence: HIGH for health() and the top of process_address_change_request()
(lines ~108-141 were on screen); the FastAPI app construction, lifespan,
CORS middleware and route declarations (seen in 2026-06-11_13-12-47.mp4 for the
sibling service) are reconstructed as faithful scaffolding and marked below.
Lines after ~141 of process_address_change_request were scrolled past.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

# --- imports were partly off-screen; these are the ones referenced in body ---
# from logger import setup_logger
# from agentic_workflow.member_entities import convert_matched_entities_to_member_entities
# from usecases import AGENT_TASK_MAPPING


async def health():
    return {
        "status": "UP",
        "service": "e360-address-change-agent",
        "version": "1.0.0",
        "timestamp": datetime.now(ZoneInfo("US/Eastern")).isoformat(),
        "checks": {
            "application": {
                "status": "UP"
            }
        },
    }


async def process_address_change_request(
    request_payload: dict, initial_query_received_at: datetime
):
    """Process address change request using simplified logic"""
    req_id = request_payload.get("unique_ticket_id", "NoMWI")
    logger = setup_logger(__name__, req_id)  # noqa: F821 - import off-screen

    try:
        # Extract required data from payload
        matched_entities_raw = request_payload.get("matched_entities", [])
        entities = convert_matched_entities_to_member_entities(  # noqa: F821
            matched_entities_raw, req_id
        )
        email_from = request_payload.get("email_from", "")
        email_subject = request_payload.get("email_subject", "")
        email_body = request_payload.get("email_body", "")
        attachment_summary = request_payload.get("attachment_summary", "")
        lob = request_payload.get("e360_lob", "")
        state_code = request_payload.get("e360_state", "")

        agent = AGENT_TASK_MAPPING["address_assistant"](logger=logger)  # noqa: F821
        task = await agent.pre_approval(
            entities=entities,
            # --- remaining keyword args scrolled past in the recording ---
            # email_from=email_from, email_subject=email_subject,
            # email_body=email_body, attachment_summary=attachment_summary,
            # lob=lob, state_code=state_code,
            # initial_query_received_at=initial_query_received_at,
        )
        # ... continuation not captured in recording ...
        return task
    except Exception as e:  # noqa: F841 - body not fully shown
        # error handling was below the visible region
        raise


# ---------------------------------------------------------------------------
# FastAPI wiring below is reconstructed from the sibling service pattern seen in
# 2026-06-11_13-12-47.mp4 (AddressChangeAgent service). Marked MEDIUM confidence.
# ---------------------------------------------------------------------------
# from contextlib import asynccontextmanager
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Initialize any required services
#     yield
#
# app = FastAPI(
#     title="Address Change Agent Service",
#     description="Standalone Address Change processing agent",
#     version="1.0.0",
#     lifespan=lifespan,
# )
# origins = ["*"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
