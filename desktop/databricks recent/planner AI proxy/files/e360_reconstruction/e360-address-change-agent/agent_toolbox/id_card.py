"""
agent_toolbox.id_card

Reconstructed from screen recording 2026-06-11_12-53-02.mp4.
Confidence: HIGH for IDCardRequestStatus dataclass and the visible portions of
fetch_id_card_plan_preference (tail) and request_new_id_card (top, lines ~11-70).
The bodies above line 11 and below line 70 were scrolled past.
"""

from dataclasses import dataclass
from typing import Literal

# Seen / inferred imports (off-screen):
# from agent_toolbox.member_details import MemberDetails
# from external_apis.soa import get_auth_soa_async, post_id_card_request
# from external_apis.edp import string_to_base64


async def fetch_id_card_plan_preference(
    member_details: "MemberDetails", logger  # noqa: F821
) -> list[dict]:
    # --- top of this function (lines 1-39) scrolled past in recording ---
    # ... builds id_card_prefs from a SOA / plan-preference call ...
    id_card_prefs: list[dict] = []  # placeholder for the off-screen construction

    if len(id_card_prefs) < 1:
        logger.error("ID Card preference not in preference list")
        return []

    return id_card_prefs


@dataclass
class IDCardRequestStatus:
    status: Literal["success", "failure", "error"]
    description: str


async def request_new_id_card(
    member: "MemberDetails", group_number: str, case_number: str, logger  # noqa: F821
) -> IDCardRequestStatus:
    """
    Raise a request for a new ID card using SOA

    Returns
    -------
    IDCardRequestStatus
    """
    try:
        client_secret = string_to_base64()  # noqa: F821
        token, status = await get_auth_soa_async(client_secret)  # noqa: F821
        if not status:
            return IDCardRequestStatus(
                status="error", description="Error getting token"
            )
        result = await post_id_card_request(  # noqa: F821
            token=token,
            # --- remaining kwargs + result handling scrolled past (lines 70+) ---
        )
        # ... continuation not captured in recording ...
        return IDCardRequestStatus(status="success", description="")
    except Exception as e:
        logger.error(f"Failed to request new ID card: {e}")
        return IDCardRequestStatus(status="error", description=str(e))
