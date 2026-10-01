"""
usecases.address_change.agent  -  AddressChangeAgent

Reconstructed from screen recording 2026-06-11_13-04-14.mp4.
Confidence: this is a LARGE class (~400+ lines in the recording). Only a window
of it was on screen (the class header `class AddressChangeAgent(E360UseCaseAgent):`,
the `tokenize_sensitive_fields` breadcrumb, and the `set_qmcso_fields` method
around lines 381-413). Everything else (most of the 400 lines) was scrolled past
and is NOT reproduced here to avoid fabrication. The visible portion is faithful.
"""

from datetime import date

from usecases.base import E360UseCaseAgent

# Seen / inferred imports (off-screen):
# from agentic_workflow.workflow_state import MemberStreetAddress, MemberDetails
# from external_apis.soa import update_limited_liability_spi_mem


class AddressChangeAgent(E360UseCaseAgent):
    agent_name = "address_assistant"

    # --- NOT shown in recording (scrolled past): -------------------------
    #   __init__, pre_approval, validate / detokenize / tokenize_sensitive_fields,
    #   the main address-change orchestration, post-approval execution, etc.
    #   (~lines 1-380 and 414+). Inventory listed in MANIFEST.md.
    # ---------------------------------------------------------------------

    async def set_qmcso_fields(
        self,
        custodial_parent_name: str,
        court_order_date: date,
        custodial_parent_address: "MemberStreetAddress",  # noqa: F821
        logger,
    ) -> bool:
        """
        Set the mainframe fields for an existing QMCSO member, who has a court order

        Returns:
            True if there is an error
            False if there is no error
        """
        spi_text = (
            f"ALT PAYEE record for QMS dep {member_details.firstName} "  # noqa: F821
            f"{member_details.lastName} "
            f"QMCSO received {court_order_date}"
        )

        if (
            member_details.dependent_type is None  # noqa: F821
            or member_details.dependent_type != "QMS"  # noqa: F821
        ):
            self.log_step("Setting Limited Liability SPI MEM note", error=False)
            result = await update_limited_liability_spi_mem(  # noqa: F821
                member_details=member_details,  # noqa: F821
                effective_date=court_order_date,
                spi_end_date="",
                spi_text=spi_text,
                logger=logger,
            )
            if result.status != "success":
                self.steps_completed.append(
                    f"Failed to set Limited Liability SPI MEM text for member: "
                    f"{result.description}"
                )
                # ... continuation scrolled past ...
        return False
