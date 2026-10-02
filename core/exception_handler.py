# ==================================================
# FRAUDGUARD AI
# EXCEPTION HANDLER
# ==================================================

def create_stage_error(
    stage,
    error
):
    """
    Creates a structured error object for a failed
    investigation stage.
    """

    return {
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
        "status": "FAILED"
    }


def create_manual_review_route(
    reason,
    risk_score=None
):
    """
    Safe fallback route used whenever the investigation
    cannot be completed reliably.

    The system does not make an automatic decision when
    critical information is unavailable.
    """

    return {
        "route": "MANUAL_REVIEW",
        "workflow_status": "EXCEPTION_REVIEW",
        "requires_human_review": True,
        "priority": "HIGH",
        "reason": reason,
        "risk_score": risk_score
    }


def create_investigation_error(
    stage,
    error,
    partial_results=None
):
    """
    Creates the complete investigation result for a
    failed investigation.

    Any investigation failure is routed to manual review.
    """

    error_details = create_stage_error(
        stage=stage,
        error=error
    )

    result = {
        "status": "ERROR",
        "error": error_details,

        "transaction_agent": (
            partial_results.get(
                "transaction_agent"
            )
            if partial_results
            else None
        ),

        "behavior_agent": (
            partial_results.get(
                "behavior_agent"
            )
            if partial_results
            else None
        ),

        "device_channel_agent": (
            partial_results.get(
                "device_channel_agent"
            )
            if partial_results
            else None
        ),

        "policy_result": (
            partial_results.get(
                "policy_result"
            )
            if partial_results
            else None
        ),

        "routing_result": create_manual_review_route(
            reason=(
                f"Investigation could not be completed "
                f"because the {stage} stage failed. "
                f"Manual review is required."
            ),
            risk_score=(
                partial_results.get(
                    "policy_result",
                    {}
                ).get("risk_score")
                if partial_results
                and isinstance(
                    partial_results.get(
                        "policy_result"
                    ),
                    dict
                )
                else None
            )
        ),

        "recommendation_agent": (
            partial_results.get(
                "recommendation_agent"
            )
            if partial_results
            else None
        )
    }

    return result