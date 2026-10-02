# ==================================================
# FRAUDGUARD AI
# CONDITIONAL ROUTING / ESCALATION ENGINE
# ==================================================

def determine_route(policy_result):
    """
    Determines the workflow route for an investigation
    based on the deterministic policy engine result.

    The routing engine does NOT make a fraud decision.
    It only determines the next workflow step.
    """

    if not policy_result:
        return {
            "route": "ERROR",
            "workflow_status": "ROUTING_ERROR",
            "requires_human_review": True,
            "priority": "UNKNOWN",
            "reason": "No policy result was provided."
        }

    risk_level = str(
        policy_result.get(
            "risk_level",
            ""
        )
    ).upper()

    recommended_action = str(
        policy_result.get(
            "recommended_action",
            ""
        )
    ).upper()

    risk_score = policy_result.get(
        "risk_score"
    )

    # ==================================================
    # LOW RISK
    # ==================================================

    if (
        risk_level == "LOW"
        and recommended_action == "AUTO_CLOSE"
    ):

        return {
            "route": "AUTO_CLOSE",
            "workflow_status": "AUTO_CLOSED",
            "requires_human_review": False,
            "priority": "LOW",
            "reason": (
                "The deterministic policy engine classified "
                "the case as LOW risk and recommended "
                "automatic closure."
            ),
            "risk_score": risk_score
        }

    # ==================================================
    # MEDIUM RISK
    # ==================================================

    if (
        risk_level == "MEDIUM"
        and recommended_action == "HUMAN_REVIEW"
    ):

        return {
            "route": "HUMAN_REVIEW",
            "workflow_status": "PENDING_HUMAN_REVIEW",
            "requires_human_review": True,
            "priority": "MEDIUM",
            "reason": (
                "The deterministic policy engine classified "
                "the case as MEDIUM risk. Human investigator "
                "review is required."
            ),
            "risk_score": risk_score
        }

    # ==================================================
    # HIGH RISK
    # ==================================================

    if (
        risk_level == "HIGH"
        and recommended_action == "ESCALATE"
    ):

        return {
            "route": "ESCALATE",
            "workflow_status": "PENDING_ESCALATION",
            "requires_human_review": True,
            "priority": "HIGH",
            "reason": (
                "The deterministic policy engine classified "
                "the case as HIGH risk. The case requires "
                "escalation and human investigation."
            ),
            "risk_score": risk_score
        }

    # ==================================================
    # FALLBACK / UNEXPECTED POLICY RESULT
    # ==================================================

    return {
        "route": "MANUAL_REVIEW",
        "workflow_status": "PENDING_MANUAL_REVIEW",
        "requires_human_review": True,
        "priority": "UNKNOWN",
        "reason": (
            "The policy result does not match a recognised "
            "routing combination. Manual review is required."
        ),
        "risk_score": risk_score
    }