import json
from datetime import datetime
from pathlib import Path


# --------------------------------------------------
# AUDIT LOG FILE
# --------------------------------------------------

AUDIT_LOG_FILE = Path("data/audit_log.jsonl")


# --------------------------------------------------
# ENSURE AUDIT LOG DIRECTORY EXISTS
# --------------------------------------------------

def _ensure_log_directory():
    AUDIT_LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


# --------------------------------------------------
# WRITE AUDIT EVENT
# --------------------------------------------------

def log_event(
    event_type,
    case_id,
    transaction_id=None,
    customer_id=None,
    risk_score=None,
    risk_level=None,
    recommended_action=None,
    human_decision=None,
    investigator_comment=None,
    metadata=None
):
    """
    Records an auditable FraudGuard AI event.

    The audit log uses JSON Lines format so that each
    investigation event is stored as a separate record.
    """

    _ensure_log_directory()

    event = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),

        "event_type": event_type,

        "case_id": case_id,

        "transaction_id": transaction_id,

        "customer_id": customer_id,

        "risk_score": risk_score,

        "risk_level": risk_level,

        "recommended_action": recommended_action,

        "human_decision": human_decision,

        "investigator_comment": investigator_comment,

        "metadata": metadata or {}
    }

    with open(
        AUDIT_LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                event,
                ensure_ascii=False
            )
            + "\n"
        )

    return event


# --------------------------------------------------
# READ AUDIT LOG
# --------------------------------------------------

def get_audit_logs():
    """
    Returns all stored audit events.
    """

    _ensure_log_directory()

    if not AUDIT_LOG_FILE.exists():
        return []

    logs = []

    with open(
        AUDIT_LOG_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                logs.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:
                continue

    return logs


# --------------------------------------------------
# CLEAR AUDIT LOG
# --------------------------------------------------

def clear_audit_logs():
    """
    Clears the audit log.

    Intended for prototype/demo use only.
    """

    _ensure_log_directory()

    if AUDIT_LOG_FILE.exists():
        AUDIT_LOG_FILE.unlink()