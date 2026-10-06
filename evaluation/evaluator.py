import re
import time
import pandas as pd

from core.orchestrator import run_investigation


# ==================================================
# MODELS USED FOR COMPARISON
# ==================================================

MODELS = {
    "Nemotron 3 Super": {
        "provider": "openrouter",
        "model": "nvidia/nemotron-3-super-120b-a12b:free"
    },

    "GPT-OSS 120B": {
        "provider": "groq",
        "model": "openai/gpt-oss-120b"
    },

    "GPT-OSS 20B": {
        "provider": "groq",
        "model": "openai/gpt-oss-20b"
    },

    "Qwen 3.8 27B": {
        "provider": "groq",
        "model": "qwen/qwen3.8-27b"
    }
}


# ==================================================
# REQUIRED OUTPUT SECTIONS
# ==================================================

TRANSACTION_SECTIONS = [
    "1. Overall Transaction Risk Assessment",
    "2. Risk Score",
    "3. Observed Anomalies",
    "4. Supporting Evidence",
    "5. Missing Information",
    "6. Explanation",
    "7. Suggested Investigation Focus",
]


BEHAVIOR_SECTIONS = [
    "1. Behavioural Risk Assessment",
    "2. Behavioural Deviation Score",
    "3. Behavioural Anomalies",
    "4. Supporting Evidence",
    "5. Consistent Behaviour",
    "6. Missing Information",
    "7. Explanation",
    "8. Suggested Investigation Focus",
]


DEVICE_SECTIONS = [
    "1. Device & Channel Risk Assessment",
    "2. Device & Channel Deviation Score",
    "3. Device Indicators",
    "4. Channel Indicators",
    "5. Supporting Evidence",
    "6. Consistent Device/Channel Behaviour",
    "7. Missing Information",
    "8. Explanation",
    "9. Suggested Investigation Focus",
]


RECOMMENDATION_SECTIONS = [
    "## 1. Investigation Recommendation",
    "## 2. Risk Context",
    "## 3. Key Supporting Evidence",
    "## 4. Consistent Behaviour",
    "## 5. Missing Information",
    "## 6. Recommended Verification Steps",
    "## 7. Human Review Requirement",
    "## 8. Final Note",
]


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def _clean_text(value):
    if value is None:
        return ""

    return str(value).strip()


def _extract_score(text, heading):
    """
    Extract a 0-100 score from an agent's existing output.

    Examples supported:
        2. Risk Score
        65

        2. Risk Score: 65

        2. Risk Score
        **65/100**
    """

    text = _clean_text(text)

    if not text:
        return None

    pattern = (
        rf"{re.escape(heading)}"
        rf"\s*(?:\n|\r\n|:)"
        rf"\s*(?:\*\*\s*)?"
        rf"(\d{{1,3}})"
        rf"(?:\s*(?:/\s*100|out of 100|%)?)?"
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE
    )

    if not match:

        match = re.search(
            rf"{re.escape(heading)}.*?"
            rf"(\d{{1,3}})"
            rf"(?:\s*(?:/\s*100|out of 100|%))?",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

    if not match:
        return None

    score = int(match.group(1))

    if 0 <= score <= 100:
        return score

    return None


def _score_to_classification(score):
    """Convert an agent diagnostic score to a risk class.

    IMPORTANT: This helper is diagnostic only. It is NOT used for the
    authoritative end-to-end classification metric. The governed workflow
    uses the deterministic Policy Engine for that decision.
    """
    if score is None:
        return "UNPARSED"

    try:
        score = float(score)
    except (TypeError, ValueError):
        return "UNPARSED"

    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


def _normalize_risk(value):
    """Normalize risk labels from policy/routing/model outputs."""
    value = _clean_text(value).upper().replace("-", "_").replace(" ", "_")

    aliases = {
        "LOW_RISK": "LOW",
        "MEDIUM_RISK": "MEDIUM",
        "MODERATE": "MEDIUM",
        "MODERATE_RISK": "MEDIUM",
        "HIGH_RISK": "HIGH",
        "CRITICAL": "HIGH",
        "CRITICAL_RISK": "HIGH",
    }

    normalized = aliases.get(value, value)
    return normalized if normalized in {"LOW", "MEDIUM", "HIGH"} else "UNPARSED"


def _normalize_action(value):
    """Normalize workflow actions across provider/model output variations."""
    value = _clean_text(value).upper().replace("-", "_").replace(" ", "_")

    aliases = {
        "AUTO_CLOSE": "AUTO_CLOSE",
        "AUTOCLOSE": "AUTO_CLOSE",
        "CLOSE": "AUTO_CLOSE",
        "AUTO_CLOSED": "AUTO_CLOSE",
        "HUMAN_REVIEW": "HUMAN_REVIEW",
        "HUMANREVIEW": "HUMAN_REVIEW",
        "MANUAL_REVIEW": "HUMAN_REVIEW",
        "REVIEW": "HUMAN_REVIEW",
        "ESCALATE": "ESCALATE",
        "ESCALATION": "ESCALATE",
        "ESCALATED": "ESCALATE",
    }

    normalized = aliases.get(value, value)
    return normalized if normalized in {
        "AUTO_CLOSE", "HUMAN_REVIEW", "ESCALATE"
    } else "UNPARSED"


def _resolve_model_config(
    selected_model_name,
    selected_model_config=None,
    selected_model_id=None
):
    """Resolve and validate provider/model parameters in one place."""
    config = selected_model_config

    if config is None and isinstance(selected_model_id, dict):
        config = selected_model_id

    if config is None:
        config = {
            "provider": "openrouter",
            "model": selected_model_id
        }

    if not isinstance(config, dict):
        raise ValueError(
            f"Model configuration for '{selected_model_name}' must be a dictionary."
        )

    provider = _clean_text(config.get("provider", "openrouter")).lower()
    model_id = _clean_text(config.get("model", selected_model_id))

    if provider not in {"openrouter", "groq"}:
        raise ValueError(
            f"Unsupported provider '{provider}' for model '{selected_model_name}'. "
            "Expected 'openrouter' or 'groq'."
        )

    if not model_id:
        raise ValueError(
            f"No model ID configured for '{selected_model_name}'."
        )

    return {
        "provider": provider,
        "model": model_id
    }


def _get_authoritative_workflow_outcome(result):
    """
    Extract the governed workflow decision.

    FraudGuard's deterministic Policy Engine is authoritative for risk and
    the Routing Engine is authoritative for the workflow action. LLM agent
    scores are retained only as diagnostic evidence and must never replace
    these decisions in end-to-end model evaluation.
    """
    if not isinstance(result, dict):
        return "UNPARSED", "UNPARSED", "", None

    policy = result.get("policy_result")
    routing = result.get("routing_result")

    if not isinstance(policy, dict):
        policy = {}
    if not isinstance(routing, dict):
        routing = {}

    risk = _normalize_risk(
        policy.get("risk_level")
    )

    action = _normalize_action(
        routing.get("route")
    )

    # Fallback to the deterministic policy action when routing is missing.
    if action == "UNPARSED":
        action = _normalize_action(
            policy.get("recommended_action")
        )

    # If risk is missing but a valid deterministic action exists, infer only
    # the corresponding class. This is a fallback for older result objects;
    # it is never derived from LLM text.
    if risk == "UNPARSED":
        risk_from_action = {
            "AUTO_CLOSE": "LOW",
            "HUMAN_REVIEW": "MEDIUM",
            "ESCALATE": "HIGH"
        }
        risk = risk_from_action.get(action, "UNPARSED")

    route = _clean_text(routing.get("route", "")).upper()
    workflow_status = routing.get("workflow_status")

    return risk, action, route, workflow_status


def _expected_classification(ground_truth):

    value = _clean_text(ground_truth).upper()

    if value in {"LOW", "MEDIUM", "HIGH"}:
        return value

    return "UNKNOWN"


def _normalize_heading(value):
    """Normalize markdown headings for model-independent section matching."""
    value = _clean_text(value).lower()
    value = re.sub(r"^#+\s*", "", value)
    value = re.sub(r"^\*+|\*+$", "", value)
    value = value.replace("&", "and")
    value = value.replace("behaviour", "behavior")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _section_compliance(text, required_sections):
    """
    Measure required-section coverage without depending on one model's exact
    Markdown style. Numbering, ## markers, bold markers, punctuation and the
    Behaviour/Behavior spelling are normalized before comparison.
    """
    text = _clean_text(text)

    if not text or not required_sections:
        return 0.0, 0, len(required_sections)

    normalized_text = _normalize_heading(text)

    present = 0
    for section in required_sections:
        normalized_section = _normalize_heading(section)
        if normalized_section and normalized_section in normalized_text:
            present += 1

    percentage = (present / len(required_sections)) * 100
    return percentage, present, len(required_sections)


def _explanation_quality(
    transaction_text,
    behavior_text,
    device_text,
    recommendation_text
):
    """
    Rule-based explanation quality score out of 10.

    2 points each:
    1. Evidence identification
    2. Risk reasoning
    3. Policy alignment
    4. Missing information
    5. Investigator usefulness
    """

    score = 0

    evidence_text = " ".join([
        _clean_text(transaction_text),
        _clean_text(behavior_text),
        _clean_text(device_text),
        _clean_text(recommendation_text),
    ]).lower()

    # 1. Evidence identification
    if any(
        term in evidence_text
        for term in [
            "supporting evidence",
            "observed anomalies",
            "key supporting evidence",
            "evidence:"
        ]
    ):
        score += 2

    # 2. Risk reasoning
    if (
        any(
            term in evidence_text
            for term in ["explanation", "risk assessment", "risk context"]
        )
        and any(
            term in evidence_text
            for term in [
                "amount", "location", "device", "channel",
                "transaction", "deviation", "risk score"
            ]
        )
    ):
        score += 2

    # 3. Decision/policy alignment
    if (
        any(
            term in evidence_text
            for term in ["policy", "recommended action", "recommendation", "decision"]
        )
        and any(
            term in evidence_text
            for term in [
                "auto_close", "auto close", "human_review",
                "human review", "escalate", "escalation"
            ]
        )
    ):
        score += 2

    # 4. Missing information
    if any(
        term in evidence_text
        for term in ["missing information", "missing data", "information gaps"]
    ):
        score += 2

    # 5. Investigator usefulness
    if any(
        term in evidence_text
        for term in [
            "verification steps", "investigation focus",
            "investigator should", "next steps", "recommended verification"
        ]
    ):
        score += 2

    return score


def _grounding_violations(text):

    text = _clean_text(text).lower()

    if not text:
        return 0

    violation_patterns = [

        # Fraud confirmation
        r"fraud has been confirmed",
        r"transaction is fraudulent",
        r"this is definitely fraud",

        # Autonomous actions
        r"block the account",
        r"freeze the account",
        r"reverse the transaction",
        r"chargeback should be initiated",
    ]

    violations = 0

    for pattern in violation_patterns:

        if re.search(pattern, text):
            violations += 1

    return violations


def _task_completed(result):

    if not isinstance(result, dict):
        return False

    if result.get("status") != "SUCCESS":
        return False

    required_outputs = [
        "transaction_agent",
        "behavior_agent",
        "device_channel_agent",
        "policy_result",
        "routing_result",
        "recommendation_agent",
    ]

    return all(
        _clean_text(result.get(key))
        for key in required_outputs
    )


# ==================================================
# CASE INPUT PREPARATION
# ==================================================

def _get_case_inputs(
    test_case,
    transactions,
    customers,
    devices
):

    transaction_id = test_case["transaction_id"]

    transaction_row = transactions[
        transactions["transaction_id"] == transaction_id
    ]

    if transaction_row.empty:

        raise ValueError(
            f"Transaction {transaction_id} not found."
        )

    transaction = transaction_row.iloc[0].to_dict()

    customer_row = customers[
        customers["customer_id"]
        == transaction["customer_id"]
    ]

    if customer_row.empty:

        raise ValueError(
            f"Customer {transaction['customer_id']} not found."
        )

    customer = customer_row.iloc[0].to_dict()

    device_row = devices[
        devices["device_id"]
        == transaction["device_id"]
    ]

    if device_row.empty:

        raise ValueError(
            f"Device {transaction['device_id']} not found."
        )

    device = device_row.iloc[0].to_dict()

    return transaction, customer, device


# ==================================================

# ==================================================
# MODEL EVALUATION HISTORY
# ==================================================

HISTORY_PATH = "data/model_evaluation_history.json"


def load_evaluation_history():
    """Load previously recorded model evaluation attempts from JSON."""
    import json
    from pathlib import Path

    path = Path(HISTORY_PATH)

    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []

    except (json.JSONDecodeError, OSError):
        return []


def _json_safe(value):
    """Convert pandas/numpy scalar values into JSON-safe Python values."""
    if value is None:
        return None

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def _summary_value(summary_row, column, default=None):
    """Read a summary value while safely handling NaN/None."""
    value = summary_row.get(column, default)

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def append_evaluation_history(
    evaluation_result,
    selected_model_name,
    selected_model_config=None,
    selected_model_id=None
):
    """
    Append one evaluation attempt to persistent history.

    History is stored at case level so that manually evaluating TC001,
    TC002 and TC003 at different times can later be combined into a
    cumulative model view without rerunning completed cases.
    """
    import json
    from datetime import datetime
    from pathlib import Path

    path = Path(HISTORY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    history = load_evaluation_history()

    selected_model_config = _resolve_model_config(
        selected_model_name=selected_model_name,
        selected_model_config=selected_model_config,
        selected_model_id=selected_model_id
    )

    provider = selected_model_config["provider"]
    model_id = selected_model_config["model"]

    summary_df = evaluation_result.get("summary", pd.DataFrame())
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        summary_row = summary_df.iloc[0].to_dict()
    else:
        summary_row = {}

    case_df = evaluation_result.get("case_results", pd.DataFrame())
    if isinstance(case_df, pd.DataFrame):
        case_records = case_df.to_dict(orient="records")
    else:
        case_records = []

    case_records = [
        _json_safe(record)
        for record in case_records
    ]

    case_ids = [
        str(record.get("case_id"))
        for record in case_records
        if record.get("case_id") is not None
    ]

    successful_cases = sum(
        1
        for record in case_records
        if str(record.get("status", "")).upper() == "SUCCESS"
    )

    failed_cases = len(case_records) - successful_cases

    entry = {
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "model": selected_model_name,
        "provider": provider,
        "model_id": model_id,
        "evaluation_mode": evaluation_result.get(
            "evaluation_mode",
            "full"
        ),
        "case_ids": case_ids,
        "successful_cases": successful_cases,
        "failed_cases": failed_cases,
        "classification_quality": _summary_value(
            summary_row,
            "Classification Accuracy (%)"
        ),
        "escalation_accuracy": _summary_value(
            summary_row,
            "Escalation Accuracy (%)"
        ),
        "false_positive_rate": _summary_value(
            summary_row,
            "False Positive Rate (%)"
        ),
        "hallucination_rate": _summary_value(
            summary_row,
            "Hallucination / Grounding Violation Rate (%)"
        ),
        "explanation_quality": _summary_value(
            summary_row,
            "Explanation Quality (/10)"
        ),
        "structured_output_compliance": _summary_value(
            summary_row,
            "Structured Output Compliance (%)"
        ),
        "average_latency": _summary_value(
            summary_row,
            "Average Latency (s)"
        ),
        "task_completion": _summary_value(
            summary_row,
            "Task Completion (%)"
        ),
        "execution_success_rate": _summary_value(
            summary_row,
            "Execution Success Rate (%)"
        ),
        "execution_failure_rate": _summary_value(
            summary_row,
            "Execution Failure Rate (%)"
        ),
        "case_results": case_records
    }

    history.insert(0, _json_safe(entry))

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            history,
            file,
            indent=2,
            ensure_ascii=False
        )

    return entry


def _successful_case_history(history, model_name, provider=None, model_id=None):
    """
    Return the latest successful result for each case for one model.

    A case that is rerun is not double-counted. A transient API failure also
    does not overwrite a previously successful case result in the cumulative
    quality metrics.
    """
    latest_successful = {}

    for entry in history:
        if entry.get("model") != model_name:
            continue

        if provider is not None and _clean_text(entry.get("provider")).lower() != _clean_text(provider).lower():
            continue

        if model_id is not None and _clean_text(entry.get("model_id")) != _clean_text(model_id):
            continue

        case_results = entry.get("case_results", [])
        if not isinstance(case_results, list):
            continue

        for case_result in case_results:
            case_id = str(case_result.get("case_id", ""))
            if not case_id:
                continue

            if (
                str(case_result.get("status", "")).upper() == "SUCCESS"
                and case_id not in latest_successful
            ):
                latest_successful[case_id] = case_result

    return latest_successful


def build_cumulative_summary(
    history,
    selected_model_name,
    provider,
    total_configured_cases,
    model_id=None
):
    """
    Build a cumulative summary from previously successful case evaluations.

    The cumulative view uses the latest successful result for each unique
    test case. It therefore supports manual case-by-case evaluation while
    avoiding duplicate weighting when a case is rerun.
    """
    successful_cases = _successful_case_history(
        history,
        selected_model_name,
        provider=provider,
        model_id=model_id
    )

    case_records = list(successful_cases.values())
    case_df = pd.DataFrame(case_records)

    if case_df.empty:
        return pd.DataFrame([
            {
                "Model": selected_model_name,
                "Provider": provider,
                "Configured Test Cases": int(total_configured_cases),
                "Cases Evaluated": 0,
                "Cases Remaining": int(total_configured_cases),
                "Classification Accuracy (%)": None,
                "Escalation Accuracy (%)": None,
                "False Positive Rate (%)": None,
                "Hallucination / Grounding Violation Rate (%)": None,
                "Explanation Quality (/10)": None,
                "Structured Output Compliance (%)": None,
                "Task Completion (%)": None,
                "Average Latency (s)": None
            }
        ])

    summary = _calculate_quality_summary(
        case_df,
        selected_model_name,
        provider
    )

    row = summary.iloc[0].to_dict()
    row["Configured Test Cases"] = int(total_configured_cases)
    row["Cases Evaluated"] = len(case_df)
    row["Cases Remaining"] = max(
        int(total_configured_cases) - len(case_df),
        0
    )

    ordered = {
        "Model": row.get("Model", selected_model_name),
        "Provider": row.get("Provider", provider),
        "Configured Test Cases": row["Configured Test Cases"],
        "Cases Evaluated": row["Cases Evaluated"],
        "Cases Remaining": row["Cases Remaining"],
        "Classification Accuracy (%)": row.get(
            "Classification Accuracy (%)"
        ),
        "Escalation Accuracy (%)": row.get(
            "Escalation Accuracy (%)"
        ),
        "False Positive Rate (%)": row.get(
            "False Positive Rate (%)"
        ),
        "Hallucination / Grounding Violation Rate (%)": row.get(
            "Hallucination / Grounding Violation Rate (%)"
        ),
        "Explanation Quality (/10)": row.get(
            "Explanation Quality (/10)"
        ),
        "Structured Output Compliance (%)": row.get(
            "Structured Output Compliance (%)"
        ),
        "Task Completion (%)": row.get("Task Completion (%)"),
        "Average Latency (s)": row.get("Average Latency (s)")
    }

    return pd.DataFrame([ordered])


# ==================================================
# METRIC CALCULATION
# ==================================================


def _calculate_quality_summary(case_df, model_name, provider):
    """
    Calculate model-quality metrics using successful cases only.

    Classification and escalation metrics come from the governed Policy and
    Routing Engines, not from model-generated risk scores. This keeps the
    benchmark comparable across different model providers.

    API/provider failures are tracked separately as execution failures and
    are never interpreted as incorrect model classifications or hallucinations.
    """
    successful_df = case_df[
        case_df["status"].astype(str).str.upper() == "SUCCESS"
    ].copy() if not case_df.empty else pd.DataFrame()

    attempted = len(case_df)
    successful_count = len(successful_df)
    failed_count = attempted - successful_count

    if successful_count > 0:
        classification_accuracy = (
            successful_df["classification_correct"].mean() * 100
        )
        escalation_accuracy = (
            successful_df["escalation_correct"].mean() * 100
        )
        task_completion = (
            successful_df["task_completed"].mean() * 100
        )
        structured_compliance = (
            successful_df["structured_compliance_pct"].mean()
        )
        explanation_quality = (
            successful_df["explanation_quality"].mean()
        )
        grounding_rate = (
            (successful_df["grounding_violations"] > 0).mean() * 100
        )

        latency_values = successful_df["latency_seconds"].dropna()
        avg_latency = (
            latency_values.mean()
            if not latency_values.empty
            else None
        )

        normal_cases = successful_df[
            successful_df["expected_risk"] == "LOW"
        ]

        false_positive_rate = (
            normal_cases["false_positive"].mean() * 100
            if not normal_cases.empty
            else None
        )
    else:
        classification_accuracy = None
        escalation_accuracy = None
        task_completion = None
        structured_compliance = None
        explanation_quality = None
        grounding_rate = None
        avg_latency = None
        false_positive_rate = None

    execution_success_rate = (
        (successful_count / attempted) * 100
        if attempted > 0
        else 0
    )

    execution_failure_rate = (
        (failed_count / attempted) * 100
        if attempted > 0
        else 0
    )

    def rounded(value, digits):
        return round(float(value), digits) if value is not None else None

    return pd.DataFrame([
        {
            "Model": model_name,
            "Provider": provider,
            "Total Test Cases": attempted,
            "Successful Cases": successful_count,
            "Failed Cases": failed_count,
            "Classification Accuracy (%)": rounded(
                classification_accuracy, 1
            ),
            "Escalation Accuracy (%)": rounded(
                escalation_accuracy, 1
            ),
            "False Positive Rate (%)": rounded(
                false_positive_rate, 1
            ),
            "Hallucination / Grounding Violation Rate (%)": rounded(
                grounding_rate, 1
            ),
            "Explanation Quality (/10)": rounded(
                explanation_quality, 1
            ),
            "Structured Output Compliance (%)": rounded(
                structured_compliance, 1
            ),
            "Task Completion (%)": rounded(
                task_completion, 1
            ),
            "Average Latency (s)": rounded(
                avg_latency, 2
            ),
            "Execution Success Rate (%)": round(
                execution_success_rate, 1
            ),
            "Execution Failure Rate (%)": round(
                execution_failure_rate, 1
            )
        }
    ])


# ==================================================
# SINGLE / FULL MODEL EVALUATION
# ==================================================


def run_model_evaluation(
    test_cases,
    transactions,
    customers,
    devices,
    selected_model_name,
    selected_model_config=None,
    selected_model_id=None,
    evaluation_mode="full",
    selected_case_id=None
):
    """
    Evaluate one selected model.

    Modes:
    - ``single``: execute exactly one selected test case.
    - ``full``: execute all configured test cases for the selected model.

    The UI can therefore evaluate TC001 with one model, later TC002 with the
    same model, and finally TC003 without rerunning already completed cases.
    """
    model_name = selected_model_name

    selected_model_config = _resolve_model_config(
        selected_model_name=model_name,
        selected_model_config=selected_model_config,
        selected_model_id=selected_model_id
    )

    provider = selected_model_config["provider"]
    model_id = selected_model_config["model"]

    evaluation_mode = str(evaluation_mode).lower().strip()

    if evaluation_mode not in {"single", "full"}:
        raise ValueError(
            "evaluation_mode must be either 'single' or 'full'."
        )

    if evaluation_mode == "single":
        if not selected_case_id:
            raise ValueError(
                "A test case must be selected in Single Case Evaluation mode."
            )

        selected_cases = test_cases[
            test_cases["case_id"].astype(str) == str(selected_case_id)
        ].copy()

        if selected_cases.empty:
            raise ValueError(
                f"Test case '{selected_case_id}' was not found."
            )
    else:
        selected_cases = test_cases.copy()

    case_results = []
    raw_results = {model_name: {}}

    for _, case_row in selected_cases.iterrows():
        case_id = str(case_row["case_id"])
        expected_risk = _expected_classification(
            case_row["ground_truth"]
        )
        expected_action = _normalize_action(
            case_row["expected_action"]
        )

        try:
            transaction, customer, device = _get_case_inputs(
                case_row,
                transactions,
                customers,
                devices
            )

            start_time = time.perf_counter()

            result = run_investigation(
                transaction=transaction,
                customer=customer,
                device=device,
                model=model_id,
                provider=provider
            )

            latency = time.perf_counter() - start_time
            raw_results[model_name][case_id] = result

            if not isinstance(result, dict) or result.get("status") != "SUCCESS":
                error_message = (
                    result.get("error", {}).get(
                        "message",
                        "Unknown investigation error."
                    )
                    if isinstance(result, dict)
                    else "Invalid investigation result."
                )

                case_results.append({
                    "model": model_name,
                    "provider": provider,
                    "model_id": model_id,
                    "case_id": case_id,
                    "expected_risk": expected_risk,
                    "expected_action": expected_action,
                    "predicted_risk": "ERROR",
                    "predicted_action": "ERROR",
                    "classification_correct": False,
                    "escalation_correct": False,
                    "false_positive": False,
                    "task_completed": False,
                    "structured_compliance_pct": 0.0,
                    "explanation_quality": 0,
                    "grounding_violations": 0,
                    "latency_seconds": round(latency, 2),
                    "model_average_score": None,
                    "agent_average_risk": "UNPARSED",
                    "decision_source": "ERROR",
                    "route_from_workflow": "ERROR",
                    "workflow_status": "ERROR",
                    "status": "ERROR",
                    "error": str(error_message)
                })
                continue

            transaction_text = _clean_text(
                result.get("transaction_agent")
            )
            behavior_text = _clean_text(
                result.get("behavior_agent")
            )
            device_text = _clean_text(
                result.get("device_channel_agent")
            )
            recommendation_text = _clean_text(
                result.get("recommendation_agent")
            )

            transaction_score = _extract_score(
                transaction_text,
                "2. Risk Score"
            )
            behavior_score = _extract_score(
                behavior_text,
                "2. Behavioural Deviation Score"
            )
            device_score = _extract_score(
                device_text,
                "2. Device & Channel Deviation Score"
            )

            scores = [
                transaction_score,
                behavior_score,
                device_score
            ]
            valid_scores = [
                score for score in scores
                if score is not None
            ]

            average_score = (
                sum(valid_scores) / len(valid_scores)
                if valid_scores
                else None
            )

            # --------------------------------------------------
            # AUTHORITATIVE END-TO-END DECISION
            # --------------------------------------------------
            # Never derive the benchmark classification from the average of
            # independent LLM scores. Agent scores are model-dependent and
            # are retained only as diagnostic information. The deterministic
            # Policy Engine and Routing Engine are the governed authorities.
            predicted_risk, predicted_action, workflow_route, workflow_status = (
                _get_authoritative_workflow_outcome(result)
            )

            transaction_compliance = _section_compliance(
                transaction_text,
                TRANSACTION_SECTIONS
            )[0]
            behavior_compliance = _section_compliance(
                behavior_text,
                BEHAVIOR_SECTIONS
            )[0]
            device_compliance = _section_compliance(
                device_text,
                DEVICE_SECTIONS
            )[0]
            recommendation_compliance = _section_compliance(
                recommendation_text,
                RECOMMENDATION_SECTIONS
            )[0]

            structured_compliance = (
                transaction_compliance
                + behavior_compliance
                + device_compliance
                + recommendation_compliance
            ) / 4

            all_output = " ".join([
                transaction_text,
                behavior_text,
                device_text,
                recommendation_text
            ])

            grounding_violations = _grounding_violations(
                all_output
            )

            # workflow_route and workflow_status were obtained from the
            # authoritative Policy/Routing decision above.
            case_results.append({
                "model": model_name,
                "provider": provider,
                "model_id": model_id,
                "case_id": case_id,
                "expected_risk": expected_risk,
                "expected_action": expected_action,
                "predicted_risk": predicted_risk,
                "predicted_action": predicted_action,
                "classification_correct": (
                    predicted_risk == expected_risk
                ),
                "escalation_correct": (
                    predicted_action == expected_action
                ),
                "false_positive": (
                    expected_risk == "LOW"
                    and predicted_risk in {"MEDIUM", "HIGH"}
                ),
                "task_completed": _task_completed(result),
                "structured_compliance_pct": round(
                    structured_compliance,
                    1
                ),
                "explanation_quality": _explanation_quality(
                    transaction_text,
                    behavior_text,
                    device_text,
                    recommendation_text
                ),
                "grounding_violations": grounding_violations,
                "latency_seconds": round(latency, 2),
                "model_average_score": (
                    round(average_score, 1)
                    if average_score is not None
                    else None
                ),
                "agent_average_risk": _score_to_classification(
                    average_score
                ),
                "decision_source": "POLICY_ENGINE + ROUTING_ENGINE",
                "route_from_workflow": workflow_route,
                "workflow_status": _clean_text(workflow_status),
                "status": "SUCCESS",
                "error": ""
            })

        except Exception as exc:
            raw_results[model_name][case_id] = {
                "status": "ERROR",
                "error": str(exc)
            }

            case_results.append({
                "model": model_name,
                "provider": provider,
                "model_id": model_id,
                "case_id": case_id,
                "expected_risk": expected_risk,
                "expected_action": expected_action,
                "predicted_risk": "ERROR",
                "predicted_action": "ERROR",
                "classification_correct": False,
                "escalation_correct": False,
                "false_positive": False,
                "task_completed": False,
                "structured_compliance_pct": 0.0,
                "explanation_quality": 0,
                "grounding_violations": 0,
                "latency_seconds": None,
                "model_average_score": None,
                "agent_average_risk": "UNPARSED",
                "decision_source": "ERROR",
                "route_from_workflow": "ERROR",
                "workflow_status": "ERROR",
                "status": "ERROR",
                "error": str(exc)
            })

    case_df = pd.DataFrame(case_results)
    summary_df = _calculate_quality_summary(
        case_df,
        model_name,
        provider
    )

    return {
        "status": "SUCCESS",
        "evaluation_mode": evaluation_mode,
        "selected_case_id": (
            str(selected_case_id)
            if evaluation_mode == "single"
            else None
        ),
        "summary": summary_df,
        "case_results": case_df,
        "raw_results": raw_results,
        "models": {
            model_name: {
                "provider": provider,
                "model": model_id
            }
        }
    }


# ==================================================
# BACKWARD-COMPATIBLE WRAPPER
# ==================================================


def run_model_comparison(
    test_cases,
    transactions,
    customers,
    devices,
    selected_models=None
):
    """Backward-compatible wrapper that evaluates one selected model."""
    models = selected_models or MODELS
    model_name, model_config = next(iter(models.items()))

    if isinstance(model_config, dict):
        selected_model_config = model_config
        selected_model_id = None
    else:
        selected_model_config = None
        selected_model_id = model_config

    return run_model_evaluation(
        test_cases=test_cases,
        transactions=transactions,
        customers=customers,
        devices=devices,
        selected_model_name=model_name,
        selected_model_config=selected_model_config,
        selected_model_id=selected_model_id,
        evaluation_mode="full"
    )
