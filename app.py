import streamlit as st
import pandas as pd
import requests

from core.data_loader import load_data
from core.audit_logger import log_event, get_audit_logs

from evaluation.evaluator import (
    MODELS,
    run_model_evaluation,
    load_evaluation_history,
    append_evaluation_history,
    build_cumulative_summary
)


# ==================================================
# N8N CONFIGURATION
# ==================================================

# n8n production webhook used by the Streamlit application.
# The n8n workflow must be Active for this endpoint to respond.
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/fraud-investigation"


def run_n8n_workflow(payload):
    """Send a complete investigation payload from Streamlit to n8n."""
    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            timeout=600
        )

        response.raise_for_status()

        return {
            "status": "SUCCESS",
            "result": response.json()
        }

    except requests.exceptions.Timeout:
        return {
            "status": "ERROR",
            "error": (
                "The n8n workflow timed out. Check the n8n execution "
                "and make sure all workflow nodes completed successfully."
            )
        }

    except requests.exceptions.ConnectionError:
        return {
            "status": "ERROR",
            "error": (
                f"Could not connect to n8n at {N8N_WEBHOOK_URL}. "
                "Make sure n8n is running and the workflow is Active."
            )
        }

    except requests.exceptions.RequestException as error:
        return {
            "status": "ERROR",
            "error": f"n8n request failed: {error}"
        }

    except ValueError:
        return {
            "status": "ERROR",
            "error": "n8n returned an invalid JSON response."
        }


def normalize_n8n_result(n8n_response, case_id):
    """
    Convert the current flattened n8n response into the same
    internal structure used by the existing Streamlit pages.
    """
    if not isinstance(n8n_response, dict):
        return {
            "status": "ERROR",
            "error": {
                "stage": "n8n Response",
                "error_type": "InvalidResponse",
                "message": "n8n returned an unexpected response format.",
                "status": "FAILED"
            },
            "routing_result": {
                "route": "MANUAL_REVIEW",
                "workflow_status": "EXCEPTION_REVIEW",
                "requires_human_review": True,
                "priority": "HIGH",
                "reason": (
                    "The n8n response could not be interpreted reliably. "
                    "Manual review is required."
                )
            }
        }

    result = n8n_response.get("result", n8n_response)

    if not isinstance(result, dict):
        result = n8n_response

    if str(result.get("status", "SUCCESS")).upper() == "ERROR":
        error_value = result.get("error", {})

        if isinstance(error_value, dict):
            error_details = error_value
        else:
            error_details = {
                "stage": "n8n Workflow",
                "error_type": "WorkflowError",
                "message": str(error_value),
                "status": "FAILED"
            }

        return {
            "status": "ERROR",
            "error": error_details,
            "transaction_agent": result.get("transaction_agent"),
            "behavior_agent": result.get("behavior_agent"),
            "device_channel_agent": result.get("device_channel_agent"),
            "policy_result": result.get("policy_result"),
            "routing_result": result.get(
                "routing_result",
                {
                    "route": "MANUAL_REVIEW",
                    "workflow_status": "EXCEPTION_REVIEW",
                    "requires_human_review": True,
                    "priority": "HIGH",
                    "reason": (
                        "The n8n workflow encountered an exception. "
                        "Manual review is required."
                    )
                }
            ),
            "recommendation_agent": result.get("recommendation_agent")
        }

    policy_result = result.get("policy_result")

    if not isinstance(policy_result, dict):
        policy_result = {
            "risk_score": result.get("risk_score"),
            "risk_level": result.get("risk_level", "N/A"),
            "recommended_action": result.get(
                "recommended_action",
                "N/A"
            ),
            "indicators": result.get("indicators", [])
        }

    routing_result = result.get("routing_result")

    if not isinstance(routing_result, dict):
        route = result.get(
            "route",
            result.get("outcome", "MANUAL_REVIEW")
        )

        requires_human_review = result.get(
            "requires_human_review",
            result.get("human_review_required", False)
        )

        priority = result.get("priority")

        if priority is None:
            if str(route).upper() == "ESCALATE":
                priority = "HIGH"
            elif str(route).upper() == "HUMAN_REVIEW":
                priority = "MEDIUM"
            else:
                priority = "LOW"

        routing_result = {
            "route": route,
            "workflow_status": result.get(
                "workflow_status",
                "N/A"
            ),
            "requires_human_review": requires_human_review,
            "priority": priority,
            "reason": result.get(
                "routing_reason",
                result.get(
                    "reason",
                    "Routing decision returned by n8n."
                )
            ),
            "risk_score": result.get(
                "risk_score",
                policy_result.get("risk_score")
            )
        }

    return {
        "status": "SUCCESS",
        "transaction_agent": result.get(
            "transaction_agent",
            ""
        ),
        "behavior_agent": result.get(
            "behavior_agent",
            ""
        ),
        "device_channel_agent": result.get(
            "device_channel_agent",
            ""
        ),
        "policy_result": policy_result,
        "routing_result": routing_result,
        "recommendation_agent": result.get(
            "recommendation_agent",
            ""
        ),
        "n8n_result": result,
        "case_id": case_id,
        "outcome": result.get(
            "outcome",
            routing_result.get("route")
        ),
        "human_review_required": result.get(
            "human_review_required",
            routing_result.get(
                "requires_human_review",
                False
            )
        ),
        "escalation_required": result.get(
            "escalation_required",
            str(
                routing_result.get("route", "")
            ).upper() == "ESCALATE"
        )
    }


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# ==================================================
# LOAD DATA
# ==================================================

customers, transactions, devices, fraud_rules, test_cases = load_data()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🛡️ FraudGuard AI")

st.sidebar.caption(
    "Governed Agentic AI for Banking Fraud Investigation"
)

# --------------------------------------------------
# NAVIGATION STATE
# --------------------------------------------------
# The navigation widget is keyed so buttons elsewhere in the
# application can safely change the active page before the next
# Streamlit rerun. This preserves the current investigation in
# session state while taking the investigator directly to Human Review.

def navigate_to_human_review():
    st.session_state["navigation_radio"] = "Human Review"


if "navigation_radio" not in st.session_state:
    st.session_state["navigation_radio"] = "Investigation Dashboard"


page = st.sidebar.radio(
    "Navigation",
    [
        "Investigation Dashboard",
        "Agent Analysis",
        "Human Review",
        "Model Comparison",
        "Governance",
        "Audit Log"
    ],
    key="navigation_radio"
)


# ==================================================
# HEADER
# ==================================================

st.title("🛡️ FraudGuard AI")

st.subheader(
    "Governed Agentic AI for Banking Fraud Investigation"
)

st.markdown(
    """
    FraudGuard AI assists fraud investigators by analysing
    transaction, customer behaviour, device and policy evidence
    while preserving human decision authority.
    """
)

st.divider()


# ==================================================
# INVESTIGATION DASHBOARD
# ==================================================

if page == "Investigation Dashboard":

    st.header("Investigation Dashboard")

    # --------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Test Cases",
            len(test_cases)
        )

    with col2:
        st.metric(
            "Normal Cases",
            len(
                test_cases[
                    test_cases["case_type"] == "Normal"
                ]
            )
        )

    with col3:
        st.metric(
            "Ambiguous Cases",
            len(
                test_cases[
                    test_cases["case_type"] == "Ambiguous"
                ]
            )
        )

    with col4:
        st.metric(
            "High-Risk Cases",
            len(
                test_cases[
                    test_cases["case_type"] == "High Risk"
                ]
            )
        )

    st.divider()

    # --------------------------------------------------
    # CASE SELECTION
    # --------------------------------------------------

    st.subheader("Select Investigation Case")

    selected_case = st.selectbox(
        "Test Case",
        test_cases["case_id"].tolist()
    )

    # --------------------------------------------------
    # CLEAR PREVIOUS RESULT WHEN CASE CHANGES
    # --------------------------------------------------

    previous_case = st.session_state.get(
        "last_selected_case"
    )

    if (
        previous_case is not None
        and previous_case != selected_case
    ):

        st.session_state.pop(
            "investigation_result",
            None
        )

        st.session_state.pop(
            "human_review",
            None
        )

        st.session_state.pop(
            "n8n_last_response",
            None
        )

    # --------------------------------------------------
    # CLEAR RESULTS CREATED BY OLDER WORKFLOW VERSION
    # --------------------------------------------------

    if "investigation_result" in st.session_state:

        existing_result = st.session_state[
            "investigation_result"
        ]

        required_keys = {
            "transaction_agent",
            "behavior_agent",
            "device_channel_agent",
            "policy_result",
            "routing_result",
            "recommendation_agent"
        }

        # Allow both successful and exception results.
        # Exception results may intentionally contain
        # None for some investigation components.
        valid_exception_result = (
            existing_result.get("status") == "ERROR"
            and "error" in existing_result
            and "routing_result" in existing_result
        )

        valid_success_result = (
            required_keys.issubset(existing_result.keys())
        )

        if not valid_success_result and not valid_exception_result:

            st.session_state.pop(
                "investigation_result",
                None
            )

    # --------------------------------------------------
    # STORE CURRENT CASE
    # --------------------------------------------------

    st.session_state["last_selected_case"] = selected_case

    # --------------------------------------------------
    # GET CASE INFORMATION
    # --------------------------------------------------

    case = test_cases[
        test_cases["case_id"] == selected_case
    ].iloc[0]

    transaction_id = case["transaction_id"]

    transaction = transactions[
        transactions["transaction_id"] == transaction_id
    ].iloc[0]

    customer = customers[
        customers["customer_id"] == transaction["customer_id"]
    ].iloc[0]

    device = devices[
        devices["device_id"] == transaction["device_id"]
    ].iloc[0]

    # --------------------------------------------------
    # WORKFLOW CONFIGURATION
    # --------------------------------------------------

    # --------------------------------------------------
    # CASE INFORMATION
    # --------------------------------------------------

    st.divider()

    st.subheader("Case Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write("**Case ID**")

        st.info(
            selected_case
        )

        st.write("**Transaction ID**")

        st.info(
            transaction_id
        )

    with col2:

        st.write("**Customer ID**")

        st.info(
            transaction["customer_id"]
        )

        st.write("**Transaction Amount**")

        st.info(
            f"₹{float(transaction['amount']):,.0f}"
        )

    with col3:

        st.write("**Location**")

        st.info(
            transaction["location"]
        )

        st.write("**Channel**")

        st.info(
            transaction["channel"]
        )

    # --------------------------------------------------
    # TRANSACTION EVIDENCE
    # --------------------------------------------------

    st.divider()

    st.subheader("Transaction Evidence")

    evidence_df = pd.DataFrame({

        "Field": [
            "Transaction Amount",
            "Customer Average",
            "Merchant",
            "Merchant Category",
            "Location",
            "Device ID",
            "Device Trusted",
            "Transaction Time"
        ],

        "Value": [

            f"₹{float(transaction['amount']):,.0f}",

            f"₹{float(customer['avg_transaction']):,.0f}",

            transaction["merchant"],

            transaction["merchant_category"],

            transaction["location"],

            transaction["device_id"],

            str(device["trusted"]),

            transaction["timestamp"]
        ]
    })

    st.dataframe(
        evidence_df,
        use_container_width=True,
        hide_index=True
    )

    # ==================================================
    # ==================================================
    # WORKFLOW EXECUTION
    # ==================================================

    st.divider()

    st.subheader("⚙️ Workflow Execution")

    model_names = list(MODELS.keys())

    selected_model_name = st.selectbox(
        "AI Model",
        model_names,
        index=0,
        help=(
            "The selected model is passed to the AI agents. "
            "The deterministic policy engine and routing logic "
            "remain unchanged."
        ),
        key="investigation_model_name"
    )

    selected_model_config = MODELS[selected_model_name]

    selected_model_provider = selected_model_config.get(
        "provider",
        "openrouter"
    )

    selected_model_id = selected_model_config.get(
        "model"
    )

    if st.button(
        "🚀 Run Investigation",
        type="primary",
        use_container_width=True
    ):

        transaction_payload = transaction.to_dict()
        customer_payload = customer.to_dict()
        device_payload = device.to_dict()

        n8n_payload = {
            "case_id": selected_case,
            "transaction": transaction_payload,
            "customer": customer_payload,
            "device": device_payload,
            "model": selected_model_id,
            "provider": selected_model_provider
        }

        with st.spinner(
            "Running investigation..."
        ):

            n8n_response = run_n8n_workflow(
                n8n_payload
            )

        if n8n_response.get("status") == "ERROR":

            error_message = n8n_response.get(
                "error",
                "Unknown workflow error."
            )

            if "timed out" in error_message.lower():
                error_type = "N8nTimeoutError"
            elif "connect" in error_message.lower():
                error_type = "N8nConnectionError"
            else:
                error_type = "N8nWorkflowError"

            st.session_state["investigation_result"] = {
                "status": "ERROR",
                "error": {
                    "stage": "Workflow",
                    "error_type": error_type,
                    "message": error_message,
                    "status": "FAILED"
                },
                "routing_result": {
                    "route": "MANUAL_REVIEW",
                    "workflow_status": "EXCEPTION_REVIEW",
                    "requires_human_review": True,
                    "priority": "HIGH",
                    "reason": (
                        "The investigation could not be completed. "
                        "Manual review is required."
                    )
                }
            }

        else:

            investigation_result = normalize_n8n_result(
                n8n_response.get("result", {}),
                selected_case
            )

            st.session_state[
                "investigation_result"
            ] = investigation_result

            st.session_state[
                "n8n_last_response"
            ] = n8n_response.get("result", {})

            st.success(
                "✅ Investigation completed successfully."
            )

        st.session_state["selected_case"] = selected_case
        st.session_state["investigation_model"] = selected_model_id
        st.session_state["investigation_provider"] = selected_model_provider

    # DISPLAY INVESTIGATION RESULT
    # ==================================================

    if "investigation_result" in st.session_state:

        investigation = st.session_state[
            "investigation_result"
        ]

        # ==================================================
        # INVESTIGATION STATUS
        # ==================================================

        investigation_status = investigation.get(
            "status",
            "SUCCESS"
        )

        # ==================================================
        # EXCEPTION HANDLING
        # ==================================================

        if investigation_status == "ERROR":

            st.divider()

            st.error(
                "⚠️ Investigation could not be completed."
            )

            error_details = investigation.get(
                "error",
                {}
            )

            # Show only controlled exception information.
            st.write(
                "**Failed Stage:**",
                error_details.get(
                    "stage",
                    "Unknown"
                )
            )

            st.write(
                "**Error Type:**",
                error_details.get(
                    "error_type",
                    "Unknown"
                )
            )

            st.write(
                "**Status:**",
                error_details.get(
                    "status",
                    "FAILED"
                )
            )

            st.warning(
                """
                The investigation encountered an exception.

                No automatic fraud decision has been made.

                The case has been routed to manual review because
                the investigation could not be completed reliably.
                """
            )

            # ------------------------------------------
            # EXCEPTION ROUTING
            # ------------------------------------------

            routing_result = investigation.get(
                "routing_result",
                {}
            )

            st.subheader("🔀 Exception Routing")

            route_col1, route_col2, route_col3 = st.columns(3)

            with route_col1:

                st.write("**Route**")

                st.info(
                    routing_result.get(
                        "route",
                        "MANUAL_REVIEW"
                    )
                )

            with route_col2:

                st.write("**Workflow Status**")

                st.info(
                    routing_result.get(
                        "workflow_status",
                        "EXCEPTION_REVIEW"
                    )
                )

            with route_col3:

                st.write("**Priority**")

                st.info(
                    routing_result.get(
                        "priority",
                        "HIGH"
                    )
                )

            st.write("**Routing Reason**")

            st.write(
                routing_result.get(
                    "reason",
                    "Manual review is required."
                )
            )

            st.warning(
                """
                👤 Human investigator review is required.

                The available evidence should be treated as
                incomplete. The investigator must manually verify
                the case before making a final decision.
                """
            )

            # --------------------------------------------------
            # SHOW PARTIAL EVIDENCE IF AVAILABLE
            # --------------------------------------------------

            st.divider()

            st.subheader("🔎 Available Investigation Evidence")

            partial_transaction = investigation.get(
                "transaction_agent"
            )

            partial_behavior = investigation.get(
                "behavior_agent"
            )

            partial_device = investigation.get(
                "device_channel_agent"
            )

            if partial_transaction:

                with st.expander(
                    "Transaction Analysis Agent",
                    expanded=False
                ):

                    st.markdown(
                        partial_transaction
                    )

            if partial_behavior:

                with st.expander(
                    "Customer Behaviour Agent",
                    expanded=False
                ):

                    st.markdown(
                        partial_behavior
                    )

            if partial_device:

                with st.expander(
                    "Device & Channel Agent",
                    expanded=False
                ):

                    st.markdown(
                        partial_device
                    )

            st.info(
                """
                The investigation stopped after the failed stage.
                No automated fraud conclusion has been generated
                from the incomplete investigation.
                """
            )

        else:

            # ==================================================
            # SUCCESSFUL INVESTIGATION
            # ==================================================

            result = investigation.get(
                "policy_result",
                {}
            )

            routing_result = investigation.get(
                "routing_result",
                {}
            )

            transaction_analysis = investigation.get(
                "transaction_agent",
                ""
            )

            behavior_analysis = investigation.get(
                "behavior_agent",
                ""
            )

            device_channel_analysis = investigation.get(
                "device_channel_agent",
                ""
            )

            recommendation = investigation.get(
                "recommendation_agent",
                ""
            )

            # ==================================================
            # RISK ASSESSMENT
            # ==================================================

            st.divider()

            st.subheader("Risk Assessment")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Risk Score",
                    f"{result.get('risk_score', 'N/A')}/100"
                )

            with col2:

                st.metric(
                    "Risk Level",
                    result.get(
                        "risk_level",
                        "N/A"
                    )
                )

            with col3:

                st.metric(
                    "Recommended Action",
                    result.get(
                        "recommended_action",
                        "N/A"
                    )
                )

            # --------------------------------------------------
            # CONDITIONAL ROUTING
            # --------------------------------------------------

            st.divider()

            st.subheader(
                "🔀 Conditional Routing"
            )

            route_col1, route_col2, route_col3 = st.columns(3)

            with route_col1:

                st.write("**Route**")

                st.info(
                    routing_result.get(
                        "route",
                        "N/A"
                    )
                )

            with route_col2:

                st.write("**Workflow Status**")

                st.info(
                    routing_result.get(
                        "workflow_status",
                        "N/A"
                    )
                )

            with route_col3:

                st.write("**Priority**")

                st.info(
                    routing_result.get(
                        "priority",
                        "N/A"
                    )
                )

            st.write("**Routing Reason**")

            st.write(
                routing_result.get(
                    "reason",
                    "No routing reason available."
                )
            )

            # --------------------------------------------------
            # WORKFLOW RESULT
            # --------------------------------------------------

            if investigation.get("n8n_result"):

                st.divider()

                st.subheader("Workflow Result")

                n8n_data = investigation.get(
                    "n8n_result",
                    {}
                )

                execution_col1, execution_col2, execution_col3 = (
                    st.columns(3)
                )

                with execution_col1:

                    st.write("**Outcome**")

                    st.info(
                        n8n_data.get(
                            "outcome",
                            routing_result.get(
                                "route",
                                "N/A"
                            )
                        )
                    )

                with execution_col2:

                    st.write("**Human Review Required**")

                    st.info(
                        "Yes"
                        if investigation.get(
                            "human_review_required",
                            routing_result.get(
                                "requires_human_review",
                                False
                            )
                        )
                        else "No"
                    )

                with execution_col3:

                    st.write("**Escalation Required**")

                    st.info(
                        "Yes"
                        if investigation.get(
                            "escalation_required",
                            False
                        )
                        else "No"
                    )

            if routing_result.get(
                "requires_human_review",
                False
            ):

                st.warning(
                    "👤 Human investigator review is required for this case."
                )

            else:

                st.success(
                    "✅ This case does not require human review under the current policy routing."
                )

            # --------------------------------------------------
            # RISK INDICATORS
            # --------------------------------------------------

            st.subheader("Risk Indicators")

            indicators = result.get(
                "indicators",
                []
            )

            if indicators:

                for indicator in indicators:

                    st.warning(
                        f"⚠️ {indicator}"
                    )

            else:

                st.success(
                    "No significant risk indicators detected."
                )

            # --------------------------------------------------
            # TRANSACTION ANALYSIS AGENT
            # --------------------------------------------------

            st.divider()

            st.subheader(
                "🤖 Transaction Analysis Agent"
            )

            st.caption(
                "AI-generated analysis based only on the "
                "transaction and customer evidence provided "
                "to the agent."
            )

            if transaction_analysis:

                st.markdown(
                    transaction_analysis
                )

            else:

                st.info(
                    "The Transaction Analysis Agent executed in the "
                    "n8n workflow, but its detailed text was not "
                    "included in the current webhook response."
                )

            # --------------------------------------------------
            # CUSTOMER BEHAVIOUR AGENT
            # --------------------------------------------------

            st.divider()

            st.subheader(
                "🧠 Customer Behaviour Agent"
            )

            st.caption(
                "AI-generated behavioural analysis comparing "
                "the current transaction with the customer's "
                "known behavioural profile."
            )

            if behavior_analysis:

                st.markdown(
                    behavior_analysis
                )

            else:

                st.info(
                    "The Customer Behaviour Agent executed in the "
                    "n8n workflow, but its detailed text was not "
                    "included in the current webhook response."
                )

            # --------------------------------------------------
            # DEVICE & CHANNEL AGENT
            # --------------------------------------------------

            st.divider()

            st.subheader(
                "📱 Device & Channel Agent"
            )

            st.caption(
                "AI-generated analysis of device trust, device "
                "consistency and transaction channel evidence."
            )

            if device_channel_analysis:

                st.markdown(
                    device_channel_analysis
                )

            else:

                st.info(
                    "The Device & Channel Agent executed in the "
                    "n8n workflow, but its detailed text was not "
                    "included in the current webhook response."
                )

            # --------------------------------------------------
            # RECOMMENDATION AGENT
            # --------------------------------------------------

            st.divider()

            st.subheader(
                "🎯 Recommendation Agent"
            )

            st.caption(
                "AI-generated investigation recommendation based "
                "on the outputs of the specialised agents and the "
                "deterministic policy engine."
            )

            if recommendation:

                st.markdown(
                    recommendation
                )

            else:

                st.info(
                    "No Recommendation Agent text was returned "
                    "by the current workflow response."
                )

            # --------------------------------------------------
            # GOVERNANCE NOTE
            # --------------------------------------------------

            st.divider()

            st.subheader(
                "🛡️ Governance Notice"
            )

            st.info(
                """
                The AI agents provide analytical evidence and
                investigation support. They do not make the final
                fraud decision.

                The deterministic policy engine calculates the
                current risk classification and recommended routing.

                Final action remains subject to human investigation
                and review.
                """
            )

            # --------------------------------------------------
            # PROCEED TO HUMAN REVIEW
            # --------------------------------------------------
            # The investigation result and selected case are already
            # stored in session state. The callback changes the sidebar
            # navigation state and the app reruns, opening Human Review
            # with the same case and evidence.

            st.divider()

            st.subheader("Next Step")

            st.caption(
                "Review the AI-generated evidence and policy assessment, "
                "then proceed to the investigator decision stage."
            )

            st.button(
                "👤 Proceed to Human Review",
                type="primary",
                use_container_width=True,
                on_click=navigate_to_human_review,
                key="proceed_to_human_review"
            )


# ==================================================
# AGENT ANALYSIS PAGE
# ==================================================

elif page == "Agent Analysis":

    st.header("🤖 Agent Analysis")

    st.info(
        """
        FraudGuard AI currently uses multiple specialised
        analytical agents and governed workflow components
        within the investigation process.
        """
    )

    st.subheader("Active Investigation Components")

    # --------------------------------------------------
    # 1. TRANSACTION ANALYSIS AGENT
    # --------------------------------------------------

    st.write(
        """
        **1. Transaction Analysis Agent**

        Analyses transaction-level evidence, identifies
        observable anomalies, highlights missing information
        and provides an explainable investigation analysis.
        """
    )

    st.divider()

    # --------------------------------------------------
    # 2. CUSTOMER BEHAVIOUR AGENT
    # --------------------------------------------------

    st.write(
        """
        **2. Customer Behaviour Agent**

        Compares the current transaction with the customer's
        known behavioural profile, including transaction
        amount, location, channel, merchant and behavioural
        patterns.
        """
    )

    st.divider()

    # --------------------------------------------------
    # 3. DEVICE & CHANNEL AGENT
    # --------------------------------------------------

    st.write(
        """
        **3. Device & Channel Agent**

        Analyses device trust, device consistency, transaction
        channel and location-related device/channel signals.
        """
    )

    st.divider()

    # --------------------------------------------------
    # 4. DETERMINISTIC POLICY ENGINE
    # --------------------------------------------------

    st.write(
        """
        **4. Deterministic Policy Engine**

        Applies predefined fraud-risk rules and produces the
        policy-based risk score, risk level and recommended
        routing.
        """
    )

    st.divider()

    # --------------------------------------------------
    # 5. RECOMMENDATION AGENT
    # --------------------------------------------------

    st.write(
        """
        **5. Recommendation Agent**

        Synthesises the outputs of the specialised investigation
        agents and the deterministic policy engine to provide
        investigative guidance for the human investigator.

        The Recommendation Agent does not independently determine
        whether fraud is confirmed.
        """
    )

    st.divider()

    # --------------------------------------------------
    # 6. INVESTIGATION ORCHESTRATOR
    # --------------------------------------------------

    st.write(
        """
        **6. Investigation Orchestrator**

        Coordinates the specialised agents, policy engine,
        recommendation stage and routing logic into a single
        investigation workflow.
        """
    )

    st.divider()

    # --------------------------------------------------
    # 7. CONDITIONAL ROUTING
    # --------------------------------------------------

    st.write(
        """
        **7. Conditional Routing**

        Uses the deterministic policy result to route cases
        to AUTO_CLOSE, HUMAN_REVIEW or ESCALATE.

        Cases requiring human intervention are not automatically
        closed by the workflow.
        """
    )

    st.divider()

    # --------------------------------------------------
    # 8. EXCEPTION HANDLING
    # --------------------------------------------------

    st.write(
        """
        **8. Exception Handling**

        Detects critical failures during the investigation
        workflow.

        If a required investigation stage fails, the workflow
        does not treat the case as low risk or automatically
        close it.

        Instead, the case is routed to:

        **MANUAL_REVIEW → EXCEPTION_REVIEW → HIGH priority**

        The investigator is informed that the available evidence
        may be incomplete and that manual verification is required.
        """
    )


# ==================================================
# HUMAN REVIEW PAGE
# ==================================================

elif page == "Human Review":

    st.header("👤 Human Review")

    st.info(
        """
        Human-in-the-loop review is the final decision stage
        of the FraudGuard AI investigation workflow.

        AI agents provide analytical evidence and recommendations.
        The final decision remains with the human investigator.
        """
    )

    # --------------------------------------------------
    # CHECK INVESTIGATION RESULT
    # --------------------------------------------------

    if "investigation_result" not in st.session_state:

        st.warning(
            """
            No completed investigation is available.

            Please go to the Investigation Dashboard,
            select a case and run the Fraud Investigation
            before starting Human Review.
            """
        )

    else:

        investigation = st.session_state[
            "investigation_result"
        ]

        selected_case = st.session_state.get(
            "selected_case",
            st.session_state.get(
                "last_selected_case",
                "Unknown"
            )
        )

        # --------------------------------------------------
        # GET INVESTIGATION STATUS
        # --------------------------------------------------

        investigation_status = investigation.get(
            "status",
            "SUCCESS"
        )

        # --------------------------------------------------
        # GET INVESTIGATION COMPONENTS
        # --------------------------------------------------

        policy_result = investigation.get(
            "policy_result",
            {}
        )

        routing_result = investigation.get(
            "routing_result",
            {}
        )

        recommendation = investigation.get(
            "recommendation_agent",
            ""
        )

        transaction_analysis = investigation.get(
            "transaction_agent",
            ""
        )

        behavior_analysis = investigation.get(
            "behavior_agent",
            ""
        )

        device_channel_analysis = investigation.get(
            "device_channel_agent",
            ""
        )

        # --------------------------------------------------
        # EXCEPTION WARNING
        # --------------------------------------------------

        if investigation_status == "ERROR":

            st.warning(
                """
                ⚠️ This case reached Human Review because the
                automated investigation encountered an exception.

                The investigator should treat the available evidence
                as incomplete and perform manual verification before
                making a final decision.
                """
            )

            error_details = investigation.get(
                "error",
                {}
            )

            st.write(
                "**Investigation Status:** ERROR"
            )

            st.write(
                "**Failed Stage:**",
                error_details.get(
                    "stage",
                    "Unknown"
                )
            )

            st.write(
                "**Error Type:**",
                error_details.get(
                    "error_type",
                    "Unknown"
                )
            )

        # --------------------------------------------------
        # GET CASE DETAILS
        # --------------------------------------------------

        case = test_cases[
            test_cases["case_id"] == selected_case
        ]

        if not case.empty:

            case = case.iloc[0]

            transaction_id = case["transaction_id"]

            transaction = transactions[
                transactions["transaction_id"]
                == transaction_id
            ].iloc[0]

            customer = customers[
                customers["customer_id"]
                == transaction["customer_id"]
            ].iloc[0]

        else:

            transaction_id = "Unknown"
            transaction = None
            customer = None

        # ==================================================
        # CASE SUMMARY
        # ==================================================

        st.divider()

        st.subheader("Investigation Case")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Case ID",
                selected_case
            )

        with col2:

            st.metric(
                "Transaction ID",
                transaction_id
            )

        with col3:

            st.metric(
                "Risk Score",
                f"{policy_result.get('risk_score', 'N/A')}/100"
            )

        with col4:

            st.metric(
                "Risk Level",
                policy_result.get(
                    "risk_level",
                    "N/A"
                )
            )

        # ==================================================
        # WORKFLOW ROUTING
        # ==================================================

        st.divider()

        st.subheader("🔀 Workflow Routing")

        route_col1, route_col2, route_col3 = st.columns(3)

        with route_col1:

            st.write("**Route**")

            st.info(
                routing_result.get(
                    "route",
                    "MANUAL_REVIEW"
                    if investigation_status == "ERROR"
                    else "N/A"
                )
            )

        with route_col2:

            st.write("**Workflow Status**")

            st.info(
                routing_result.get(
                    "workflow_status",
                    "EXCEPTION_REVIEW"
                    if investigation_status == "ERROR"
                    else "N/A"
                )
            )

        with route_col3:

            st.write("**Priority**")

            st.info(
                routing_result.get(
                    "priority",
                    "HIGH"
                    if investigation_status == "ERROR"
                    else "N/A"
                )
            )

        st.write("**Routing Reason**")

        st.write(
            routing_result.get(
                "reason",
                "Manual review is required."
            )
        )

        # ==================================================
        # POLICY ROUTING
        # ==================================================

        st.divider()

        st.subheader("⚖️ Deterministic Policy Assessment")

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Risk Score**")

            st.info(
                f"{policy_result.get('risk_score', 'N/A')}/100"
            )

        with col2:

            st.write("**Policy Recommended Action**")

            st.info(
                policy_result.get(
                    "recommended_action",
                    "N/A"
                )
            )

        # --------------------------------------------------
        # RISK INDICATORS
        # --------------------------------------------------

        indicators = policy_result.get(
            "indicators",
            []
        )

        if indicators:

            st.write("**Policy Risk Indicators**")

            for indicator in indicators:

                st.warning(
                    f"⚠️ {indicator}"
                )

        # ==================================================
        # RECOMMENDATION AGENT
        # ==================================================

        if recommendation:

            st.divider()

            st.subheader("🎯 Recommendation Agent")

            st.caption(
                """
                This is an AI-generated investigative recommendation.
                It does not constitute the final fraud decision.
                """
            )

            st.markdown(
                recommendation
            )

        # ==================================================
        # AGENT EVIDENCE
        # ==================================================

        st.divider()

        st.subheader("🔎 Investigation Evidence")

        if transaction_analysis:

            with st.expander(
                "Transaction Analysis Agent",
                expanded=False
            ):

                st.markdown(
                    transaction_analysis
                )

        if behavior_analysis:

            with st.expander(
                "Customer Behaviour Agent",
                expanded=False
            ):

                st.markdown(
                    behavior_analysis
                )

        if device_channel_analysis:

            with st.expander(
                "Device & Channel Agent",
                expanded=False
            ):

                st.markdown(
                    device_channel_analysis
                )

        # ==================================================
        # HUMAN DECISION
        # ==================================================

        st.divider()

        st.subheader("👤 Investigator Decision")

        st.warning(
            """
            Review the evidence above before submitting a decision.

            The AI-generated recommendation and policy result are
            supporting inputs only. The investigator remains
            responsible for the final decision.
            """
        )

        if investigation_status == "ERROR":

            st.warning(
                """
                Because this case reached Human Review through
                exception handling, verify the transaction,
                customer and device information manually before
                finalising the decision.
                """
            )

        decision = st.radio(
            "Select investigation decision",
            [
                "Approve / No Further Action",
                "Escalate for Investigation",
                "Block / Hold Transaction",
                "Request More Information"
            ],
            key=f"human_decision_{selected_case}"
        )

        # --------------------------------------------------
        # INVESTIGATOR COMMENT
        # --------------------------------------------------

        investigator_comment = st.text_area(
            "Investigator Justification",
            placeholder=(
                "Enter the reason for your decision. "
                "Reference the relevant evidence, risk indicators "
                "or missing information."
            ),
            height=150,
            key=f"investigator_comment_{selected_case}"
        )

        # ==================================================
        # SUBMIT HUMAN DECISION
        # ==================================================

        if st.button(
            "✅ Submit Human Decision",
            type="primary",
            use_container_width=True
        ):

            if not investigator_comment.strip():

                st.error(
                    "Please provide a justification before submitting the decision."
                )

            else:

                # ------------------------------------------
                # AUDIT EVENT
                # ------------------------------------------

                audit_event = log_event(
                    event_type="HUMAN_REVIEW_DECISION",
                    case_id=selected_case,
                    transaction_id=transaction_id,
                    customer_id=(
                        transaction["customer_id"]
                        if transaction is not None
                        else None
                    ),
                    risk_score=policy_result.get(
                        "risk_score"
                    ),
                    risk_level=policy_result.get(
                        "risk_level"
                    ),
                    recommended_action=policy_result.get(
                        "recommended_action"
                    ),
                    human_decision=decision,
                    investigator_comment=(
                        investigator_comment.strip()
                    ),
                    metadata={
                        "review_source": "FraudGuard AI",
                        "investigation_status": investigation_status,
                        "workflow_route": routing_result.get(
                            "route"
                        ),
                        "workflow_status": routing_result.get(
                            "workflow_status"
                        ),
                        "workflow_priority": routing_result.get(
                            "priority"
                        ),
                        "policy_indicators": indicators
                    }
                )

                # ------------------------------------------
                # STORE HUMAN REVIEW
                # ------------------------------------------

                st.session_state[
                    "human_review"
                ] = audit_event

                st.success(
                    "Human decision submitted and recorded in the audit log."
                )

        # ==================================================
        # PREVIOUS HUMAN DECISION
        # ==================================================

        if "human_review" in st.session_state:

            review = st.session_state[
                "human_review"
            ]

            if review.get("case_id") == selected_case:

                st.divider()

                st.subheader(
                    "📋 Submitted Human Review"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        "**Decision**"
                    )

                    st.info(
                        review.get(
                            "human_decision",
                            "N/A"
                        )
                    )

                with col2:

                    st.write(
                        "**Submitted At**"
                    )

                    st.info(
                        review.get(
                            "timestamp",
                            "N/A"
                        )
                    )

                st.write(
                    "**Investigator Justification**"
                )

                st.write(
                    review.get(
                        "investigator_comment",
                        ""
                    )
                )

                st.caption(
                    """
                    This review has been recorded as an audit event.
                    """
                )


# ==================================================
# MODEL COMPARISON PAGE
# ==================================================

elif page == "Model Comparison":

    st.header("📊 Model Comparison")

    # --------------------------------------------------
    # EVALUATION TEST CASES
    # --------------------------------------------------

    st.subheader("Evaluation Test Cases")

    evaluation_cases = test_cases[
        [
            "case_id",
            "case_type",
            "ground_truth",
            "expected_action"
        ]
    ].copy()

    st.dataframe(
        evaluation_cases,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------
    # EVALUATION MODE
    # --------------------------------------------------

    st.subheader("Evaluation Configuration")

    evaluation_mode_label = st.radio(
        "Evaluation Mode",
        [
            "Single Case Evaluation",
            "Full Model Evaluation"
        ],
        horizontal=True,
        key="evaluation_mode",
        help=(
            "Single Case Evaluation runs exactly one test case and is "
            "recommended during development to reduce API usage. Full Model "
            "Evaluation runs every configured test case for the selected model."
        )
    )

    evaluation_mode = (
        "single"
        if evaluation_mode_label == "Single Case Evaluation"
        else "full"
    )

    selected_evaluation_model = st.selectbox(
        "AI Model",
        list(MODELS.keys()),
        index=0,
        key="evaluation_model_name"
    )

    selected_evaluation_model_config = MODELS[
        selected_evaluation_model
    ]

    selected_evaluation_model_provider = (
        selected_evaluation_model_config.get(
            "provider",
            "openrouter"
        )
    )

    selected_evaluation_model_id = (
        selected_evaluation_model_config.get("model")
    )

    selected_evaluation_case = None

    if evaluation_mode == "single":

        selected_evaluation_case = st.selectbox(
            "Test Case",
            test_cases["case_id"].astype(str).tolist(),
            index=0,
            key="evaluation_case_id",
            help=(
                "Only this case will be sent through the model. "
                "Run another case later to build cumulative coverage."
            )
        )

        selected_case_row = test_cases[
            test_cases["case_id"].astype(str) == str(
                selected_evaluation_case
            )
        ].iloc[0]

        case_info_col1, case_info_col2, case_info_col3 = st.columns(3)

        with case_info_col1:
            st.metric(
                "Case Type",
                str(selected_case_row["case_type"])
            )

        with case_info_col2:
            st.metric(
                "Expected Risk",
                str(selected_case_row["ground_truth"])
            )

        with case_info_col3:
            st.metric(
                "Expected Action",
                str(selected_case_row["expected_action"])
            )

    else:

        st.caption(
            f"All {len(test_cases)} configured test cases will be evaluated "
            "for the selected model."
        )

    st.caption(
        f"Provider: **{selected_evaluation_model_provider}**  |  "
        f"Model ID: **{selected_evaluation_model_id}**"
    )

    # --------------------------------------------------
    # RUN MODEL EVALUATION
    # --------------------------------------------------

    run_label = (
        "▶ Run Selected Case"
        if evaluation_mode == "single"
        else "▶ Run Full Model Evaluation"
    )

    if st.button(
        run_label,
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Running selected case..."
            if evaluation_mode == "single"
            else "Running full model evaluation..."
        ):

            try:
                evaluation_result = run_model_evaluation(
                    test_cases=test_cases,
                    transactions=transactions,
                    customers=customers,
                    devices=devices,
                    selected_model_name=selected_evaluation_model,
                    selected_model_config=selected_evaluation_model_config,
                    evaluation_mode=evaluation_mode,
                    selected_case_id=selected_evaluation_case
                )

            except Exception as error:
                evaluation_result = {
                    "status": "ERROR",
                    "error": str(error),
                    "evaluation_mode": evaluation_mode
                }

        if evaluation_result.get("status") == "ERROR":

            st.session_state[
                "model_evaluation_result"
            ] = evaluation_result

            st.error(
                evaluation_result.get(
                    "error",
                    "Model evaluation failed."
                )
            )

        else:

            # Save both successful and failed case attempts. Failed provider
            # calls are retained for execution-health/audit purposes but are
            # not used as model-quality failures in cumulative metrics.
            history_entry = append_evaluation_history(
                evaluation_result,
                selected_model_name=selected_evaluation_model,
                selected_model_config=selected_evaluation_model_config
            )

            evaluation_result[
                "history_entry"
            ] = history_entry

            st.session_state[
                "model_evaluation_result"
            ] = evaluation_result

            successful_cases = int(
                evaluation_result["summary"].iloc[0][
                    "Successful Cases"
                ]
            )

            failed_cases = int(
                evaluation_result["summary"].iloc[0][
                    "Failed Cases"
                ]
            )

            if failed_cases:
                st.warning(
                    f"Evaluation attempt saved. {successful_cases} case(s) "
                    f"completed successfully and {failed_cases} case(s) "
                    "failed at execution/provider level."
                )
            else:
                st.success(
                    "Evaluation completed and the case result was saved to history."
                )

    # --------------------------------------------------
    # CURRENT EVALUATION RESULTS
    # --------------------------------------------------

    if "model_evaluation_result" in st.session_state:

        evaluation_result = st.session_state[
            "model_evaluation_result"
        ]

        if evaluation_result.get("status") != "ERROR":

            summary_df = evaluation_result["summary"]
            case_results_df = evaluation_result["case_results"]
            summary_row = summary_df.iloc[0]

            st.divider()
            st.subheader("Current Evaluation Results")

            st.caption(
                "These metrics describe the cases executed in the latest run. "
                "API/provider failures are reported separately under Execution Health."
            )

            def _metric_text(value, suffix="", decimals=1):
                if value is None or pd.isna(value):
                    return "N/A"
                return f"{float(value):.{decimals}f}{suffix}"

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Classification Quality",
                    _metric_text(
                        summary_row["Classification Accuracy (%)"],
                        "%"
                    )
                )

            with col2:
                st.metric(
                    "Escalation Accuracy",
                    _metric_text(
                        summary_row["Escalation Accuracy (%)"],
                        "%"
                    )
                )

            with col3:
                st.metric(
                    "False Positive Rate",
                    _metric_text(
                        summary_row["False Positive Rate (%)"],
                        "%"
                    )
                )

            with col4:
                st.metric(
                    "Hallucination Rate",
                    _metric_text(
                        summary_row[
                            "Hallucination / Grounding Violation Rate (%)"
                        ],
                        "%"
                    )
                )

            col5, col6, col7, col8 = st.columns(4)

            with col5:
                st.metric(
                    "Explanation Quality",
                    _metric_text(
                        summary_row["Explanation Quality (/10)"],
                        "/10"
                    )
                )

            with col6:
                st.metric(
                    "Structured Output",
                    _metric_text(
                        summary_row["Structured Output Compliance (%)"],
                        "%"
                    )
                )

            with col7:
                st.metric(
                    "Average Latency",
                    _metric_text(
                        summary_row["Average Latency (s)"],
                        " s",
                        decimals=2
                    )
                )

            with col8:
                st.metric(
                    "Task Completion",
                    _metric_text(
                        summary_row["Task Completion (%)"],
                        "%"
                    )
                )

            st.caption(
                f"Model: {summary_row['Model']} | "
                f"Provider: {summary_row.get('Provider', 'N/A')} | "
                f"Run mode: {evaluation_result.get('evaluation_mode', 'N/A')}"
            )

            # --------------------------------------------------
            # EXECUTION HEALTH
            # --------------------------------------------------

            st.subheader("Execution Health")

            health_col1, health_col2, health_col3, health_col4 = st.columns(4)

            with health_col1:
                st.metric(
                    "Cases Attempted",
                    int(summary_row["Total Test Cases"])
                )

            with health_col2:
                st.metric(
                    "Successful Cases",
                    int(summary_row["Successful Cases"])
                )

            with health_col3:
                st.metric(
                    "Execution Success",
                    f"{float(summary_row['Execution Success Rate (%)']):.1f}%"
                )

            with health_col4:
                st.metric(
                    "Execution Failure",
                    f"{float(summary_row['Execution Failure Rate (%)']):.1f}%"
                )

            if int(summary_row["Failed Cases"]) > 0:
                st.warning(
                    "Execution/provider failures are not treated as hallucinations "
                    "or incorrect model classifications. Retry the failed case when "
                    "API capacity is available."
                )

            # --------------------------------------------------
            # CASE-LEVEL RESULTS
            # --------------------------------------------------

            st.divider()
            st.subheader("Current Run — Case-Level Results")

            case_display_columns = [
                "case_id",
                "expected_risk",
                "predicted_risk",
                "expected_action",
                "predicted_action",
                "classification_correct",
                "escalation_correct",
                "false_positive",
                "task_completed",
                "structured_compliance_pct",
                "explanation_quality",
                "grounding_violations",
                "latency_seconds",
                "status",
                "error"
            ]

            available_columns = [
                column
                for column in case_display_columns
                if column in case_results_df.columns
            ]

            st.dataframe(
                case_results_df[available_columns],
                use_container_width=True,
                hide_index=True
            )

            failed_cases = case_results_df[
                case_results_df["status"].astype(str).str.upper() == "ERROR"
            ] if "status" in case_results_df.columns else pd.DataFrame()

            if not failed_cases.empty:
                st.subheader("Failed Case Details")

                for _, failed_case in failed_cases.iterrows():
                    case_id = failed_case.get("case_id", "Unknown")
                    error_message = failed_case.get(
                        "error",
                        "Unknown error"
                    )

                    with st.expander(
                        f"{case_id} — Evaluation Error"
                    ):
                        st.error(str(error_message))
                        st.caption(
                            "This is an execution/provider failure. It is not "
                            "counted as a hallucination or model-quality error."
                        )

            # --------------------------------------------------
            # CUMULATIVE MODEL RESULTS
            # --------------------------------------------------

            st.divider()
            st.subheader("Cumulative Model Results")

            current_history = load_evaluation_history()

            cumulative_df = build_cumulative_summary(
                history=current_history,
                selected_model_name=selected_evaluation_model,
                provider=selected_evaluation_model_provider,
                total_configured_cases=len(test_cases)
            )

            cumulative_row = cumulative_df.iloc[0]
            cumulative_evaluated = int(
                cumulative_row["Cases Evaluated"]
            )
            cumulative_configured = int(
                cumulative_row["Configured Test Cases"]
            )
            cumulative_remaining = int(
                cumulative_row["Cases Remaining"]
            )

            st.caption(
                f"Latest successful result retained for each unique case. "
                f"Coverage: {cumulative_evaluated}/{cumulative_configured} cases. "
                f"{cumulative_remaining} case(s) remaining."
            )

            cum_col1, cum_col2, cum_col3, cum_col4 = st.columns(4)

            with cum_col1:
                st.metric(
                    "Classification Quality",
                    _metric_text(
                        cumulative_row["Classification Accuracy (%)"],
                        "%"
                    )
                )

            with cum_col2:
                st.metric(
                    "Escalation Accuracy",
                    _metric_text(
                        cumulative_row["Escalation Accuracy (%)"],
                        "%"
                    )
                )

            with cum_col3:
                st.metric(
                    "False Positive Rate",
                    _metric_text(
                        cumulative_row["False Positive Rate (%)"],
                        "%"
                    )
                )

            with cum_col4:
                st.metric(
                    "Hallucination Rate",
                    _metric_text(
                        cumulative_row[
                            "Hallucination / Grounding Violation Rate (%)"
                        ],
                        "%"
                    )
                )

            cum_col5, cum_col6, cum_col7, cum_col8 = st.columns(4)

            with cum_col5:
                st.metric(
                    "Explanation Quality",
                    _metric_text(
                        cumulative_row["Explanation Quality (/10)"],
                        "/10"
                    )
                )

            with cum_col6:
                st.metric(
                    "Structured Output",
                    _metric_text(
                        cumulative_row["Structured Output Compliance (%)"],
                        "%"
                    )
                )

            with cum_col7:
                st.metric(
                    "Average Latency",
                    _metric_text(
                        cumulative_row["Average Latency (s)"],
                        " s",
                        decimals=2
                    )
                )

            with cum_col8:
                st.metric(
                    "Task Completion",
                    _metric_text(
                        cumulative_row["Task Completion (%)"],
                        "%"
                    )
                )

            if cumulative_evaluated < cumulative_configured:
                st.info(
                    "Cumulative metrics are based only on completed cases. "
                    "They should not be interpreted as a full benchmark until "
                    "all configured test cases have been evaluated."
                )
            else:
                st.success(
                    "All configured test cases have a successful result for "
                    "this model. The cumulative metrics now cover the full test set."
                )

    # --------------------------------------------------
    # MODEL EVALUATION HISTORY
    # --------------------------------------------------

    st.divider()
    st.subheader("Model Evaluation History")

    history = load_evaluation_history()

    if not history:
        st.info("No model evaluations have been recorded yet.")

    else:
        history_rows = []

        for entry in history:
            case_ids = entry.get("case_ids", [])
            if isinstance(case_ids, list):
                case_label = ", ".join(str(case_id) for case_id in case_ids)
            else:
                case_label = str(case_ids or "")

            history_rows.append({
                "Timestamp": entry.get("timestamp", ""),
                "Mode": entry.get("evaluation_mode", "legacy"),
                "Model": entry.get("model", ""),
                "Provider": entry.get("provider", ""),
                "Cases": case_label,
                "Successful": entry.get("successful_cases", ""),
                "Failed": entry.get("failed_cases", ""),
                "Classification Quality (%)": entry.get(
                    "classification_quality"
                ),
                "Escalation Accuracy (%)": entry.get(
                    "escalation_accuracy"
                ),
                "False Positive Rate (%)": entry.get(
                    "false_positive_rate"
                ),
                "Hallucination Rate (%)": entry.get(
                    "hallucination_rate"
                ),
                "Explanation Quality (/10)": entry.get(
                    "explanation_quality"
                ),
                "Structured Output (%)": entry.get(
                    "structured_output_compliance"
                ),
                "Average Latency (s)": entry.get(
                    "average_latency"
                ),
                "Task Completion (%)": entry.get(
                    "task_completion"
                )
            })

        history_df = pd.DataFrame(history_rows)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        with st.expander("View Evaluation Details"):

            history_index = st.selectbox(
                "Select Evaluation",
                range(len(history)),
                format_func=lambda index: (
                    f"{history[index].get('timestamp', '')} | "
                    f"{history[index].get('model', '')} | "
                    f"{history[index].get('evaluation_mode', 'legacy')}"
                ),
                key="evaluation_history_selector"
            )

            st.json(history[history_index])

# GOVERNANCE PAGE
# ==================================================

elif page == "Governance":

    st.header("🛡️ Governance")

    st.subheader("AI Governance Principles")

    governance_items = [

        (
            "Human-in-the-Loop",
            "Final investigation decisions remain with human investigators."
        ),

        (
            "Explainability",
            "AI analysis provides supporting evidence and explanations."
        ),

        (
            "Evidence Grounding",
            "Agents are instructed to use only the evidence provided."
        ),

        (
            "Deterministic Policy",
            "Risk routing is supported by predefined policy rules."
        ),

        (
            "Conditional Routing",
            "Cases are routed according to deterministic risk classification and workflow rules. In the n8n orchestration flow, LOW, MEDIUM and HIGH outcomes are routed to AUTO_CLOSE, HUMAN_REVIEW and ESCALATE respectively."
        ),

        (
            "Exception Handling",
            "Failed or incomplete investigations are not treated as low risk. Critical workflow failures are routed to manual review with exception-review status and high priority. The n8n orchestration layer also returns workflow failures to the Streamlit application for controlled review."
        ),

        (
            "Auditability",
            "Investigation decisions and human review actions are recorded for later review."
        ),

        (
            "Data Protection",
            "The prototype uses synthetic banking data."
        )
    ]

    for title, description in governance_items:

        st.markdown(
            f"### {title}"
        )

        st.write(
            description
        )

        st.divider()


# ==================================================
# AUDIT LOG PAGE
# ==================================================

elif page == "Audit Log":

    st.header("📋 Audit Log")

    st.info(
        """
        FraudGuard AI records important investigation events
        so that the investigation process can be reviewed later.
        """
    )

    # --------------------------------------------------
    # LOAD AUDIT EVENTS
    # --------------------------------------------------

    audit_logs = get_audit_logs()

    if not audit_logs:

        st.warning(
            "No audit events have been recorded yet."
        )

    else:

        st.subheader(
            f"Recorded Events: {len(audit_logs)}"
        )

        # --------------------------------------------------
        # DISPLAY AUDIT EVENTS
        # --------------------------------------------------

        audit_table = []

        for event in reversed(audit_logs):

            audit_table.append(
                {
                    "Timestamp": event.get(
                        "timestamp",
                        ""
                    ),

                    "Event": event.get(
                        "event_type",
                        ""
                    ),

                    "Case ID": event.get(
                        "case_id",
                        ""
                    ),

                    "Transaction ID": event.get(
                        "transaction_id",
                        ""
                    ),

                    "Customer ID": event.get(
                        "customer_id",
                        ""
                    ),

                    "Risk Score": event.get(
                        "risk_score",
                        ""
                    ),

                    "Risk Level": event.get(
                        "risk_level",
                        ""
                    ),

                    "Policy Action": event.get(
                        "recommended_action",
                        ""
                    ),

                    "Human Decision": event.get(
                        "human_decision",
                        ""
                    ),

                    "Investigator Comment": event.get(
                        "investigator_comment",
                        ""
                    )
                }
            )

        audit_df = pd.DataFrame(
            audit_table
        )

        st.dataframe(
            audit_df,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # EVENT DETAILS
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "Audit Event Details"
        )

        selected_event_index = st.selectbox(
            "Select Audit Event",
            range(
                len(audit_logs)
            ),
            format_func=lambda index: (
                f"{audit_logs[::-1][index].get('timestamp', '')} | "
                f"{audit_logs[::-1][index].get('event_type', '')} | "
                f"Case {audit_logs[::-1][index].get('case_id', '')}"
            )
        )

        selected_event = audit_logs[::-1][
            selected_event_index
        ]

        st.json(
            selected_event
        )
