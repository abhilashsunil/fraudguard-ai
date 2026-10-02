from services.model_router import call_model


# --------------------------------------------------
# RECOMMENDATION AGENT SYSTEM PROMPT
# --------------------------------------------------

RECOMMENDATION_AGENT_SYSTEM_PROMPT = """
You are the Recommendation Agent in FraudGuard AI,
a governed banking fraud investigation system.

Your role is to synthesise investigation evidence from
specialised agents and the deterministic policy engine.

You provide investigation guidance to a human fraud investigator.

IMPORTANT GOVERNANCE RULES:

1. Do not declare that fraud has been confirmed.
2. Do not invent evidence.
3. Use only the information provided.
4. Treat the deterministic policy result as the authoritative
   risk-routing input.
5. Clearly distinguish observed evidence from missing information.
6. The final decision must remain with a human investigator.
7. If evidence is incomplete, explicitly state what needs
   to be verified.
8. Recommendations must be investigative guidance, not an
   autonomous fraud decision.
"""


# --------------------------------------------------
# RECOMMENDATION AGENT
# --------------------------------------------------

def generate_recommendation(
    transaction,
    customer,
    device,
    transaction_analysis,
    behavior_analysis,
    device_channel_analysis,
    policy_result,
    model="nvidia/nemotron-3-super-120b-a12b:free",
    provider="openrouter"
):
    """
    Recommendation Agent

    Synthesises the outputs of:
    1. Transaction Analysis Agent
    2. Customer Behaviour Agent
    3. Device & Channel Agent
    4. Deterministic Policy Engine

    The agent provides an investigation recommendation.
    It does not make the final fraud decision.
    """

    # --------------------------------------------------
    # USER PROMPT
    # --------------------------------------------------

    user_prompt = f"""
--------------------------------------------------
TRANSACTION
--------------------------------------------------

{transaction}

--------------------------------------------------
CUSTOMER
--------------------------------------------------

{customer}

--------------------------------------------------
DEVICE
--------------------------------------------------

{device}

--------------------------------------------------
TRANSACTION ANALYSIS AGENT
--------------------------------------------------

{transaction_analysis}

--------------------------------------------------
CUSTOMER BEHAVIOUR AGENT
--------------------------------------------------

{behavior_analysis}

--------------------------------------------------
DEVICE & CHANNEL AGENT
--------------------------------------------------

{device_channel_analysis}

--------------------------------------------------
DETERMINISTIC POLICY RESULT
--------------------------------------------------

{policy_result}

--------------------------------------------------
REQUIRED OUTPUT
--------------------------------------------------

Provide the response using exactly these sections:

## 1. Investigation Recommendation

Explain the recommended investigation route based on the
available evidence and deterministic policy result.

Do not confirm fraud.

## 2. Risk Context

Summarise the important risk signals identified across
the transaction, customer behaviour, device/channel and
policy analysis.

## 3. Key Supporting Evidence

List the strongest pieces of evidence supporting the
recommendation.

Only use evidence provided above.

## 4. Consistent Behaviour

Identify transaction characteristics that are consistent
with the customer's known behaviour.

## 5. Missing Information

Identify information that should be verified before a
final decision can be made.

## 6. Recommended Verification Steps

Provide specific actions a human investigator should perform.

## 7. Human Review Requirement

Clearly explain why human review is required and what the
investigator should verify or decide.

The system must not make the final fraud decision.

## 8. Final Note

State that the recommendation is investigative guidance
and does not itself confirm fraud.

Do not add any additional sections.
"""

    # --------------------------------------------------
    # CALL OPENROUTER
    # --------------------------------------------------

    response = call_model(
        provider=provider,
        model=model,
        system_prompt=RECOMMENDATION_AGENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=800
    )

    return response