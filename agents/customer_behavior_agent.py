from services.model_router import call_model

CUSTOMER_BEHAVIOR_AGENT_SYSTEM_PROMPT = """
You are the Customer Behaviour Analysis Agent in FraudGuard AI,
a governed banking fraud investigation system.

Your responsibility is to compare the current transaction
against the customer's known historical behaviour.

Your analysis must focus specifically on behavioural deviation.

STRICT EVIDENCE RULES:

1. Use ONLY information explicitly provided.
2. Do NOT invent historical transactions.
3. Do NOT invent customer preferences.
4. Do NOT assume a normal transaction time unless it is explicitly
   provided.
5. Do NOT assume a customer location pattern beyond the supplied
   home/usual location.
6. Do NOT assume a merchant is suspicious.
7. Do NOT assume that a transaction is fraudulent.
8. If information is unavailable, explicitly state:
   "Not provided in available evidence."
9. Clearly distinguish observed facts from interpretation.
10. Do not make autonomous decisions about blocking, declining,
    reversing or freezing a transaction.
11. Your output is advisory evidence for a human investigator.

BEHAVIOURAL COMPARISONS:

Analyse the transaction against the available customer profile:

- Transaction amount vs historical average transaction amount.
- Transaction location vs customer's known home/usual location.
- Transaction channel vs customer's usual channel.
- Device usage vs customer's known usual device information.
- Transaction frequency where relevant information is available.
- Any other explicitly provided behavioural information.

IMPORTANT:

Do not treat every difference as fraud.

A behavioural difference should be described as an anomaly,
deviation or investigation signal rather than proof of fraud.

RISK SCORE:

Provide an analytical behavioural deviation score from 0 to 100.

This score represents the degree of behavioural deviation based
on the supplied evidence.

It must NOT override the deterministic policy engine.

OUTPUT FORMAT:

Provide these sections:

1. Behavioural Risk Assessment
2. Behavioural Deviation Score
3. Behavioural Anomalies
4. Supporting Evidence
5. Consistent Behaviour
6. Missing Information
7. Explanation
8. Suggested Investigation Focus

For every important conclusion, identify the supplied evidence
that supports it.

Do not make a final fraud determination.
"""


def analyze_customer_behavior(
    transaction,
    customer,
    device=None,
    model="nvidia/nemotron-3-super-120b-a12b:free",
    provider="openrouter"
):

    device_information = "Not provided in available evidence."

    if device is not None:

        device_information = (
            f"Device ID: {device.get('device_id', 'Not provided')}\n"
            f"Trusted Status: {device.get('trusted', 'Not provided')}"
        )

    user_prompt = f"""
Analyse the customer's behavioural deviation for the transaction
provided below.

CURRENT TRANSACTION
-------------------
Transaction ID:
{transaction.get("transaction_id", "Not provided")}

Customer ID:
{transaction.get("customer_id", "Not provided")}

Transaction Amount:
₹{transaction.get("amount", "Not provided")}

Transaction Location:
{transaction.get("location", "Not provided")}

Transaction Channel:
{transaction.get("channel", "Not provided")}

Transaction Timestamp:
{transaction.get("timestamp", "Not provided")}

Merchant:
{transaction.get("merchant", "Not provided")}

Merchant Category:
{transaction.get("merchant_category", "Not provided")}


CUSTOMER PROFILE
----------------
Customer ID:
{customer.get("customer_id", "Not provided")}

Home / Known City:
{customer.get("home_city", "Not provided")}

Historical Average Transaction:
₹{customer.get("avg_transaction", "Not provided")}

Monthly Transaction Count:
{customer.get("monthly_txn_count", "Not provided")}

Usual Channel:
{customer.get("usual_channel", "Not provided")}

Usual Device Count:
{customer.get("usual_device_count", "Not provided")}


DEVICE INFORMATION
------------------
{device_information}


TASK
----
Determine how the current transaction differs from the customer's
known behavioural profile.

Focus on measurable deviations.

Do not invent missing historical information.

Provide an explainable investigation-oriented analysis.
"""

    response = call_model(
        provider=provider,
        model=model,
        system_prompt=CUSTOMER_BEHAVIOR_AGENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=800
    )

    return response