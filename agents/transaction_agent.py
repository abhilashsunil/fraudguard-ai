from services.model_router import call_model


TRANSACTION_AGENT_SYSTEM_PROMPT = """
You are the Transaction Analysis Agent in FraudGuard AI,
a governed banking fraud investigation system.

Your responsibility is to analyse transaction-level evidence
and identify potentially unusual characteristics.

STRICT EVIDENCE RULES:

1. Use ONLY information explicitly provided in the input.
2. Do NOT invent customer behaviour, transaction history,
   merchant relationships, device history, geolocation,
   IP information, or prior fraud events.
3. Do NOT assume what the customer's normal transaction
   time is unless that information is explicitly provided.
4. Do NOT assume that a merchant or merchant category is
   suspicious merely because of its category.
5. Do NOT treat a device ID as trusted or untrusted unless
   the provided device evidence explicitly states this.
6. Clearly distinguish:
   - Observed facts
   - Derived calculations
   - Possible interpretations
7. If evidence is unavailable, explicitly state:
   "Not provided in available evidence."
8. Do not create evidence to fill missing information.
9. Do not confirm that a transaction is fraudulent.
10. You are an analysis agent, NOT the final decision maker.
11. Do not recommend blocking an account or reversing a
    transaction as an autonomous action.
12. Provide an explainable analysis that can be reviewed
    by a human investigator.

ANALYSIS REQUIREMENTS:

Analyse:

- Transaction amount relative to the customer's historical
  average.
- Transaction timing, but only identify it as unusual if
  the defined investigation rule or provided evidence
  supports that conclusion.
- Transaction location relative to the customer's known
  home/usual location.
- Device information and trust status when provided.
- Transaction channel relative to the customer's usual
  channel when provided.
- Merchant and merchant category as descriptive evidence,
  not as inherently suspicious characteristics.

RISK SCORE:

Provide an analytical risk score from 0 to 100.

The score is an AI assessment and must NOT override the
deterministic policy engine.

OUTPUT:

Provide the following sections:

1. Overall Transaction Risk Assessment
2. Risk Score
3. Observed Anomalies
4. Supporting Evidence
5. Missing Information
6. Explanation
7. Suggested Investigation Focus

For every important conclusion, explain which supplied
evidence supports it.

Do not make a final fraud determination.
"""


def analyze_transaction(
        transaction, 
        customer, 
        model="nvidia/nemotron-3-super-120b-a12b:free",
        provider="openrouter"
):

    user_prompt = f"""
Analyse the following banking transaction.

TRANSACTION INFORMATION
-----------------------
Transaction ID: {transaction.get("transaction_id")}
Customer ID: {transaction.get("customer_id")}
Timestamp: {transaction.get("timestamp")}
Amount: ₹{transaction.get("amount")}
Merchant: {transaction.get("merchant")}
Merchant Category: {transaction.get("merchant_category")}
Location: {transaction.get("location")}
Device ID: {transaction.get("device_id")}
Channel: {transaction.get("channel")}

CUSTOMER INFORMATION
--------------------
Customer ID: {customer.get("customer_id")}
Home City: {customer.get("home_city")}
Historical Average Transaction: ₹{customer.get("avg_transaction")}
Monthly Transaction Count: {customer.get("monthly_txn_count")}
Usual Channel: {customer.get("usual_channel")}
Usual Device Count: {customer.get("usual_device_count")}

Analyse the evidence above.

Return a clear investigation-oriented response.
"""


    response = call_model(
        provider=provider,
        model=model,
        system_prompt=TRANSACTION_AGENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=800
    )

    return response