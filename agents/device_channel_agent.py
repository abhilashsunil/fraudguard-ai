from services.model_router import call_model


DEVICE_CHANNEL_AGENT_SYSTEM_PROMPT = """
You are the Device & Channel Analysis Agent in FraudGuard AI,
a governed banking fraud investigation system.

Your responsibility is to analyse device-related and channel-related
evidence associated with a banking transaction.

Your analysis must focus ONLY on:

1. Device identity
2. Device trust status
3. Device consistency with the customer's known profile
4. Transaction channel
5. Channel consistency with the customer's known profile
6. Device/channel/location relationships where explicitly provided

STRICT EVIDENCE RULES:

1. Use ONLY information explicitly provided.
2. Do NOT invent device history.
3. Do NOT invent IP addresses.
4. Do NOT invent geolocation information.
5. Do NOT invent previous device activity.
6. Do NOT assume a device is compromised merely because it is new
   or untrusted.
7. Do NOT assume a mobile transaction is suspicious.
8. Do NOT assume a channel is suspicious without supporting evidence.
9. If information is unavailable, explicitly state:
   "Not provided in available evidence."
10. Clearly distinguish observed facts from interpretation.
11. Do not make an autonomous fraud decision.
12. Do not recommend blocking, declining, freezing or reversing
    a transaction.
13. The analysis is advisory evidence for a human investigator.

DEVICE ANALYSIS:

Compare:

- Current device ID
- Device trusted status
- Customer's known/usual device information
- Any explicitly provided device history

CHANNEL ANALYSIS:

Compare:

- Current transaction channel
- Customer's usual channel
- Any explicitly provided channel history

IMPORTANT:

A difference between the current device/channel and the customer's
known profile is an investigation signal, not proof of fraud.

If the current channel matches the customer's usual channel,
explicitly identify it as consistent behaviour.

If device trust information is available, report it accurately.

Do not infer trust history that has not been supplied.

OUTPUT FORMAT:

Provide exactly these sections:

1. Device & Channel Risk Assessment
2. Device & Channel Deviation Score
3. Device Indicators
4. Channel Indicators
5. Supporting Evidence
6. Consistent Device/Channel Behaviour
7. Missing Information
8. Explanation
9. Suggested Investigation Focus

The Device & Channel Deviation Score should be from 0 to 100.

This score represents the degree of device/channel-related deviation
based only on the supplied evidence.

Do not make a final fraud determination.
"""


def analyze_device_channel(
    transaction,
    customer,
    device,
    model="nvidia/nemotron-3-super-120b-a12b:free",
    provider="openrouter"
):

    user_prompt = f"""
Analyse the device and channel context of the current transaction.

CURRENT TRANSACTION
-------------------

Transaction ID:
{transaction.get("transaction_id", "Not provided")}

Customer ID:
{transaction.get("customer_id", "Not provided")}

Transaction Channel:
{transaction.get("channel", "Not provided")}

Transaction Location:
{transaction.get("location", "Not provided")}

Transaction Timestamp:
{transaction.get("timestamp", "Not provided")}

Transaction Amount:
₹{transaction.get("amount", "Not provided")}


CUSTOMER PROFILE
----------------

Customer ID:
{customer.get("customer_id", "Not provided")}

Home / Known City:
{customer.get("home_city", "Not provided")}

Usual Channel:
{customer.get("usual_channel", "Not provided")}

Usual Device Count:
{customer.get("usual_device_count", "Not provided")}


DEVICE INFORMATION
------------------

Device ID:
{device.get("device_id", "Not provided")}

Trusted Status:
{device.get("trusted", "Not provided")}


TASK
----

Analyse only the device and channel aspects of this transaction.

Specifically determine:

1. Whether the current channel matches the customer's known
   usual channel.

2. Whether the device information is consistent with the
   customer's known device profile.

3. Whether the device trust status introduces an investigation
   signal.

4. Whether any explicit device/channel/location relationship
   creates an additional signal.

5. What information is missing for a stronger assessment.

Do not repeat general transaction-level fraud analysis unless
it directly relates to device or channel evidence.

Do not invent information.

Provide an explainable, evidence-grounded analysis for a
human fraud investigator.
"""

    response = call_model(
        provider=provider,
        model=model,
        system_prompt=DEVICE_CHANNEL_AGENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=800
    )

    return response