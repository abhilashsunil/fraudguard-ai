from datetime import datetime


def calculate_risk(transaction, customer, device):

    score = 0
    indicators = []

    amount = float(transaction["amount"])
    average = float(customer["avg_transaction"])

    # Rule 1: transaction amount
    if amount > 3 * average:
        score += 25
        indicators.append(
            "Transaction exceeds 3x customer average"
        )

    # Rule 2: device
    if str(device["trusted"]).lower() != "true":
        score += 20
        indicators.append(
            "New or untrusted device detected"
        )

    # Rule 3: location
    if transaction["location"] != customer["home_city"]:
        score += 40
        indicators.append(
            "Transaction location differs from usual location"
        )

    # Rule 4: unusual time
    timestamp = datetime.strptime(
        str(transaction["timestamp"]),
        "%Y-%m-%d %H:%M:%S"
    )

    if timestamp.hour < 6 or timestamp.hour >= 23:
        score += 10
        indicators.append(
            "Transaction occurred during unusual hours"
        )

    # Risk classification
    if score >= 70:
        risk_level = "HIGH"
        action = "ESCALATE"

    elif score >= 40:
        risk_level = "MEDIUM"
        action = "HUMAN_REVIEW"

    else:
        risk_level = "LOW"
        action = "AUTO_CLOSE"

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "recommended_action": action,
        "indicators": indicators
    }