from evaluation.evaluator import _get_case_inputs
from core.orchestrator import run_investigation
from core.data_loader import load_data


MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
CASE_ID = "TC003"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

customers, transactions, devices, fraud_rules, test_cases = load_data()


# --------------------------------------------------
# GET TC003
# --------------------------------------------------

case = test_cases[
    test_cases["case_id"] == CASE_ID
].iloc[0]


transaction, customer, device = _get_case_inputs(
    case,
    transactions,
    customers,
    devices
)


print()
print("=" * 80)
print("DIAGNOSTIC TEST")
print("=" * 80)
print(f"Case      : {CASE_ID}")
print(f"Transaction: {case['transaction_id']}")
print(f"Model     : {MODEL}")
print("=" * 80)


# --------------------------------------------------
# RUN INVESTIGATION
# --------------------------------------------------

result = run_investigation(
    transaction=transaction,
    customer=customer,
    device=device,
    model=MODEL
)


# --------------------------------------------------
# PRINT COMPLETE RESULT
# --------------------------------------------------

print()
print("=" * 80)
print("WORKFLOW STATUS")
print("=" * 80)

print(result.get("status"))


print()
print("=" * 80)
print("ERROR")
print("=" * 80)

print(result.get("error"))


print()
print("=" * 80)
print("TRANSACTION AGENT")
print("=" * 80)

print(result.get("transaction_agent"))


print()
print("=" * 80)
print("CUSTOMER BEHAVIOUR AGENT")
print("=" * 80)

print(result.get("behavior_agent"))


print()
print("=" * 80)
print("DEVICE & CHANNEL AGENT")
print("=" * 80)

print(result.get("device_channel_agent"))


print()
print("=" * 80)
print("POLICY RESULT")
print("=" * 80)

print(result.get("policy_result"))


print()
print("=" * 80)
print("ROUTING RESULT")
print("=" * 80)

print(result.get("routing_result"))


print()
print("=" * 80)
print("RECOMMENDATION AGENT")
print("=" * 80)

print(result.get("recommendation_agent"))


print()
print("=" * 80)
print("END DIAGNOSTIC")
print("=" * 80)