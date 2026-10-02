import pandas as pd


def load_data():
    customers = pd.read_csv("data/customers.csv")
    transactions = pd.read_csv("data/transactions.csv")
    devices = pd.read_csv("data/devices.csv")
    fraud_rules = pd.read_csv("data/fraud_rules.csv")
    test_cases = pd.read_csv("data/test_cases.csv")

    return (
        customers,
        transactions,
        devices,
        fraud_rules,
        test_cases
    )