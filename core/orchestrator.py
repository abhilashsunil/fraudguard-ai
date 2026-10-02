import pandas as pd

from agents.transaction_agent import analyze_transaction
from agents.customer_behavior_agent import analyze_customer_behavior
from agents.device_channel_agent import analyze_device_channel
from agents.recommendation_agent import generate_recommendation

from core.policy_engine import calculate_risk
from core.routing_engine import determine_route

from core.exception_handler import (
    create_investigation_error
)


def run_investigation(
    transaction,
    customer,
    device,
    model,
    provider="openrouter"
):

    partial_results = {}

    # ==================================================
    # PREPARE INPUT DATA
    # ==================================================

    try:

        if isinstance(
            transaction,
            pd.Series
        ):
            transaction_data = (
                transaction.to_dict()
            )

        else:
            transaction_data = transaction


        if isinstance(
            customer,
            pd.Series
        ):
            customer_data = (
                customer.to_dict()
            )

        else:
            customer_data = customer


        if isinstance(
            device,
            pd.Series
        ):
            device_data = (
                device.to_dict()
            )

        else:
            device_data = device


        # --------------------------------------------------
        # BASIC DATA VALIDATION
        # --------------------------------------------------

        if not transaction_data:
            raise ValueError(
                "Transaction data is missing."
            )

        if not customer_data:
            raise ValueError(
                "Customer data is missing."
            )

        if not device_data:
            raise ValueError(
                "Device data is missing."
            )


    except Exception as error:

        return create_investigation_error(
            stage="Input Validation",
            error=error,
            partial_results=partial_results
        )


    # ==================================================
    # TRANSACTION ANALYSIS AGENT
    # ==================================================

    try:

        transaction_analysis = analyze_transaction(
            transaction=transaction_data,
            customer=customer_data,
            model=model,
            provider=provider
        )

        partial_results[
            "transaction_agent"
        ] = transaction_analysis

    except Exception as error:

        return create_investigation_error(
            stage="Transaction Analysis Agent",
            error=error,
            partial_results=partial_results
        )


    # ==================================================
    # CUSTOMER BEHAVIOUR AGENT
    # ==================================================

    try:

        behavior_analysis = (
            analyze_customer_behavior(
                transaction=transaction_data,
                customer=customer_data,
                device=device_data,
                model=model,
                provider=provider
            )
        )

        partial_results[
            "behavior_agent"
        ] = behavior_analysis

    except Exception as error:

        return create_investigation_error(
            stage="Customer Behaviour Agent",
            error=error,
            partial_results=partial_results
        )


    # ==================================================
    # DEVICE & CHANNEL AGENT
    # ==================================================

    try:

        device_channel_analysis = (
            analyze_device_channel(
                transaction=transaction_data,
                customer=customer_data,
                device=device_data,
                model=model,
                provider=provider
            )
        )

        partial_results[
            "device_channel_agent"
        ] = device_channel_analysis

    except Exception as error:

        return create_investigation_error(
            stage="Device & Channel Agent",
            error=error,
            partial_results=partial_results
        )


    # ==================================================
    # PREPARE POLICY INPUT
    # ==================================================

    try:

        transaction_series = (
            transaction
            if isinstance(
                transaction,
                pd.Series
            )
            else pd.Series(
                transaction_data
            )
        )

        customer_series = (
            customer
            if isinstance(
                customer,
                pd.Series
            )
            else pd.Series(
                customer_data
            )
        )

        device_series = (
            device
            if isinstance(
                device,
                pd.Series
            )
            else pd.Series(
                device_data
            )
        )

    except Exception as error:

        return create_investigation_error(
            stage="Policy Input Preparation",
            error=error,
            partial_results=partial_results
        )


    # ==================================================
    # DETERMINISTIC POLICY ENGINE
    # ==================================================

    try:

        policy_result = calculate_risk(
            transaction_series,
            customer_series,
            device_series
        )

        partial_results[
            "policy_result"
        ] = policy_result

    except Exception as error:

        return create_investigation_error(
            stage="Deterministic Policy Engine",
            error=error,
            partial_results=partial_results
        )


    # ==================================================
    # CONDITIONAL ROUTING
    # ==================================================

    try:

        routing_result = determine_route(
            policy_result
        )

        partial_results[
            "routing_result"
        ] = routing_result

    except Exception as error:

        return create_investigation_error(
            stage="Conditional Routing Engine",
            error=error,
            partial_results=partial_results
        )


    # ==================================================
    # RECOMMENDATION AGENT
    # ==================================================

    try:

        recommendation = generate_recommendation(
            transaction=transaction_data,
            customer=customer_data,
            device=device_data,
            transaction_analysis=(
                transaction_analysis
            ),
            behavior_analysis=(
                behavior_analysis
            ),
            device_channel_analysis=(
                device_channel_analysis
            ),
            policy_result=(
                policy_result
            ),
            model=model,
            provider=provider
        )

        partial_results[
            "recommendation_agent"
        ] = recommendation

    except Exception as error:

        return create_investigation_error(
            stage="Recommendation Agent",
            error=error,
            partial_results=partial_results
        )


    # ==================================================
    # SUCCESSFUL INVESTIGATION
    # ==================================================

    return {
        "status": "SUCCESS",

        "transaction_agent": (
            transaction_analysis
        ),

        "behavior_agent": (
            behavior_analysis
        ),

        "device_channel_agent": (
            device_channel_analysis
        ),

        "policy_result": (
            policy_result
        ),

        "routing_result": (
            routing_result
        ),

        "recommendation_agent": (
            recommendation
        )
    }