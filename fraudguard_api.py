"""
FraudGuard AI - n8n Integration API
-----------------------------------
Additive integration layer. This file does NOT modify the existing
Streamlit application or core/orchestrator.py.

Run from the existing FraudGuard AI project root:
    python fraudguard_api.py

The API binds to 0.0.0.0:8000 so n8n in Docker Desktop can reach it
through http://host.docker.internal:8000.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pandas as pd

from agents.transaction_agent import analyze_transaction
from agents.customer_behavior_agent import analyze_customer_behavior
from agents.device_channel_agent import analyze_device_channel
from agents.recommendation_agent import generate_recommendation

from core.policy_engine import calculate_risk
from core.routing_engine import determine_route

HOST = "0.0.0.0"
PORT = 8000
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def require_common(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    transaction = payload.get("transaction")
    customer = payload.get("customer")
    device = payload.get("device")

    if not transaction:
        raise ValueError("transaction is required.")
    if not customer:
        raise ValueError("customer is required.")
    if not device:
        raise ValueError("device is required.")

    model = payload.get("model") or DEFAULT_MODEL
    provider = payload.get("provider") or "openrouter"

    return transaction, customer, device, model, provider


class Handler(BaseHTTPRequestHandler):

    def send_json(self, status, data):
        body = json.dumps(
            json_safe(data),
            ensure_ascii=False
        ).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )
        self.send_header(
            "Content-Length",
            str(len(body))
        )
        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)

        if not raw:
            return {}

        return json.loads(
            raw.decode("utf-8")
        )

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS"
        )
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/health":
            self.send_json(
                200,
                {
                    "status": "OK",
                    "service": "FraudGuard AI n8n Integration API"
                }
            )
            return

        self.send_json(
            404,
            {
                "status": "ERROR",
                "message": "Endpoint not found."
            }
        )

    def do_POST(self):
        path = urlparse(self.path).path

        try:
            payload = self.read_json()

            if path == "/agent/transaction":
                transaction, customer, _, model, provider = require_common(
                    payload
                )

                result = analyze_transaction(
                    transaction=transaction,
                    customer=customer,
                    model=model,
                    provider=provider
                )

                self.send_json(
                    200,
                    {
                        "status": "SUCCESS",
                        "agent": "Transaction Analysis Agent",
                        "provider": provider,
                        "model": model,
                        "result": result
                    }
                )
                return

            if path == "/agent/customer-behaviour":
                transaction, customer, device, model, provider = require_common(
                    payload
                )

                result = analyze_customer_behavior(
                    transaction=transaction,
                    customer=customer,
                    device=device,
                    model=model,
                    provider=provider
                )

                self.send_json(
                    200,
                    {
                        "status": "SUCCESS",
                        "agent": "Customer Behaviour Agent",
                        "provider": provider,
                        "model": model,
                        "result": result
                    }
                )
                return

            if path == "/agent/device-channel":
                transaction, customer, device, model, provider = require_common(
                    payload
                )

                result = analyze_device_channel(
                    transaction=transaction,
                    customer=customer,
                    device=device,
                    model=model,
                    provider=provider
                )

                self.send_json(
                    200,
                    {
                        "status": "SUCCESS",
                        "agent": "Device & Channel Agent",
                        "provider": provider,
                        "model": model,
                        "result": result
                    }
                )
                return

            if path == "/policy":
                transaction, customer, device, _, _ = require_common(
                    payload
                )

                policy = calculate_risk(
                    pd.Series(transaction),
                    pd.Series(customer),
                    pd.Series(device)
                )

                routing = determine_route(policy)

                self.send_json(
                    200,
                    {
                        "status": "SUCCESS",
                        "policy_result": policy,
                        "routing_result": routing
                    }
                )
                return

            if path == "/agent/recommendation":
                transaction, customer, device, model, provider = require_common(
                    payload
                )

                recommendation = generate_recommendation(
                    transaction=transaction,
                    customer=customer,
                    device=device,
                    transaction_analysis=payload.get(
                        "transaction_analysis",
                        ""
                    ),
                    behavior_analysis=payload.get(
                        "behavior_analysis",
                        ""
                    ),
                    device_channel_analysis=payload.get(
                        "device_channel_analysis",
                        ""
                    ),
                    policy_result=payload.get(
                        "policy_result",
                        {}
                    ),
                    model=model,
                    provider=provider
                )

                self.send_json(
                    200,
                    {
                        "status": "SUCCESS",
                        "agent": "Recommendation Agent",
                        "provider": provider,
                        "model": model,
                        "result": recommendation
                    }
                )
                return

            if path == "/investigate":
                # Backward-compatible convenience endpoint.
                # The n8n workflow itself uses the individual endpoints
                # so that n8n visibly performs the orchestration.
                from core.orchestrator import run_investigation

                transaction, customer, device, model, provider = require_common(
                    payload
                )

                result = run_investigation(
                    transaction=transaction,
                    customer=customer,
                    device=device,
                    model=model,
                    provider=provider
                )

                self.send_json(200, result)
                return

            self.send_json(
                404,
                {
                    "status": "ERROR",
                    "message": f"Unknown endpoint: {path}"
                }
            )

        except Exception as exc:
            self.send_json(
                500,
                {
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "path": path
                }
            )

    def log_message(self, format, *args):
        print(
            "[FraudGuard API]",
            format % args
        )


if __name__ == "__main__":
    print(
        "FraudGuard AI n8n Integration API listening on "
        "http://0.0.0.0:8000"
    )
    print(
        "Health check: http://localhost:8000/health"
    )

    ThreadingHTTPServer(
        (HOST, PORT),
        Handler
    ).serve_forever()
