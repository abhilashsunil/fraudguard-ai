# FraudGuard AI

## Governed Agentic AI for Banking Fraud Investigation

FraudGuard AI is an academic prototype for investigating suspicious banking transactions using a multi-agent AI architecture combined with deterministic policy controls and human-in-the-loop governance.

## Architecture

Fraud Alert
→ Transaction Analysis Agent
→ Customer Behaviour Agent
→ Device & Channel Agent
→ Deterministic Policy Engine
→ Recommendation Agent
→ Human Review / Escalation
→ Final Outcome

## Key Components

- Transaction Analysis Agent
- Customer Behaviour Agent
- Device & Channel Agent
- Recommendation Agent
- Deterministic Risk & Policy Engine
- Conditional Routing
- Human-in-the-Loop Review
- Audit Logging
- Model Evaluation

## Technology Stack

- Python
- Streamlit
- n8n
- OpenRouter
- Groq
- Pandas
- Plotly

## Project Structure

```text
agents/       AI investigation agents
core/         Policy, routing and orchestration
data/         Test data and evaluation data
evaluation/   Model evaluation framework
services/     LLM provider integrations
utils/        Utility functions
app.py        Streamlit application
fraudguard_api.py  API layer