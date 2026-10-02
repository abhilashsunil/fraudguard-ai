import time
import requests
import streamlit as st


OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)


def call_openrouter(
    model,
    system_prompt,
    user_prompt,
    temperature=0.1,
    max_tokens=800,
    max_retries=2
):
    """
    Call an OpenRouter-hosted model.

    Provider fallback is enabled because free OpenRouter
    endpoints can temporarily become unavailable.
    """

    api_key = st.secrets["OPENROUTER_API_KEY"]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "FraudGuard AI"
    }

    payload = {
        "model": model,

        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        "temperature": temperature,
        "max_tokens": max_tokens,

        "provider": {
            "allow_fallbacks": True,
            "sort": "throughput"
        }
    }

    last_error = None

    for attempt in range(max_retries + 1):

        try:

            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=120
            )

            if response.status_code == 200:

                data = response.json()

                if "choices" not in data:
                    raise RuntimeError(
                        f"OpenRouter returned an unexpected "
                        f"response for model '{model}': {data}"
                    )

                if not data["choices"]:
                    raise RuntimeError(
                        f"OpenRouter returned no choices "
                        f"for model '{model}'."
                    )

                message = data["choices"][0].get(
                    "message",
                    {}
                )

                content = message.get("content")

                if content is None:
                    raise RuntimeError(
                        f"OpenRouter returned no message content "
                        f"for model '{model}': {data}"
                    )

                return content

            if response.status_code == 429:

                try:
                    error_body = response.json()
                except Exception:
                    error_body = response.text

                last_error = (
                    f"OpenRouter API error | "
                    f"HTTP 429 | "
                    f"Model: {model} | "
                    f"Response: {error_body}"
                )

                if attempt < max_retries:

                    wait_time = 3 * (2 ** attempt)

                    time.sleep(wait_time)

                    continue

                raise RuntimeError(last_error)

            try:
                error_body = response.json()
            except Exception:
                error_body = response.text

            raise RuntimeError(
                f"OpenRouter API error | "
                f"HTTP {response.status_code} | "
                f"Model: {model} | "
                f"Response: {error_body}"
            )

        except requests.exceptions.Timeout:

            last_error = (
                f"OpenRouter request timed out "
                f"for model '{model}'."
            )

            if attempt < max_retries:

                wait_time = 3 * (2 ** attempt)

                time.sleep(wait_time)

                continue

            raise RuntimeError(last_error)

        except requests.exceptions.RequestException as exc:

            last_error = (
                f"OpenRouter network error "
                f"for model '{model}': {exc}"
            )

            if attempt < max_retries:

                wait_time = 3 * (2 ** attempt)

                time.sleep(wait_time)

                continue

            raise RuntimeError(last_error)

    raise RuntimeError(
        last_error or
        f"OpenRouter request failed for model '{model}'."
    )