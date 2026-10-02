import os
import time
import requests
import streamlit as st


OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)

DEFAULT_HTTP_REFERER = (
    "https://fraudguard-ai-demo.streamlit.app"
)

DEFAULT_APP_TITLE = "FraudGuard AI"


def _get_api_key():
    """
    Retrieve the OpenRouter API key.

    Priority:
    1. Environment variable - used by cloud/backend deployments
    2. Streamlit secrets - used by Streamlit Cloud/local Streamlit
    """

    api_key = os.getenv("OPENROUTER_API_KEY")

    if api_key:
        return api_key

    try:
        api_key = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        api_key = None

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured. "
            "Set it as an environment variable or "
            "in Streamlit secrets."
        )

    return api_key


def _get_http_referer():
    """
    Get the application URL used for OpenRouter attribution.

    Allows the backend deployment to override the value through
    OPENROUTER_HTTP_REFERER.
    """

    return os.getenv(
        "OPENROUTER_HTTP_REFERER",
        DEFAULT_HTTP_REFERER
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

    Supports:
    - Environment-based API keys for cloud deployments
    - Streamlit secrets for Streamlit deployments
    - 429 retry handling
    - Request timeout handling
    - Network error handling
    - Provider fallback
    """

    api_key = _get_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": _get_http_referer(),
        "X-Title": DEFAULT_APP_TITLE
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

            # -------------------------------------------------
            # SUCCESS
            # -------------------------------------------------
            if response.status_code == 200:

                try:
                    data = response.json()
                except ValueError as exc:
                    raise RuntimeError(
                        "OpenRouter returned invalid JSON "
                        f"for model '{model}': {exc}"
                    ) from exc

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

            # -------------------------------------------------
            # RATE LIMIT
            # -------------------------------------------------
            if response.status_code == 429:

                try:
                    error_body = response.json()
                except Exception:
                    error_body = response.text

                last_error = (
                    "OpenRouter API error | "
                    "HTTP 429 | "
                    f"Model: {model} | "
                    f"Response: {error_body}"
                )

                if attempt < max_retries:

                    retry_after = response.headers.get(
                        "retry-after"
                    )

                    if retry_after:
                        try:
                            wait_time = float(retry_after)
                        except (ValueError, TypeError):
                            wait_time = 3 * (2 ** attempt)
                    else:
                        wait_time = 3 * (2 ** attempt)

                    # Cap retry delay to avoid excessively long waits.
                    wait_time = min(wait_time, 60)

                    print(
                        f"[OpenRouter] Rate limit reached "
                        f"for {model}. "
                        f"Retrying in {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/"
                        f"{max_retries})"
                    )

                    time.sleep(wait_time)
                    continue

                raise RuntimeError(last_error)

            # -------------------------------------------------
            # OTHER API ERRORS
            # -------------------------------------------------
            try:
                error_body = response.json()
            except Exception:
                error_body = response.text

            raise RuntimeError(
                "OpenRouter API error | "
                f"HTTP {response.status_code} | "
                f"Model: {model} | "
                f"Response: {error_body}"
            )

        # -----------------------------------------------------
        # TIMEOUT
        # -----------------------------------------------------
        except requests.exceptions.Timeout as exc:

            last_error = (
                f"OpenRouter request timed out "
                f"for model '{model}': {exc}"
            )

            if attempt < max_retries:

                wait_time = min(
                    3 * (2 ** attempt),
                    60
                )

                print(
                    f"[OpenRouter] Request timeout "
                    f"for {model}. "
                    f"Retrying in {wait_time}s "
                    f"(attempt {attempt + 1}/"
                    f"{max_retries})"
                )

                time.sleep(wait_time)
                continue

            raise RuntimeError(last_error) from exc

        # -----------------------------------------------------
        # NETWORK ERROR
        # -----------------------------------------------------
        except requests.exceptions.RequestException as exc:

            last_error = (
                f"OpenRouter network error "
                f"for model '{model}': {exc}"
            )

            if attempt < max_retries:

                wait_time = min(
                    3 * (2 ** attempt),
                    60
                )

                print(
                    f"[OpenRouter] Network error "
                    f"for {model}: {exc}. "
                    f"Retrying in {wait_time}s "
                    f"(attempt {attempt + 1}/"
                    f"{max_retries})"
                )

                time.sleep(wait_time)
                continue

            raise RuntimeError(last_error) from exc

    raise RuntimeError(
        last_error or
        f"OpenRouter request failed for model '{model}'."
    )