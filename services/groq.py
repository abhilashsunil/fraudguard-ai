import os
import time
import random
import requests
import streamlit as st


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Maximum number of retries after the initial request
MAX_RETRIES = 3

# Maximum amount of time to wait for a single retry
MAX_RETRY_WAIT = 60

# Delay between successful Groq calls.
# Helps prevent bursts when several agents run sequentially.
POST_REQUEST_DELAY = 2


def _get_api_key():
    """
    Retrieve the Groq API key.

    Priority:
    1. Environment variable - used by cloud/backend deployments
    2. Streamlit secrets - used by Streamlit Cloud/local Streamlit
    """

    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key

    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Set it as an environment variable or "
            "in Streamlit secrets."
        )

    return api_key


def _get_retry_delay(response, attempt):
    """
    Determine how long to wait before retrying.

    Priority:
    1. Retry-After header from Groq
    2. Exponential backoff
    3. Small random jitter to avoid repeated collisions
    """

    retry_after = response.headers.get("retry-after")

    if retry_after:
        try:
            delay = float(retry_after)
        except (ValueError, TypeError):
            delay = 2 ** attempt
    else:
        delay = 2 ** attempt

    # Add a small amount of jitter
    delay += random.uniform(0.5, 1.5)

    # Prevent excessively long waits
    return min(delay, MAX_RETRY_WAIT)


def _rate_limit_details(response):
    """
    Extract useful Groq rate-limit information for diagnostics.
    """

    return {
        "retry_after": response.headers.get("retry-after"),
        "remaining_tokens": response.headers.get(
            "x-ratelimit-remaining-tokens"
        ),
        "reset_tokens": response.headers.get(
            "x-ratelimit-reset-tokens"
        ),
        "remaining_requests": response.headers.get(
            "x-ratelimit-remaining-requests"
        ),
        "reset_requests": response.headers.get(
            "x-ratelimit-reset-requests"
        )
    }


def call_groq(
    model,
    system_prompt,
    user_prompt,
    temperature=0.1,
    max_tokens=800,
    reasoning_effort="low"
):
    """
    Call a Groq-hosted model using the OpenAI-compatible
    Groq Chat Completions API.

    Handles:
    - 429 rate limits
    - temporary 5xx server errors
    - request timeouts
    - network errors
    - malformed API responses

    The function keeps the same interface expected by
    services.model_router.call_model().
    """

    api_key = _get_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
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
        "max_tokens": max_tokens
    }

    # Keep reasoning configurable.
    # This is currently used by the model router.
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    last_error = None

    for attempt in range(MAX_RETRIES + 1):

        try:

            response = requests.post(
                GROQ_URL,
                headers=headers,
                json=payload,
                timeout=120
            )

            # -------------------------------------------------
            # RATE LIMIT
            # -------------------------------------------------
            if response.status_code == 429:

                rate_info = _rate_limit_details(response)

                if attempt < MAX_RETRIES:

                    delay = _get_retry_delay(
                        response,
                        attempt
                    )

                    print(
                        f"[Groq] Rate limit reached for "
                        f"{model}. "
                        f"Retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/"
                        f"{MAX_RETRIES})"
                    )

                    time.sleep(delay)
                    continue

                # All retries exhausted
                raise RuntimeError(
                    "Groq API rate limit exceeded after "
                    f"{MAX_RETRIES} retries. "
                    f"Model: {model}. "
                    f"Retry-After: "
                    f"{rate_info['retry_after']}. "
                    f"Remaining tokens: "
                    f"{rate_info['remaining_tokens']}. "
                    f"Token reset: "
                    f"{rate_info['reset_tokens']}. "
                    f"Remaining requests: "
                    f"{rate_info['remaining_requests']}. "
                    f"Request reset: "
                    f"{rate_info['reset_requests']}."
                )

            # -------------------------------------------------
            # TEMPORARY SERVER ERRORS
            # -------------------------------------------------
            if response.status_code in (500, 502, 503, 504):

                if attempt < MAX_RETRIES:

                    delay = min(
                        (2 ** attempt) + random.uniform(0.5, 1.5),
                        MAX_RETRY_WAIT
                    )

                    print(
                        f"[Groq] Temporary server error "
                        f"HTTP {response.status_code}. "
                        f"Retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/"
                        f"{MAX_RETRIES})"
                    )

                    time.sleep(delay)
                    continue

                raise RuntimeError(
                    "Groq temporary server error after "
                    f"{MAX_RETRIES} retries. "
                    f"HTTP {response.status_code}. "
                    f"Model: {model}."
                )

            # -------------------------------------------------
            # OTHER API ERRORS
            # -------------------------------------------------
            if response.status_code != 200:

                try:
                    error_body = response.json()
                except Exception:
                    error_body = response.text

                raise RuntimeError(
                    "Groq API error | "
                    f"HTTP {response.status_code} | "
                    f"Model: {model} | "
                    f"Response: {error_body}"
                )

            # -------------------------------------------------
            # PARSE RESPONSE
            # -------------------------------------------------
            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"Groq returned invalid JSON for "
                    f"model '{model}': {exc}"
                ) from exc

            if "choices" not in data:

                raise RuntimeError(
                    f"Groq returned an unexpected response "
                    f"for model '{model}': {data}"
                )

            if not data["choices"]:

                raise RuntimeError(
                    f"Groq returned no choices "
                    f"for model '{model}'."
                )

            message = data["choices"][0].get(
                "message",
                {}
            )

            content = message.get("content")

            if content is None:

                raise RuntimeError(
                    f"Groq returned no message content "
                    f"for model '{model}': {data}"
                )

            # Small delay before the next agent/request.
            # This reduces burst traffic in multi-agent workflows.
            time.sleep(POST_REQUEST_DELAY)

            return content

        # -----------------------------------------------------
        # TIMEOUT
        # -----------------------------------------------------
        except requests.exceptions.Timeout as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                delay = min(
                    (2 ** attempt) + random.uniform(0.5, 1.5),
                    MAX_RETRY_WAIT
                )

                print(
                    f"[Groq] Request timeout for "
                    f"{model}. "
                    f"Retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/"
                    f"{MAX_RETRIES})"
                )

                time.sleep(delay)
                continue

            raise RuntimeError(
                f"Groq request timed out after "
                f"{MAX_RETRIES} retries for model "
                f"'{model}'."
            ) from exc

        # -----------------------------------------------------
        # NETWORK ERROR
        # -----------------------------------------------------
        except requests.exceptions.RequestException as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                delay = min(
                    (2 ** attempt) + random.uniform(0.5, 1.5),
                    MAX_RETRY_WAIT
                )

                print(
                    f"[Groq] Network error for "
                    f"{model}: {exc}. "
                    f"Retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/"
                    f"{MAX_RETRIES})"
                )

                time.sleep(delay)
                continue

            raise RuntimeError(
                f"Groq network error after "
                f"{MAX_RETRIES} retries for "
                f"model '{model}': {exc}"
            ) from exc

    # Safety fallback
    raise RuntimeError(
        f"Groq request failed for model "
        f"'{model}'. Last error: {last_error}"
    )