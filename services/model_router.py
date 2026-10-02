from services.openrouter import call_openrouter
from services.groq import call_groq


def call_model(
    provider,
    model,
    system_prompt,
    user_prompt,
    temperature=0.1,
    max_tokens=800
):
    """
    Provider-independent model invocation.

    The agents do not need to know whether the selected
    model is hosted by OpenRouter or Groq.
    """

    provider = str(provider).lower().strip()

    if provider == "openrouter":

        return call_openrouter(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

    if provider == "groq":

        return call_groq(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort="low"
        )

    raise ValueError(
        f"Unsupported model provider: {provider}"
    )