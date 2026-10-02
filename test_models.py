from services.openrouter import call_openrouter

MODELS = {
    "Nemotron 3 Super":
        "nvidia/nemotron-3-super-120b-a12b:free"
}

for model_name, model_id in MODELS.items():

    print()
    print("=" * 70)
    print(f"Testing: {model_name}")
    print(f"Model ID: {model_id}")
    print("=" * 70)

    try:

        response = call_openrouter(
            model=model_id,
            system_prompt=(
                "You are a test assistant. "
                "Respond with exactly: API TEST SUCCESS."
            ),
            user_prompt=(
                "Return the required test response."
            ),
            temperature=0.1
        )

        print()
        print("STATUS: SUCCESS")
        print()
        print("MODEL RESPONSE:")
        print(response)

    except Exception as exc:

        print()
        print("STATUS: FAILED")
        print()
        print("ERROR:")
        print(exc)

    print()