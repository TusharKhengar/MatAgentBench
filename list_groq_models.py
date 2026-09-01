#!/usr/bin/env python3
"""List all available models on Groq."""
import os
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("[ERROR] GROQ_API_KEY environment variable not set")
    print("Set it in your .env file or export GROQ_API_KEY=your_key_here")
    exit(1)

try:
    client = Groq(api_key=api_key)

    print("[INFO] Fetching available models from your Groq account...\n")

    models = client.models.list()

    print(f"[SUCCESS] Found {len(models.data)} models:\n")

    for model in models.data:
        print(f"  - {model.id}")
        if hasattr(model, 'owned_by'):
            print(f"    Owner: {model.owned_by}")
        print()

    # Try the first model
    if models.data:
        test_model = models.data[0].id
        print(f"[TEST] Testing with first available model: {test_model}")

        response = client.chat.completions.create(
            model=test_model,
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=10
        )

        print(f"[SUCCESS] Response: {response.choices[0].message.content}")
        print(f"\n✅ USE THIS MODEL: {test_model}")

except Exception as e:
    print(f"[ERROR] {e}")
