#!/usr/bin/env python3
"""Test Groq using their official SDK."""
import os

try:
    from groq import Groq
    print("[INFO] Groq SDK installed")
except ImportError:
    print("[ERROR] Groq SDK not installed. Run: pip install groq")
    exit(1)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("[ERROR] GROQ_API_KEY environment variable not set")
    print("Set it in your .env file or export GROQ_API_KEY=your_key_here")
    exit(1)

try:
    client = Groq(api_key=api_key)

    print("[TEST] Calling Groq with official SDK...")
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": "Say hello"}
        ],
        max_tokens=20
    )

    print(f"[SUCCESS] Response: {completion.choices[0].message.content}")
    print(f"[MODEL] {completion.model}")

except Exception as e:
    print(f"[ERROR] {e}")
    print("\nTry these steps:")
    print("1. Go to https://console.groq.com/keys")
    print("2. Create a NEW API key")
    print("3. Make sure your account is verified (check email)")
