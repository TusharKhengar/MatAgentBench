#!/usr/bin/env python3
"""Quick test of Groq API to find the correct endpoint and model."""
import os
import requests

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("[ERROR] GROQ_API_KEY environment variable not set")
    print("Set it in your .env file or export GROQ_API_KEY=your_key_here")
    exit(1)

# Test different endpoints and models
tests = [
    ("https://api.groq.com/openai/v1/chat/completions", "llama3-70b-8192"),
    ("https://api.groq.com/openai/v1/chat/completions", "llama3-8b-8192"),
    ("https://api.groq.com/openai/v1/chat/completions", "mixtral-8x7b-32768"),
    ("https://api.groq.com/openai/v1/chat/completions", "gemma-7b-it"),
]

for url, model in tests:
    print(f"\n[TEST] {model} at {url}")
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10
            },
            timeout=10
        )
        print(f"[STATUS] {response.status_code}")
        if response.status_code == 200:
            print(f"[SUCCESS] Model {model} works!")
            print(f"[RESPONSE] {response.json()}")
            break
        else:
            error = response.json() if response.headers.get('content-type') == 'application/json' else response.text
            print(f"[ERROR] {error}")
    except Exception as e:
        print(f"[EXCEPTION] {e}")
