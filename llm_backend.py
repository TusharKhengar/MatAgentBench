"""
Real LLM Backend Integration for MatAgentBench
Supports Groq, OpenRouter, and Omniroute APIs
"""

import json
import os
from typing import Any

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[WARNING] 'requests' module not found. Install with: pip install requests")


def call_groq_llm(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    """Call Groq API with the given prompt."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "[ERROR] GROQ_API_KEY environment variable not set. Get your free key at https://console.groq.com/keys"

    if not REQUESTS_AVAILABLE:
        return "[ERROR] 'requests' library not installed. Run: pip install requests"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a scientific reasoning agent. You solve physics and chemistry problems step-by-step. For each step, output: THOUGHT: (your reasoning), ACTION: (tool name), ARGS: (JSON arguments), then wait for OBSERVATION."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        # Debug: print status and response
        print(f"[DEBUG] Groq API Status: {response.status_code}")
        if response.status_code != 200:
            print(f"[DEBUG] Groq Response: {response.text[:500]}")

        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        error_detail = ""
        try:
            error_detail = e.response.json()
        except:
            error_detail = e.response.text[:200]
        return f"[ERROR] Groq API call failed: {e}. Details: {error_detail}"
    except Exception as e:
        return f"[ERROR] Groq API call failed: {str(e)}"


def call_openrouter_llm(prompt: str, model: str = "meta-llama/llama-3.3-70b-instruct:free") -> str:
    """Call OpenRouter free models."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return "[ERROR] OPENROUTER_API_KEY not set"

    if not REQUESTS_AVAILABLE:
        return "[ERROR] 'requests' library not installed"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/MatAgentBench",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a scientific AI agent solving physics/chemistry problems."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR] OpenRouter API call failed: {str(e)}"


def call_omniroute_llm(prompt: str, model: str = "amidia-ai") -> str:
    """Call local Omniroute proxy."""
    base_url = os.getenv("ANTHROPIC_BASE_URL", "http://localhost:20128")
    api_key = os.getenv("ANTHROPIC_AUTH_TOKEN", "dummy")

    if not REQUESTS_AVAILABLE:
        return "[ERROR] 'requests' library not installed"

    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a scientific AI agent."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR] Omniroute call failed: {str(e)}"


def parse_llm_response_to_steps(llm_output: str, formula: str) -> list[dict[str, Any]]:
    """Parse LLM free-form output into structured steps."""
    steps = []
    lines = llm_output.split("\n")

    current_step = {"thought": "", "action": "", "args": {}, "result": ""}
    step_idx = 0

    for line in lines:
        line_upper = line.upper().strip()

        if line_upper.startswith("THOUGHT:") or line_upper.startswith("REASONING:"):
            if current_step["thought"]:
                # Save previous step
                steps.append({
                    "step": step_idx,
                    "thought": current_step["thought"],
                    "action": current_step["action"] or "reasoning",
                    "args": current_step["args"],
                    "result": current_step["result"] or "Processing...",
                    "ok": True
                })
                step_idx += 1
            current_step = {
                "thought": line.split(":", 1)[1].strip() if ":" in line else line,
                "action": "",
                "args": {},
                "result": ""
            }

        elif line_upper.startswith("ACTION:"):
            current_step["action"] = line.split(":", 1)[1].strip() if ":" in line else "compute"

        elif line_upper.startswith("ARGS:") or line_upper.startswith("ARGUMENTS:"):
            try:
                args_str = line.split(":", 1)[1].strip() if ":" in line else "{}"
                current_step["args"] = json.loads(args_str)
            except:
                current_step["args"] = {"raw": line.split(":", 1)[1].strip() if ":" in line else ""}

        elif line_upper.startswith("OBSERVATION:") or line_upper.startswith("RESULT:"):
            current_step["result"] = line.split(":", 1)[1].strip() if ":" in line else line

        elif line_upper.startswith("FINAL ANSWER:") or line_upper.startswith("ANSWER:"):
            current_step["result"] = line.split(":", 1)[1].strip() if ":" in line else line

    # Save last step
    if current_step["thought"]:
        steps.append({
            "step": step_idx,
            "thought": current_step["thought"],
            "action": current_step["action"] or "final_answer",
            "args": current_step["args"],
            "result": current_step["result"] or llm_output[-200:],
            "ok": True
        })

    # If no structured steps found, treat whole output as one reasoning step
    if not steps:
        steps.append({
            "step": 0,
            "thought": llm_output[:500],
            "action": "llm_reasoning",
            "args": {"formula": formula},
            "result": llm_output[-500:] if len(llm_output) > 500 else llm_output,
            "ok": True
        })

    return steps
