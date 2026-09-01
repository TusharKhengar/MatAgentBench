"""Pluggable LLM backends.

Every provider we use is OpenAI-chat-compatible, so one client covers Groq, Cerebras,
OpenRouter, Hugging Face and any local llama.cpp/Ollama server -- they differ only in
base URL, key and model id.

Two behaviours here exist specifically because we run on free tiers:

  * **Disk caching**, keyed by an exact hash of the request. A resumed sweep never pays
    for a completion it has already seen, which is what makes an interrupted overnight
    run cheap to restart.
  * **Rate-limit-aware retries** that honour `Retry-After`. Free tiers throttle hard;
    a sweep that dies on the first 429 is unusable.

Adding a paid frontier model later means adding one row to `PRESETS` -- nothing else.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from ..schema import ModelSpec


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMBackend(Protocol):
    """The whole contract. Implement this and the entire harness works against it."""

    spec: ModelSpec

    def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse: ...


# --------------------------------------------------------------------------------------
# Provider presets -- all free tiers, all open-weight models
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key_env: str
    open_weights: bool = True
    notes: str = ""


PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        "groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", True, "Free tier, very fast."
    ),
    "cerebras": Provider(
        "cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", True, "Free tier."
    ),
    "openrouter": Provider(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        True,
        "Use ':free' model suffixes.",
    ),
    "huggingface": Provider(
        "huggingface", "https://router.huggingface.co/v1", "HF_TOKEN", True, "Small free credit."
    ),
    "local": Provider("local", "", "", True, "Ollama / llama.cpp. Base URL from LOCAL_BASE_URL."),
}

# Verify current availability before a full sweep -- free-tier model catalogues move.
PRESETS: dict[str, tuple[str, str]] = {
    "large-open": ("cerebras", "gpt-oss-120b"),
    "large-open-alt": ("groq", "openai/gpt-oss-120b"),
    "mid-open": ("groq", "qwen/qwen3-32b"),
    "small-open": ("groq", "llama-3.1-8b-instant"),
    "local-small": ("local", os.getenv("LOCAL_MODEL", "qwen3:4b")),
}


class RateLimited(RuntimeError):
    pass


def _cache_key(model: str, messages: list[ChatMessage], params: dict[str, Any]) -> str:
    payload = json.dumps(
        {"model": model, "messages": [m.to_dict() for m in messages], "params": params},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OpenAICompatBackend:
    """Works against any OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        cache_dir: str | Path | None = None,
        timeout: float = 120.0,
        max_retries: int | None = None,
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}. Known: {sorted(PROVIDERS)}")
        self.provider = PROVIDERS[provider]
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries or int(os.getenv("MAB_MAX_RETRIES", "6"))

        if provider == "local":
            self.base_url = base_url or os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1")
            self.api_key = api_key or "not-needed"
        else:
            self.base_url = base_url or self.provider.base_url
            self.api_key = api_key or os.getenv(self.provider.api_key_env, "")
            if not self.api_key:
                raise RuntimeError(
                    f"No API key for {provider}. Set {self.provider.api_key_env} "
                    f"in .env or as a GitHub Actions secret."
                )

        cache_root = cache_dir or os.getenv("MAB_CACHE_DIR", ".mab_cache")
        self.cache_dir = Path(cache_root) / provider
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = httpx.Client(timeout=timeout)
        self.spec = ModelSpec(
            backend=provider,
            model=model,
            open_weights=self.provider.open_weights,
            temperature=temperature,
        )

    # -- cache ------------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> LLMResponse | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return LLMResponse(
            text=payload["text"],
            prompt_tokens=payload.get("prompt_tokens", 0),
            completion_tokens=payload.get("completion_tokens", 0),
            latency_ms=payload.get("latency_ms", 0.0),
            cached=True,
        )

    def _write_cache(self, key: str, response: LLMResponse) -> None:
        try:
            self._cache_path(key).write_text(
                json.dumps(
                    {
                        "text": response.text,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "latency_ms": response.latency_ms,
                        "model": self.model,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass  # a cache miss is recoverable; a crash mid-sweep is not

    # -- request ----------------------------------------------------------------------

    def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        params: dict[str, Any] = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if (stop := kwargs.get("stop")) is not None:
            params["stop"] = stop

        key = _cache_key(self.model, messages, params)
        if not kwargs.get("no_cache") and (hit := self._read_cache(key)) is not None:
            return hit

        body = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            **params,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            started = time.perf_counter()
            try:
                resp = self._client.post(url, json=body, headers=headers)
            except httpx.HTTPError as exc:
                last_error = exc
                self._backoff(attempt)
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("retry-after")
                last_error = RateLimited(f"{resp.status_code}: {resp.text[:200]}")
                self._backoff(attempt, retry_after)
                continue

            if resp.status_code >= 400:
                raise RuntimeError(
                    f"{self.provider.name} returned {resp.status_code}: {resp.text[:500]}"
                )

            payload = resp.json()
            latency_ms = (time.perf_counter() - started) * 1000
            usage = payload.get("usage") or {}
            try:
                text = payload["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError) as exc:
                raise RuntimeError(
                    f"Malformed response from {self.provider.name}: {payload}"
                ) from exc

            response = LLMResponse(
                text=text,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                latency_ms=latency_ms,
                raw=payload,
            )
            self._write_cache(key, response)
            return response

        raise RuntimeError(
            f"{self.provider.name} failed after {self.max_retries} attempts: {last_error}"
        )

    def _backoff(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                time.sleep(min(float(retry_after), 120.0))
                return
            except ValueError:
                pass
        time.sleep(min(2.0**attempt + random.uniform(0, 1.0), 90.0))

    def close(self) -> None:
        self._client.close()


class ReplayBackend:
    """Serves a recorded transcript, optionally diverging at a chosen step.

    This is what makes counterfactual attribution affordable: replaying a trajectory up
    to step k costs nothing, and only the steps after the intervention hit the API.
    """

    def __init__(
        self,
        transcript: list[str],
        fallback: LLMBackend | None = None,
        diverge_at: int | None = None,
    ):
        self.transcript = transcript
        self.fallback = fallback
        self.diverge_at = diverge_at
        self._calls = 0
        self.spec = ModelSpec(
            backend="replay",
            model=fallback.spec.model if fallback else "replay",
            open_weights=True,
        )

    def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        index = self._calls
        self._calls += 1
        replaying = index < len(self.transcript) and (
            self.diverge_at is None or index < self.diverge_at
        )
        if replaying:
            return LLMResponse(text=self.transcript[index], cached=True)
        if self.fallback is None:
            raise RuntimeError("ReplayBackend exhausted its transcript with no live fallback.")
        return self.fallback.complete(messages, **kwargs)


def build_backend(
    preset_or_provider: str, model: str | None = None, **kwargs: Any
) -> OpenAICompatBackend:
    """`build_backend("mid-open")` or `build_backend("groq", "qwen/qwen3-32b")`."""
    if preset_or_provider in PRESETS:
        provider, preset_model = PRESETS[preset_or_provider]
        return OpenAICompatBackend(provider, model or preset_model, **kwargs)
    if model is None:
        raise ValueError(f"{preset_or_provider!r} is not a preset; pass an explicit model id.")
    return OpenAICompatBackend(preset_or_provider, model, **kwargs)


def available_backends() -> list[str]:
    """Presets whose API key is actually present in the environment."""
    out = []
    for preset, (provider, _) in PRESETS.items():
        p = PROVIDERS[provider]
        if provider == "local" or os.getenv(p.api_key_env):
            out.append(preset)
    return out
