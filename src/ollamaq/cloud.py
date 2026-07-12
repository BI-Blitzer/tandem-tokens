"""Optional cloud LLM backends for hybrid fallback / escalation.

Local Ollama is always the default path. Cloud providers are opt-in via
config/models.yaml `cloud:` + an API key env var. Today: xAI Grok
(OpenAI-compatible chat completions at api.x.ai).

Used when a job's escalate_to resolves to `cloud:grok` or `cloud:grok/<model>`,
or when fallbacks.escalate_to points there — residual items the local model
couldn't fix go to Grok instead of burning premium Claude tokens on bulk work.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from .client import GenerateResult
from .errors import ConfigError, OllamaConnectionError, OllamaHTTPError, OllamaqError

# Accepted env vars for the xAI key (either name works).
GROK_KEY_ENVS = ("XAI_API_KEY", "GROK_API_KEY")


class GrokSettings(BaseModel):
    """Runtime settings for one Grok/xAI call (from config + env)."""

    base_url: str = "https://api.x.ai/v1"
    api_key: str
    default_model: str = "grok-3-mini"
    timeout_s: float = 120.0
    enabled: bool = True


def resolve_grok_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for env in GROK_KEY_ENVS:
        v = os.environ.get(env)
        if v:
            return v
    return None


def parse_cloud_target(target: str) -> tuple[str, str | None]:
    """Parse `cloud:grok` or `cloud:grok/grok-3-mini` → (provider, model|None).

    Raises ConfigError if the string is not a cloud target.
    """
    if not target.startswith("cloud:"):
        raise ConfigError(f"not a cloud target: {target!r}")
    rest = target[len("cloud:"):]
    if not rest:
        raise ConfigError("cloud target missing provider (expected cloud:grok[/model])")
    if "/" in rest:
        provider, model = rest.split("/", 1)
        return provider.strip(), (model.strip() or None)
    return rest.strip(), None


def is_cloud_target(name: str | None) -> bool:
    return bool(name) and name.startswith("cloud:")


class GrokClient:
    """Async OpenAI-compatible chat client for xAI Grok."""

    def __init__(self, settings: GrokSettings) -> None:
        if not settings.api_key:
            raise ConfigError(
                "Grok cloud fallback enabled but no API key set "
                f"(set {' or '.join(GROK_KEY_ENVS)}, or cloud.grok.api_key)"
            )
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"),
            timeout=settings.timeout_s,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "GrokClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def generate(
        self,
        model: str | None,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict | None = None,
        options: dict[str, Any] | None = None,
    ) -> GenerateResult:
        """Chat-completions generate; returns a GenerateResult-shaped object so
        the worker/enforce path can treat cloud and local the same."""
        options = options or {}
        use_model = model or self.settings.default_model
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": use_model,
            "messages": messages,
        }
        if "temperature" in options:
            body["temperature"] = options["temperature"]
        # Best-effort JSON mode when a schema is requested (full JSON Schema is
        # not uniformly supported across cloud providers).
        if schema is not None:
            body["response_format"] = {"type": "json_object"}
        # Map Ollama num_predict → max_tokens when present.
        if "num_predict" in options:
            body["max_tokens"] = int(options["num_predict"])

        try:
            resp = await self._client.post("/chat/completions", json=body)
        except httpx.RequestError as exc:
            raise OllamaConnectionError(f"Grok API connection failed: {exc}") from exc

        if resp.status_code >= 400:
            raise OllamaHTTPError(
                f"Grok API HTTP {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise OllamaqError(f"Grok API unexpected response shape: {data!r}") from exc

        usage = data.get("usage") or {}
        return GenerateResult(
            model=use_model,
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )
