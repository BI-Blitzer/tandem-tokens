"""Async HTTP client for the Ollama REST API (localhost).

This is the ONLY module that talks HTTP to Ollama. Everything above it
(bench, worker, MCP server) goes through OllamaClient.

Endpoints used:
  GET  /api/version   -> ping()
  GET  /api/tags      -> list_models()
  POST /api/generate  -> generate()   (always stream=False)
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel

from .config import OllamaSettings
from .errors import OllamaConnectionError, OllamaHTTPError


class GenerateResult(BaseModel):
    """One completed generation, with the raw Ollama timing/token metrics.

    Duration fields are nanoseconds, exactly as Ollama reports them.
    """

    model: str
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    eval_duration_ns: int = 0

    @property
    def tokens_per_second(self) -> float:
        """Decode speed. 0.0 when Ollama reported no eval duration."""
        if self.eval_duration_ns <= 0:
            return 0.0
        return self.completion_tokens / (self.eval_duration_ns / 1e9)

    @property
    def load_seconds(self) -> float:
        return self.load_duration_ns / 1e9


class EmbedResult(BaseModel):
    """One text's embedding vector plus the input token count Ollama reported."""

    model: str
    embedding: list[float]
    prompt_tokens: int = 0

    @property
    def dim(self) -> int:
        return len(self.embedding)


class OllamaClient:
    """Thin async wrapper. Use as ``async with OllamaClient(settings) as c:``."""

    def __init__(self, settings: OllamaSettings | None = None) -> None:
        self.settings = settings or OllamaSettings()
        headers = ({"Authorization": f"Bearer {self.settings.api_key}"}
                   if self.settings.api_key else None)
        self._client = httpx.AsyncClient(
            base_url=self.settings.base_url, timeout=self.settings.timeout_s,
            headers=headers,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def ping(self) -> bool:
        """True iff the server answers /api/version with 200. Never raises."""
        try:
            resp = await self._client.get("/api/version")
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    async def list_models(self) -> list[str]:
        """Installed model names as Ollama reports them (usually with a :tag suffix)."""
        try:
            resp = await self._client.get("/api/tags")
        except httpx.HTTPError as exc:
            raise OllamaConnectionError(
                f"cannot reach Ollama at {self.settings.base_url}: {exc}"
            ) from exc
        if resp.status_code != 200:
            raise OllamaHTTPError(resp.status_code, resp.text[:500])
        return [m["name"] for m in resp.json().get("models", [])]

    async def embed(self, model: str, text: str,
                    keep_alive: str | None = None) -> EmbedResult:
        """Embed one text via /api/embed. Raises OllamaConnectionError /
        OllamaHTTPError like generate()."""
        payload: dict[str, Any] = {"model": model, "input": text,
                                   "keep_alive": keep_alive or self.settings.keep_alive}
        try:
            resp = await self._client.post("/api/embed", json=payload)
        except httpx.HTTPError as exc:
            raise OllamaConnectionError(
                f"embed failed against {self.settings.base_url}: {exc}") from exc
        if resp.status_code != 200:
            raise OllamaHTTPError(resp.status_code, resp.text[:500])
        data = resp.json()
        embs = data.get("embeddings") or []
        return EmbedResult(model=model, embedding=embs[0] if embs else [],
                           prompt_tokens=data.get("prompt_eval_count") or 0)

    async def model_sizes(self) -> dict[str, float]:
        """Installed model name -> on-disk size in MB (from /api/tags)."""
        try:
            resp = await self._client.get("/api/tags")
        except httpx.HTTPError as exc:
            raise OllamaConnectionError(
                f"cannot reach Ollama at {self.settings.base_url}: {exc}"
            ) from exc
        if resp.status_code != 200:
            raise OllamaHTTPError(resp.status_code, resp.text[:500])
        return {m["name"]: (m.get("size", 0) or 0) / 1e6
                for m in resp.json().get("models", [])}

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        images: list[str] | None = None,
        keep_alive: str | None = None,
    ) -> GenerateResult:
        """Single non-streaming generation.

        - ``schema`` (a JSON schema dict) is passed as Ollama's ``format`` field,
          which constrains decoding to schema-valid JSON.
        - ``options`` maps straight to Ollama options (num_ctx, temperature, ...).
        - ``images`` is a list of base64-encoded image strings (vision models).
        - ``keep_alive`` defaults to settings.keep_alive.

        Raises OllamaConnectionError (unreachable/timeout) or OllamaHTTPError
        (non-200, e.g. model not found).
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive or self.settings.keep_alive,
        }
        if system:
            payload["system"] = system
        if schema is not None:
            payload["format"] = schema
        if options:
            payload["options"] = options
        if images:
            payload["images"] = images
        try:
            resp = await self._client.post("/api/generate", json=payload)
        except httpx.HTTPError as exc:
            raise OllamaConnectionError(
                f"generate failed against {self.settings.base_url}: {exc}"
            ) from exc
        if resp.status_code != 200:
            raise OllamaHTTPError(resp.status_code, resp.text[:500])
        data = resp.json()
        return GenerateResult(
            model=model,
            text=data.get("response", ""),
            prompt_tokens=data.get("prompt_eval_count") or 0,
            completion_tokens=data.get("eval_count") or 0,
            total_duration_ns=data.get("total_duration") or 0,
            load_duration_ns=data.get("load_duration") or 0,
            eval_duration_ns=data.get("eval_duration") or 0,
        )
