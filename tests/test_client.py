"""Contract tests for OllamaClient, using respx to mock the HTTP layer.

No live server needed. These pin the request shape ollamaq sends and the
metric parsing it expects back.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from ollamaq.client import GenerateResult, OllamaClient
from ollamaq.config import OllamaSettings
from ollamaq.errors import OllamaConnectionError, OllamaHTTPError

BASE = "http://localhost:11434"


def _settings() -> OllamaSettings:
    return OllamaSettings(base_url=BASE, timeout_s=5)


def test_tokens_per_second_math() -> None:
    r = GenerateResult(
        model="m", text="x", completion_tokens=100, eval_duration_ns=2_000_000_000
    )
    assert r.tokens_per_second == 50.0
    assert r.load_seconds == 0.0


def test_tokens_per_second_zero_guard() -> None:
    r = GenerateResult(model="m", text="x", completion_tokens=5, eval_duration_ns=0)
    assert r.tokens_per_second == 0.0


@respx.mock
async def test_ping_true() -> None:
    respx.get(f"{BASE}/api/version").mock(return_value=httpx.Response(200, json={"version": "0.1"}))
    async with OllamaClient(_settings()) as c:
        assert await c.ping() is True


@respx.mock
async def test_ping_false_on_error() -> None:
    respx.get(f"{BASE}/api/version").mock(side_effect=httpx.ConnectError("down"))
    async with OllamaClient(_settings()) as c:
        assert await c.ping() is False


@respx.mock
async def test_list_models() -> None:
    respx.get(f"{BASE}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "gemma2:9b"}, {"name": "qwen3:8b"}]})
    )
    async with OllamaClient(_settings()) as c:
        assert await c.list_models() == ["gemma2:9b", "qwen3:8b"]


@respx.mock
async def test_generate_parses_metrics_and_payload() -> None:
    route = respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(
            200,
            json={
                "response": "hello",
                "prompt_eval_count": 12,
                "eval_count": 34,
                "total_duration": 5_000_000_000,
                "load_duration": 1_000_000_000,
                "eval_duration": 2_000_000_000,
            },
        )
    )
    schema = {"type": "object"}
    async with OllamaClient(_settings()) as c:
        res = await c.generate(
            "gemma2:9b",
            "hi",
            system="sys",
            schema=schema,
            options={"num_ctx": 8192, "temperature": 0.0},
        )
    assert res.text == "hello"
    assert res.prompt_tokens == 12
    assert res.completion_tokens == 34
    assert res.tokens_per_second == 17.0

    sent = route.calls.last.request
    import json

    body = json.loads(sent.content)
    assert body["model"] == "gemma2:9b"
    assert body["stream"] is False
    assert body["system"] == "sys"
    assert body["format"] == schema
    assert body["options"]["num_ctx"] == 8192
    assert body["keep_alive"] == "30m"


@respx.mock
async def test_generate_includes_images() -> None:
    route = respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "a table", "eval_count": 3,
                                               "eval_duration": 1_000_000_000}))
    async with OllamaClient(_settings()) as c:
        await c.generate("llama3.2-vision:11b", "extract", images=["BASE64IMG"])
    import json
    body = json.loads(route.calls.last.request.content)
    assert body["images"] == ["BASE64IMG"]


@respx.mock
async def test_generate_omits_images_when_none() -> None:
    route = respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "x", "eval_count": 1,
                                               "eval_duration": 1}))
    async with OllamaClient(_settings()) as c:
        await c.generate("m", "hi")
    import json
    assert "images" not in json.loads(route.calls.last.request.content)


@respx.mock
async def test_generate_http_error() -> None:
    respx.post(f"{BASE}/api/generate").mock(return_value=httpx.Response(404, text="model not found"))
    async with OllamaClient(_settings()) as c:
        with pytest.raises(OllamaHTTPError) as ei:
            await c.generate("ghost", "hi")
    assert ei.value.status_code == 404


@respx.mock
async def test_generate_connection_error() -> None:
    respx.post(f"{BASE}/api/generate").mock(side_effect=httpx.ConnectError("refused"))
    async with OllamaClient(_settings()) as c:
        with pytest.raises(OllamaConnectionError):
            await c.generate("m", "hi")
