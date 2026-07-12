"""Cloud target parsing + Grok client generate (HTTP mocked)."""

from __future__ import annotations

import httpx
import pytest
import respx

from ollamaq.cloud import (
    GrokClient,
    GrokSettings,
    is_cloud_target,
    parse_cloud_target,
)
from ollamaq.config import Config
from ollamaq.errors import ConfigError


def test_parse_cloud_target() -> None:
    assert parse_cloud_target("cloud:grok") == ("grok", None)
    assert parse_cloud_target("cloud:grok/grok-3-mini") == ("grok", "grok-3-mini")
    with pytest.raises(ConfigError):
        parse_cloud_target("gemma2:9b")
    with pytest.raises(ConfigError):
        parse_cloud_target("cloud:")


def test_is_cloud_target() -> None:
    assert is_cloud_target("cloud:grok")
    assert not is_cloud_target("qwen2.5-14b-16k")
    assert not is_cloud_target(None)


def test_resolve_escalate_local() -> None:
    cfg = Config.model_validate({
        "defaults": {"model": "gemma2:9b"},
        "models": {
            "gemma2:9b": {"num_ctx": 8192},
            "qwen2.5-14b-16k": {"num_ctx": 16384},
        },
        "routing": {"clean": "gemma2:9b"},
        "fallbacks": {"escalate_to": "qwen2.5-14b-16k"},
    })
    assert cfg.resolve_escalate_target(None) == "qwen2.5-14b-16k"
    assert cfg.resolve_escalate_target("auto") == "qwen2.5-14b-16k"
    assert cfg.resolve_escalate_target("gemma2:9b") == "gemma2:9b"


def test_resolve_escalate_cloud() -> None:
    cfg = Config.model_validate({
        "defaults": {"model": "gemma2:9b"},
        "models": {"gemma2:9b": {"num_ctx": 8192}},
        "routing": {"clean": "gemma2:9b"},
        "fallbacks": {"escalate_to": "cloud:grok"},
        "cloud": {"grok": {"enabled": True, "default_model": "grok-3-mini"}},
    })
    assert cfg.resolve_escalate_target("auto") == "cloud:grok"
    assert cfg.resolve_escalate_target("cloud:grok/grok-3") == "cloud:grok/grok-3"


def test_resolve_escalate_by_task_tag() -> None:
    cfg = Config.model_validate({
        "defaults": {"model": "gemma2:9b"},
        "models": {
            "gemma2:9b": {"num_ctx": 8192},
            "qwen2.5-14b-16k": {"num_ctx": 16384},
        },
        "routing": {"clean": "gemma2:9b", "code": "gemma2:9b"},
        "fallbacks": {
            "escalate_to": "qwen2.5-14b-16k",
            "by_task_tag": {"code": "cloud:grok"},
        },
    })
    assert cfg.resolve_escalate_target("auto", "code") == "cloud:grok"
    assert cfg.resolve_escalate_target("auto", "clean") == "qwen2.5-14b-16k"


@respx.mock
async def test_grok_client_generate() -> None:
    route = respx.post("https://api.x.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })
    )
    settings = GrokSettings(api_key="test-key", default_model="grok-3-mini")
    async with GrokClient(settings) as client:
        res = await client.generate(
            None, "hello", system="sys", schema={"type": "object"},
            options={"temperature": 0.1},
        )
    assert res.text == '{"ok": true}'
    assert res.prompt_tokens == 10 and res.completion_tokens == 5
    body = route.calls.last.request.content
    import json
    data = json.loads(body)
    assert data["model"] == "grok-3-mini"
    assert data["response_format"] == {"type": "json_object"}
    assert data["messages"][0]["role"] == "system"
