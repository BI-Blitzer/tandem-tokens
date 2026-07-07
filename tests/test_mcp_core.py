"""Contract tests for the MCP tool logic (mcp_core). HTTP mocked; no MCP client."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx

from ollamaq import mcp_core
from ollamaq.config import Config
from ollamaq.queue import JobQueue, State

BASE = "http://localhost:11434"

SPEC = {"name": "j", "model": "gemma2:9b",
        "input": {"kind": "inline", "values": [1], "chunk": {"strategy": "rows", "size": 1}},
        "prompt": {"template": "do {{chunk}}"}}


def _cfg() -> Config:
    return Config.model_validate({
        "ollama": {"base_url": BASE},
        "defaults": {"model": "gemma2:9b", "temperature": 0.1},
        "models": {"gemma2:9b": {"max_parallel": 4, "num_ctx": 8192}},
        "routing": {"clean": "gemma2:9b"},
    })


# ---- ollama_quick ------------------------------------------------------------

@respx.mock
async def test_quick_returns_text_and_caps_tokens() -> None:
    route = respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "hello there",
                                               "eval_count": 2, "eval_duration": 1}))
    out = await mcp_core.quick(_cfg(), "hi")
    assert out == "hello there"
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "gemma2:9b"
    assert body["options"]["num_predict"] == mcp_core.QUICK_MAX_TOKENS


@respx.mock
async def test_quick_truncates_long_output() -> None:
    big = "x" * (mcp_core.QUICK_MAX_CHARS + 500)
    respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": big, "eval_count": 1,
                                               "eval_duration": 1}))
    out = await mcp_core.quick(_cfg(), "hi")
    assert len(out) < len(big)
    assert "truncated" in out


@respx.mock
async def test_quick_routes_by_task_tag() -> None:
    route = respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "ok", "eval_count": 1,
                                               "eval_duration": 1}))
    await mcp_core.quick(_cfg(), "hi", model="auto", task_tag="clean")
    assert json.loads(route.calls.last.request.content)["model"] == "gemma2:9b"


# ---- queue_submit / status / digest ------------------------------------------

def test_submit_and_status(tmp_path: Path) -> None:
    q = JobQueue(tmp_path / "queue")
    jid = mcp_core.submit(q, _cfg(), SPEC, job_id="j1")
    assert jid == "j1"
    assert q.find("j1")[0] == State.pending
    counts = mcp_core.status(q)
    assert counts["pending"] == 1 and counts["done"] == 0


def test_submit_invalid_spec_raises(tmp_path: Path) -> None:
    import pytest
    from ollamaq.errors import OllamaqError

    q = JobQueue(tmp_path / "queue")
    with pytest.raises(OllamaqError, match="invalid job spec"):
        mcp_core.submit(q, _cfg(), {"name": "bad"})  # missing input/prompt


def test_status_single_job(tmp_path: Path) -> None:
    q = JobQueue(tmp_path / "queue")
    mcp_core.submit(q, _cfg(), SPEC, job_id="j1")
    detail = mcp_core.status(q, "j1")
    assert detail["found"] is True and detail["state"] == "pending"
    assert mcp_core.status(q, "nope")["found"] is False


def test_digest_present_and_absent(tmp_path: Path) -> None:
    d = tmp_path / "results" / "j1"
    d.mkdir(parents=True)
    (d / "digest.md").write_text("# hi digest\n", encoding="utf-8")
    assert "hi digest" in mcp_core.digest(tmp_path / "results", "j1")
    assert "no digest" in mcp_core.digest(tmp_path / "results", "missing")
