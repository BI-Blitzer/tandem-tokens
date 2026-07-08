"""Contract tests for embed mode: client.embed, the engine, and dispatch."""

from __future__ import annotations

import httpx
import respx

from ollamaq.client import OllamaClient
from ollamaq.config import Config, OllamaSettings
from ollamaq.embed import run_embed
from ollamaq.jobspec import JobSpec

BASE = "http://localhost:11434"


def _cfg() -> Config:
    return Config.model_validate({
        "ollama": {"base_url": BASE},
        "defaults": {"model": "nomic-embed-text:latest"},
        "models": {"nomic-embed-text:latest": {"max_parallel": 4, "num_ctx": 8192}},
        "routing": {"embed": "nomic-embed-text:latest"},
    })


def _spec() -> JobSpec:
    return JobSpec.model_validate({
        "name": "e", "mode": "embed", "model": "nomic-embed-text:latest",
        "input": {"kind": "inline", "values": ["a", "b", "c"],
                  "chunk": {"strategy": "rows", "size": 1}},
    })


def _embed_resp(vec, tokens=3) -> httpx.Response:
    return httpx.Response(200, json={"embeddings": [vec], "prompt_eval_count": tokens})


# ---- client.embed ------------------------------------------------------------

@respx.mock
async def test_client_embed_parses_vector() -> None:
    route = respx.post(f"{BASE}/api/embed").mock(return_value=_embed_resp([0.1, 0.2, 0.3]))
    async with OllamaClient(OllamaSettings(base_url=BASE)) as c:
        res = await c.embed("nomic-embed-text:latest", "hello")
    assert res.embedding == [0.1, 0.2, 0.3] and res.dim == 3
    assert res.prompt_tokens == 3
    import json
    assert json.loads(route.calls.last.request.content)["input"] == "hello"


# ---- embed engine ------------------------------------------------------------

@respx.mock
async def test_run_embed_all_items() -> None:
    respx.post(f"{BASE}/api/embed").mock(return_value=_embed_resp([0.5, 0.5]))
    items = [(i, f"s{i}", t, ()) for i, t in enumerate(["a", "b", "c"])]
    run = await run_embed(_cfg(), _spec(), items)
    assert run.ok_count == 3
    assert run.results[0].value == {"dim": 2, "embedding": [0.5, 0.5]}
    assert run.total_prompt_tokens == 9          # 3 tokens x 3 items
    assert run.total_completion_tokens == 0      # embeddings have no output tokens


@respx.mock
async def test_run_embed_empty_vector_fails() -> None:
    respx.post(f"{BASE}/api/embed").mock(return_value=httpx.Response(
        200, json={"embeddings": []}))
    run = await run_embed(_cfg(), _spec(), [(0, "s", "a", ())])
    assert run.fail_count == 1
    assert "empty vector" in run.results[0].error


@respx.mock
async def test_run_embed_transient_retry() -> None:
    respx.post(f"{BASE}/api/embed").mock(side_effect=[
        httpx.ConnectError("down"), _embed_resp([1.0])])
    run = await run_embed(_cfg(), _spec(), [(0, "s", "a", ())])
    assert run.ok_count == 1 and run.results[0].value["embedding"] == [1.0]


# ---- jobspec validation ------------------------------------------------------

def test_embed_needs_no_prompt() -> None:
    spec = JobSpec.model_validate({
        "name": "e", "mode": "embed", "model": "nomic-embed-text:latest",
        "input": {"kind": "inline", "values": [1], "chunk": {"strategy": "rows", "size": 1}},
    })
    assert spec.prompt is None and spec.mode.value == "embed"


def test_embed_auto_still_needs_task_tag() -> None:
    import pytest
    with pytest.raises(Exception, match="task_tag"):
        JobSpec.model_validate({
            "name": "e", "mode": "embed",  # model defaults to auto
            "input": {"kind": "inline", "values": [1], "chunk": {"strategy": "rows", "size": 1}},
        })
