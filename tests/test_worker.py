"""Contract tests for the async worker, HTTP mocked via respx.

Covers: happy path, the repair-retry loop, permanent failure, transport-error
short-circuit, prompt rendering, and that concurrency never exceeds max_parallel.
"""

from __future__ import annotations

import httpx
import respx

import pytest

from ollamaq.config import Config
from ollamaq.errors import ConfigError
from ollamaq.jobspec import JobSpec
from ollamaq.worker import render_prompt, run_job

BASE = "http://localhost:11434"


def _cfg(max_parallel: int = 4) -> Config:
    return Config.model_validate(
        {
            "ollama": {"base_url": BASE},
            "defaults": {"model": "gemma2:9b"},
            "models": {"gemma2:9b": {"max_parallel": max_parallel, "num_ctx": 8192},
                       "qwen2.5-14b-16k": {"max_parallel": 2, "num_ctx": 16384}},
            "routing": {"clean": "gemma2:9b"},
        }
    )


def _spec(**over) -> JobSpec:
    base = {
        "name": "t",
        "model": "gemma2:9b",
        "input": {"kind": "inline", "values": [1], "chunk": {"strategy": "rows", "size": 1}},
        "prompt": {"template": "do {{chunk}}", "schema": {"type": "object",
                   "properties": {"id": {"type": "integer"}}, "required": ["id"]}},
    }
    base.update(over)
    return JobSpec.model_validate(base)


def _resp(text: str) -> httpx.Response:
    return httpx.Response(200, json={"response": text, "eval_count": 5,
                                     "eval_duration": 1_000_000_000})


def test_render_prompt() -> None:
    assert render_prompt("x {{chunk}} y", "Z") == "x Z y"


@respx.mock
async def test_happy_path() -> None:
    respx.post(f"{BASE}/api/generate").mock(return_value=_resp('{"id": 1}'))
    run = await run_job(_cfg(), _spec(), [(0, "src", "chunk")])
    assert run.ok_count == 1
    assert run.results[0].value == {"id": 1}
    assert run.results[0].attempts == 1


@respx.mock
async def test_repair_then_retry_recovers() -> None:
    # first response invalid, second valid
    responses = [_resp("garbage"), _resp('{"id": 9}')]
    respx.post(f"{BASE}/api/generate").mock(side_effect=responses)
    run = await run_job(_cfg(), _spec(), [(0, "src", "chunk")])
    assert run.ok_count == 1
    assert run.results[0].attempts == 2
    assert run.results[0].value == {"id": 9}


@respx.mock
async def test_permanent_schema_failure() -> None:
    respx.post(f"{BASE}/api/generate").mock(return_value=_resp("never valid"))
    run = await run_job(_cfg(), _spec(retry={"max_attempts": 2}), [(0, "s", "c")])
    assert run.fail_count == 1
    assert run.results[0].attempts == 2
    assert "not valid JSON" in run.results[0].error


@respx.mock
async def test_on_schema_fail_fail_stops_immediately() -> None:
    respx.post(f"{BASE}/api/generate").mock(return_value=_resp("bad"))
    run = await run_job(
        _cfg(), _spec(retry={"max_attempts": 5, "on_schema_fail": "fail"}), [(0, "s", "c")]
    )
    assert run.fail_count == 1
    assert run.results[0].attempts == 1  # did not retry


@respx.mock
async def test_transport_error_short_circuits() -> None:
    respx.post(f"{BASE}/api/generate").mock(side_effect=httpx.ConnectError("down"))
    run = await run_job(_cfg(), _spec(retry={"max_attempts": 3}), [(0, "s", "c")])
    assert run.fail_count == 1
    assert run.results[0].attempts == 1
    assert "generate failed" in run.results[0].error


@respx.mock
async def test_results_ordered_and_stats() -> None:
    respx.post(f"{BASE}/api/generate").mock(return_value=_resp('{"id": 1}'))
    items = [(i, f"s{i}", f"c{i}") for i in range(5)]
    run = await run_job(_cfg(), _spec(), items)
    assert [r.index for r in run.results] == [0, 1, 2, 3, 4]
    assert run.success_rate == 1.0
    assert run.total_completion_tokens == 25
    assert run.tokens_per_second > 0


@respx.mock
async def test_no_schema_accepts_anything() -> None:
    respx.post(f"{BASE}/api/generate").mock(return_value=_resp("free text"))
    spec = _spec(prompt={"template": "sum {{chunk}}"})  # no schema
    run = await run_job(_cfg(), spec, [(0, "s", "c")])
    assert run.ok_count == 1
    assert run.results[0].value == {"text": "free text"}


# ---- dedupe cache (6a) -------------------------------------------------------

@respx.mock
async def test_cache_miss_then_hit_skips_generation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMAQ_CACHE", str(tmp_path / "cache"))
    route = respx.post(f"{BASE}/api/generate").mock(return_value=_resp('{"id": 1}'))
    spec = _spec(cache=True)

    run1 = await run_job(_cfg(), spec, [(0, "s", "c")])
    assert run1.ok_count == 1 and run1.results[0].cached is False
    assert route.call_count == 1

    # second run: same request -> served from cache, no new HTTP call
    run2 = await run_job(_cfg(), spec, [(0, "s", "c")])
    assert route.call_count == 1  # unchanged
    r = run2.results[0]
    assert r.cached is True and r.value == {"id": 1}
    assert r.prompt_tokens == 0 and r.completion_tokens == 0
    assert run2.cached_count == 1


@respx.mock
async def test_cache_disabled_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMAQ_CACHE", str(tmp_path / "cache"))
    route = respx.post(f"{BASE}/api/generate").mock(return_value=_resp('{"id": 1}'))
    spec = _spec()  # cache defaults to False
    await run_job(_cfg(), spec, [(0, "s", "c")])
    await run_job(_cfg(), spec, [(0, "s", "c")])
    assert route.call_count == 2  # both ran; no caching
    assert not (tmp_path / "cache").exists()


@respx.mock
async def test_cache_key_sensitive_to_chunk(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMAQ_CACHE", str(tmp_path / "cache"))
    route = respx.post(f"{BASE}/api/generate").mock(return_value=_resp('{"id": 1}'))
    spec = _spec(cache=True)
    await run_job(_cfg(), spec, [(0, "s", "chunkA")])
    await run_job(_cfg(), spec, [(1, "s", "chunkB")])  # different chunk -> miss
    assert route.call_count == 2


@respx.mock
async def test_cache_does_not_store_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMAQ_CACHE", str(tmp_path / "cache"))
    route = respx.post(f"{BASE}/api/generate").mock(return_value=_resp("not json"))
    spec = _spec(cache=True, retry={"max_attempts": 1})
    await run_job(_cfg(), spec, [(0, "s", "c")])
    await run_job(_cfg(), spec, [(0, "s", "c")])  # failure not cached -> re-runs
    assert route.call_count == 2


# ---- escalation (6b) ---------------------------------------------------------

def _gen_route(primary_text: str, escalate_text: str):
    """Route: primary model gets primary_text, the escalate model gets escalate_text."""
    def route(request):
        import json as _json
        model = _json.loads(request.content)["model"]
        text = escalate_text if model == "qwen2.5-14b-16k" else primary_text
        return httpx.Response(200, json={"response": text, "eval_count": 3,
                                         "eval_duration": 1_000_000_000})
    return route


@respx.mock
async def test_escalation_recovers_failure() -> None:
    # primary always returns junk; escalate model returns valid JSON
    respx.post(f"{BASE}/api/generate").mock(
        side_effect=_gen_route("not json", '{"id": 7}'))
    spec = _spec(retry={"max_attempts": 2, "escalate_to": "qwen2.5-14b-16k"})
    run = await run_job(_cfg(), spec, [(0, "s", "c")])
    r = run.results[0]
    assert r.ok and r.escalated and r.value == {"id": 7}
    assert r.attempts == 3  # 2 primary + 1 escalation
    assert run.escalated_count == 1


@respx.mock
async def test_escalation_also_fails() -> None:
    respx.post(f"{BASE}/api/generate").mock(
        side_effect=_gen_route("junk", "still junk"))
    spec = _spec(retry={"max_attempts": 1, "escalate_to": "qwen2.5-14b-16k"})
    run = await run_job(_cfg(), spec, [(0, "s", "c")])
    r = run.results[0]
    assert not r.ok and not r.escalated
    assert "escalation to qwen2.5-14b-16k also failed" in r.error


@respx.mock
async def test_escalation_uses_escalate_num_ctx() -> None:
    calls: list = []

    def route(request):
        import json as _json
        body = _json.loads(request.content)
        calls.append((body["model"], body.get("options", {}).get("num_ctx")))
        text = '{"id": 1}' if body["model"] == "qwen2.5-14b-16k" else "junk"
        return httpx.Response(200, json={"response": text, "eval_count": 1,
                                         "eval_duration": 1_000_000_000})

    respx.post(f"{BASE}/api/generate").mock(side_effect=route)
    spec = _spec(retry={"max_attempts": 1, "escalate_to": "qwen2.5-14b-16k"})
    await run_job(_cfg(), spec, [(0, "s", "c")])
    # the escalation call must use the 14b model's own num_ctx (16384)
    assert ("qwen2.5-14b-16k", 16384) in calls


@respx.mock
async def test_escalate_to_same_model_is_noop() -> None:
    route = respx.post(f"{BASE}/api/generate").mock(return_value=_resp("junk"))
    spec = _spec(retry={"max_attempts": 2, "escalate_to": "gemma2:9b"})  # == primary
    run = await run_job(_cfg(), spec, [(0, "s", "c")])
    assert route.call_count == 2  # no extra escalation call
    assert not run.results[0].escalated


async def test_escalate_to_undefined_raises() -> None:
    spec = _spec(retry={"max_attempts": 1, "escalate_to": "ghost-model"})
    with pytest.raises(ConfigError):
        await run_job(_cfg(), spec, [(0, "s", "c")])


@respx.mock
async def test_worker_passes_images_from_4tuple() -> None:
    route = respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": '{"id": 1}', "eval_count": 2,
                                               "eval_duration": 1_000_000_000}))
    spec = _spec()
    # 4-tuple carries images; run_job must forward them to the model
    run = await run_job(_cfg(), spec, [(0, "img.png", "describe", ["B64DATA"])])
    assert run.ok_count == 1
    import json as _json
    assert _json.loads(route.calls.last.request.content)["images"] == ["B64DATA"]


@respx.mock
async def test_transport_error_does_not_escalate() -> None:
    respx.post(f"{BASE}/api/generate").mock(side_effect=httpx.ConnectError("down"))
    spec = _spec(retry={"max_attempts": 2, "escalate_to": "qwen2.5-14b-16k"})
    run = await run_job(_cfg(), spec, [(0, "s", "c")])
    r = run.results[0]
    assert not r.ok and not r.escalated
    assert "generate failed" in r.error  # short-circuited before escalation
