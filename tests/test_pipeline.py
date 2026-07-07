"""Contract tests for pipeline mode: filter matching + two-stage execution."""

from __future__ import annotations

import httpx
import respx

from ollamaq.config import Config
from ollamaq.jobspec import JobSpec, StageFilter
from ollamaq.pipeline import matches, run_pipeline

BASE = "http://localhost:11434"


# ---- filter matching (pure) --------------------------------------------------

def _f(**kw) -> StageFilter:
    return StageFilter.model_validate(kw)


def test_filter_eq() -> None:
    assert matches(_f(field="dirty", op="==", value=True), {"dirty": True})
    assert not matches(_f(field="dirty", op="==", value=True), {"dirty": False})


def test_filter_truthy_falsy() -> None:
    assert matches(_f(field="x", op="truthy"), {"x": 1})
    assert not matches(_f(field="x", op="truthy"), {"x": 0})
    assert matches(_f(field="x", op="falsy"), {"x": 0})
    assert matches(_f(field="x", op="falsy"), {})  # absent -> falsy


def test_filter_numeric() -> None:
    assert matches(_f(field="score", op=">=", value=0.5), {"score": 0.7})
    assert not matches(_f(field="score", op=">=", value=0.5), {"score": 0.3})


def test_filter_dotted_path() -> None:
    assert matches(_f(field="meta.ok", op="==", value=True), {"meta": {"ok": True}})
    assert not matches(_f(field="meta.ok", op="truthy"), {"meta": {}})


def test_filter_noncomparable_is_false() -> None:
    # None vs number must not raise, just not match
    assert not matches(_f(field="x", op=">", value=1), {"x": None})


# ---- two-stage execution (mocked HTTP) ---------------------------------------

def _cfg() -> Config:
    return Config.model_validate({
        "ollama": {"base_url": BASE},
        "defaults": {"model": "gemma2:9b"},
        "models": {"gemma2:9b": {"max_parallel": 4, "num_ctx": 8192},
                   "qwen2.5-14b-16k": {"max_parallel": 2, "num_ctx": 16384}},
        "routing": {"classify": "gemma2:9b", "clean": "qwen2.5-14b-16k"},
    })


def _pipeline_spec() -> JobSpec:
    return JobSpec.model_validate({
        "name": "classify-then-clean",
        "mode": "pipeline",
        "input": {"kind": "inline", "values": ["a", "b"],
                  "chunk": {"strategy": "rows", "size": 1}},
        "stages": [
            {"name": "classify", "model": "gemma2:9b",
             "prompt": {"template": "flag {{chunk}}",
                        "schema": {"type": "object", "required": ["dirty"]}},
             "filter": {"field": "dirty", "op": "==", "value": True}},
            {"name": "clean", "model": "qwen2.5-14b-16k",
             "prompt": {"template": "clean {{chunk}}",
                        "schema": {"type": "object", "required": ["cleaned"]}}},
        ],
    })


@respx.mock
async def test_pipeline_routes_by_filter() -> None:
    # Stage 1 (classify): item 0 -> dirty True, item 1 -> dirty False.
    # Stage 2 (clean): only the dirty one arrives.
    import json as _json

    def route(request):
        prompt = _json.loads(request.content)["prompt"]  # decoded prompt text
        if prompt.startswith("flag"):
            dirty = '"a"' in prompt  # item 'a' is dirty, 'b' is clean
            return httpx.Response(200, json={
                "response": f'{{"dirty": {"true" if dirty else "false"}}}',
                "eval_count": 2, "eval_duration": 1_000_000_000})
        # clean stage
        return httpx.Response(200, json={
            "response": '{"cleaned": "yes"}', "eval_count": 3,
            "eval_duration": 1_000_000_000})

    respx.post(f"{BASE}/api/generate").mock(side_effect=route)
    run = await run_pipeline(_cfg(), _pipeline_spec())

    assert len(run.stages) == 2
    s1, s2 = run.stages
    assert s1["stage"] == "classify" and s1["in"] == 2 and s1["continued"] == 1
    assert s2["stage"] == "clean" and s2["in"] == 1  # only the dirty item

    # both items present in the final result (all indices retained)
    assert {r.index for r in run.results} == {0, 1}
    by_index = {r.index: r.value for r in run.results}
    assert by_index[0] == {"cleaned": "yes"}   # dirty -> cleaned
    assert by_index[1] == {"dirty": False}     # clean -> finalized at stage 1
    # token totals span both stages
    assert run.total_completion_tokens > 0
    assert run.model == "gemma2:9b -> qwen2.5-14b-16k"


@respx.mock
async def test_pipeline_all_filtered_stops_early() -> None:
    # Everything classified clean -> stage 2 never runs.
    respx.post(f"{BASE}/api/generate").mock(return_value=httpx.Response(
        200, json={"response": '{"dirty": false}', "eval_count": 1,
                   "eval_duration": 1_000_000_000}))
    run = await run_pipeline(_cfg(), _pipeline_spec())
    assert len(run.stages) == 1  # broke after stage 1 (nothing continued)
    assert run.stages[0]["continued"] == 0
    assert all(r.value == {"dirty": False} for r in run.results)
