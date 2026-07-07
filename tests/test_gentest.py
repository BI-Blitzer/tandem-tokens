"""Contract tests for the gen_test loop. HTTP mocked; verifier monkeypatched."""

from __future__ import annotations

import itertools

import httpx
import respx

from ollamaq.config import Config
from ollamaq import gentest
from ollamaq.gentest import run_gentest
from ollamaq.jobspec import JobSpec
from ollamaq.verify import Verdict

BASE = "http://localhost:11434"


def _cfg() -> Config:
    return Config.model_validate({
        "ollama": {"base_url": BASE},
        "defaults": {"model": "qwen3:8b"},
        "models": {"qwen3:8b": {"max_parallel": 2, "num_ctx": 8192}},
        "routing": {"generate": "qwen3:8b"},
    })


def _spec(stop_when: str = "", max_iterations: int = 5, candidates: int = 2,
          keep_top: int = 3, template: str = "make one {{chunk}}") -> JobSpec:
    return JobSpec.model_validate({
        "name": "g",
        "mode": "gen_test",
        "model": "qwen3:8b",
        "input": {"kind": "inline", "values": ["thing"], "chunk": {"strategy": "rows", "size": 1}},
        "prompt": {"template": template},
        "verifier": {"cmd": "python fake.py", "keep_top": keep_top},
        "loop": {"candidates": candidates, "max_iterations": max_iterations,
                 "stop_when": stop_when},
    })


def _mock_incrementing_gen() -> None:
    """Each generation returns text 'c<N>' with N incrementing across all calls."""
    counter = itertools.count()

    def gen(request):
        n = next(counter)
        return httpx.Response(200, json={"response": f"c{n}", "eval_count": 1,
                                         "eval_duration": 1_000_000_000})

    respx.post(f"{BASE}/api/generate").mock(side_effect=gen)


def _score_by_text(monkeypatch) -> None:
    """Verifier scores 'c<N>' as N/100; passes at N>=5."""
    def fake(cmd, value, timeout_s=30.0):
        n = int(value["text"][1:])
        return Verdict(ok=True, passed=n >= 5, score=n / 100, notes=f"n={n}")
    monkeypatch.setattr(gentest, "run_verifier", fake)


@respx.mock
async def test_runs_all_iterations_without_stop(monkeypatch) -> None:
    _mock_incrementing_gen()
    _score_by_text(monkeypatch)
    run = await run_gentest(_cfg(), _spec(max_iterations=4, candidates=2))
    assert len(run.iterations) == 4  # no stop_when -> full budget
    # 4 iterations * 2 candidates = 8 generations, each 1 completion token
    assert run.total_completion_tokens == 8
    assert run.prompt_tokens_total is not None


@respx.mock
async def test_leaderboard_sorted_and_capped(monkeypatch) -> None:
    _mock_incrementing_gen()
    _score_by_text(monkeypatch)
    run = await run_gentest(_cfg(), _spec(max_iterations=5, candidates=2, keep_top=3))
    assert len(run.results) == 3  # keep_top
    scores = [r.score for r in run.results]
    assert scores == sorted(scores, reverse=True)  # best first
    assert run.results[0].verdict_notes.startswith("n=")


@respx.mock
async def test_stops_on_score_threshold(monkeypatch) -> None:
    _mock_incrementing_gen()
    _score_by_text(monkeypatch)
    # scores climb 0.00, 0.01, ...; stop once global best >= 0.05
    run = await run_gentest(_cfg(), _spec(stop_when="score>=0.05", max_iterations=20,
                                          candidates=2))
    assert run.iterations[-1]["global_best"] >= 0.05
    assert len(run.iterations) < 20  # stopped early


@respx.mock
async def test_exemplars_injected_after_first_iter(monkeypatch) -> None:
    import json

    counter = itertools.count()

    def gen(request):
        n = next(counter)
        return httpx.Response(200, json={"response": f"c{n}", "eval_count": 1,
                                         "eval_duration": 1_000_000_000})

    route = respx.post(f"{BASE}/api/generate").mock(side_effect=gen)
    _score_by_text(monkeypatch)
    await run_gentest(_cfg(), _spec(max_iterations=2, candidates=2,
                                    template="make one {{chunk}}; better than:\n{{exemplars}}"))
    # iteration 2's prompts must carry an injected exemplar line ("score 0.")
    bodies = [json.loads(c.request.content)["prompt"] for c in route.calls]
    assert any("score 0." in b for b in bodies)


@respx.mock
async def test_bad_verifier_scores_zero(monkeypatch) -> None:
    _mock_incrementing_gen()

    def bad(cmd, value, timeout_s=30.0):
        return Verdict(ok=False, passed=False, score=0.0, notes="verifier broke")
    monkeypatch.setattr(gentest, "run_verifier", bad)

    run = await run_gentest(_cfg(), _spec(max_iterations=2, candidates=2))
    assert all(r.score == 0.0 for r in run.results)
    assert all(not r.ok for r in run.results)  # nothing passed
