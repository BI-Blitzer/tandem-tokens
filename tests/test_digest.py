"""Contract tests for artifact writing and the 8 KB digest cap."""

from __future__ import annotations

import json
from pathlib import Path

from ollamaq.digest import DIGEST_MAX_BYTES, build_digest_md, write_results
from ollamaq.jobspec import JobSpec
from ollamaq.pricing import Pricing
from ollamaq.worker import ItemResult, RunResult


def _pricing() -> Pricing:
    return Pricing.model_validate({
        "default_compare": "claude-opus-4-8",
        "models": {
            "claude-opus-4-8": {"input_per_mtok": 5.0, "output_per_mtok": 25.0,
                                "label": "Claude Opus 4.8"},
            "claude-haiku-4-5": {"input_per_mtok": 1.0, "output_per_mtok": 5.0,
                                 "label": "Claude Haiku 4.5"},
        },
    })


def _spec() -> JobSpec:
    return JobSpec.model_validate(
        {
            "name": "demo",
            "model": "gemma2:9b",
            "input": {"kind": "inline", "values": [1], "chunk": {"strategy": "rows", "size": 1}},
            "prompt": {"template": "{{chunk}}"},
        }
    )


def _run(results) -> RunResult:
    return RunResult(model="gemma2:9b", results=results, wall_seconds=1.0)


def test_write_results_layout(tmp_path: Path) -> None:
    run = _run([
        ItemResult(0, "src0", True, {"id": 0}, attempts=1, completion_tokens=5,
                   eval_duration_ns=1_000_000_000),
        ItemResult(1, "src1", False, error="schema violation: bad", attempts=3),
    ])
    d = write_results("demo-1", _spec(), run, tmp_path)
    assert (d / "items" / "0.json").exists()
    assert not (d / "items" / "1.json").exists()  # failures not written as items

    merged = [json.loads(l) for l in (d / "merged.jsonl").read_text().splitlines()]
    assert merged == [{"index": 0, "source": "src0", "value": {"id": 0}}]

    failed = [json.loads(l) for l in (d / "failed_items.jsonl").read_text().splitlines()]
    assert failed[0]["index"] == 1 and "schema violation" in failed[0]["error"]

    stats = json.loads((d / "run.json").read_text())
    assert stats["ok"] == 1 and stats["failed"] == 1
    assert (d / "digest.md").exists()


def test_digest_reports_counts(tmp_path: Path) -> None:
    run = _run([ItemResult(0, "s", True, {"ok": True}, attempts=1)])
    md = build_digest_md("demo-1", _spec(), run, tmp_path)
    assert "ok: 1" in md
    assert "failed: 0" in md
    assert "failures: none" in md


def test_digest_shows_cache_hits(tmp_path: Path) -> None:
    run = _run([
        ItemResult(0, "s0", True, {"a": 1}, attempts=1),
        ItemResult(1, "s1", True, {"a": 2}, cached=True, attempts=0),
    ])
    md = build_digest_md("j", _spec(), run, tmp_path)
    assert "cache: 1/2 items served from cache" in md


def test_digest_no_cache_line_when_none(tmp_path: Path) -> None:
    run = _run([ItemResult(0, "s", True, {"a": 1}, attempts=1)])
    md = build_digest_md("j", _spec(), run, tmp_path)
    assert "served from cache" not in md


def test_digest_shows_escalations(tmp_path: Path) -> None:
    run = _run([
        ItemResult(0, "s0", True, {"a": 1}, attempts=1),
        ItemResult(1, "s1", True, {"a": 2}, escalated=True, attempts=3),
    ])
    md = build_digest_md("j", _spec(), run, tmp_path)
    assert "escalated: 1 item(s) recovered" in md


def test_digest_capped_at_8kb(tmp_path: Path) -> None:
    # 500 failures with long errors would blow past 8 KB if uncapped
    results = [
        ItemResult(i, f"source-number-{i}", False, error="x" * 300, attempts=3)
        for i in range(500)
    ]
    md = build_digest_md("big", _spec(), _run(results), tmp_path)
    assert len(md.encode("utf-8")) <= DIGEST_MAX_BYTES
    assert "truncated" in md


def test_digest_savings_block(tmp_path: Path) -> None:
    run = RunResult(
        model="gemma2:9b",
        results=[ItemResult(0, "s", True, {"a": 1}, attempts=1,
                            prompt_tokens=1_000_000, completion_tokens=1_000_000)],
        wall_seconds=1.0,
    )
    md = build_digest_md("j", _spec(), run, tmp_path, _pricing())
    assert "estimated cloud cost avoided" in md
    assert "Claude Opus 4.8: ~$30.0000" in md  # 1M in @ $5 + 1M out @ $25
    assert "$0" in md  # ran locally for free
    assert "reduce/clean pass" in md  # honest caveat present


def test_digest_no_savings_without_pricing(tmp_path: Path) -> None:
    run = _run([ItemResult(0, "s", True, {"a": 1}, attempts=1)])
    md = build_digest_md("j", _spec(), run, tmp_path)  # pricing omitted
    assert "estimated cloud cost" not in md


def test_write_results_records_cost(tmp_path: Path) -> None:
    run = RunResult(
        model="gemma2:9b",
        results=[ItemResult(0, "s", True, {"a": 1}, attempts=1,
                            prompt_tokens=200_000, completion_tokens=100_000)],
        wall_seconds=1.0,
    )
    d = write_results("cost-1", _spec(), run, tmp_path, _pricing())
    stats = json.loads((d / "run.json").read_text())
    assert stats["prompt_tokens"] == 200_000
    assert "estimated_cloud_cost_usd" in stats
    # 0.2M in @ $5 + 0.1M out @ $25 = $1.00 + $2.50 = $3.50
    assert stats["estimated_cloud_cost_usd"]["claude-opus-4-8"] == 3.5


def test_digest_gentest_leaderboard(tmp_path: Path) -> None:
    run = RunResult(
        model="qwen3:8b",
        results=[
            ItemResult(0, "iter3", True, {"text": "best"}, score=0.9, verdict_notes="n=9"),
            ItemResult(1, "iter2", False, {"text": "meh"}, score=0.4, verdict_notes="n=4"),
        ],
        wall_seconds=2.0,
        iterations=[
            {"iteration": 1, "candidates": 4, "best_score": 0.3, "global_best": 0.3},
            {"iteration": 2, "candidates": 4, "best_score": 0.9, "global_best": 0.9},
        ],
        prompt_tokens_total=100, completion_tokens_total=200, eval_ns_total=1_000_000_000,
    )
    md = build_digest_md("g", _spec(), run, tmp_path)
    assert "leaderboard" in md
    assert "0.900 PASS" in md          # top candidate, passed
    assert "n=9" in md                 # verdict notes
    assert "## iterations" in md
    assert "iter 2:" in md and "global_best 0.9" in md
    # map-only sections should be absent
    assert "sample outputs" not in md


def test_digest_pipeline_stages(tmp_path: Path) -> None:
    run = RunResult(
        model="gemma2:9b -> qwen2.5-14b-16k",
        results=[ItemResult(0, "r0", True, {"cleaned": "yes"}, attempts=1),
                 ItemResult(1, "r1", True, {"dirty": False}, attempts=1)],
        wall_seconds=3.0,
        stages=[
            {"stage": "classify", "model": "gemma2:9b", "in": 2, "ok": 2,
             "failed": 0, "continued": 1},
            {"stage": "clean", "model": "qwen2.5-14b-16k", "in": 1, "ok": 1,
             "failed": 0, "continued": 0},
        ],
        prompt_tokens_total=50, completion_tokens_total=80, eval_ns_total=1_000_000_000,
    )
    md = build_digest_md("p", _spec(), run, tmp_path)
    assert "## pipeline stages" in md
    assert "classify (gemma2:9b): in 2, ok 2, failed 0, continued 1" in md
    assert "sample outputs" in md  # pipeline uses the map-shaped body too


def test_digest_lists_failures(tmp_path: Path) -> None:
    run = _run([
        ItemResult(0, "s0", True, {"a": 1}, attempts=1),
        ItemResult(1, "s1", False, error="boom", attempts=2),
    ])
    md = build_digest_md("j", _spec(), run, tmp_path)
    assert "failures (1)" in md
    assert "boom" in md
