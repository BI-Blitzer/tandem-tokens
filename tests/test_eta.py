"""Contract tests for ETA projection math and bench lookup."""

from __future__ import annotations

import json
from pathlib import Path

from ollamaq.eta import bench_lookup, project
from ollamaq.worker import ItemResult, RunResult


def _sample(n: int, completion_each: int, eval_ns_each: int) -> RunResult:
    return RunResult(
        model="gemma2:9b",
        results=[ItemResult(i, f"s{i}", True, {"x": i}, completion_tokens=completion_each,
                            eval_duration_ns=eval_ns_each) for i in range(n)],
        wall_seconds=1.0,
    )


def test_projection_scales_linearly() -> None:
    # 2 items, 100 completion tokens each, 1s decode each -> 100 tok/s per stream
    s = _sample(2, 100, 1_000_000_000)
    p = project(s, total_items=10, max_parallel=1)
    assert p.mean_completion_tokens == 100
    assert p.est_completion_tokens == 1000          # 100 * 10
    assert p.tokens_per_second == 100.0
    assert p.est_seconds == 10.0                     # 1000 / (100 * 1)


def test_projection_divides_by_parallelism() -> None:
    s = _sample(2, 100, 1_000_000_000)  # 100 tok/s
    p = project(s, total_items=10, max_parallel=4)
    assert p.est_seconds == 2.5                       # 1000 / (100 * 4)


def test_projection_adds_load_time() -> None:
    s = _sample(1, 100, 1_000_000_000)
    p = project(s, total_items=10, max_parallel=1, load_seconds=22.0)
    assert p.est_seconds == 1000 / 100 + 22.0         # 32.0


def test_projection_uses_tps_fallback_when_no_eval() -> None:
    s = _sample(2, 100, 0)  # no decode time reported
    p = project(s, total_items=10, max_parallel=1, tps_fallback=50.0)
    assert p.tokens_per_second == 50.0
    assert p.est_seconds == 1000 / 50.0


def test_projection_zero_tps_is_safe() -> None:
    s = _sample(1, 100, 0)
    p = project(s, total_items=10, max_parallel=1)  # no fallback
    assert p.est_seconds == 0.0
    assert "cannot estimate" in p.format()


def test_projection_format_mentions_counts() -> None:
    s = _sample(2, 100, 1_000_000_000)
    txt = project(s, total_items=10, max_parallel=2).format()
    assert "10 items" in txt and "tok/s" in txt


def test_bench_lookup_hit(tmp_path: Path) -> None:
    p = tmp_path / "bench.json"
    p.write_text(json.dumps({"results": [
        {"model": "gemma2:9b", "tokens_per_second": 88.0, "load_seconds": 22.0}]}),
        encoding="utf-8")
    assert bench_lookup("gemma2:9b", p) == (88.0, 22.0)


def test_bench_lookup_missing_file_or_model(tmp_path: Path) -> None:
    assert bench_lookup("x", tmp_path / "nope.json") == (0.0, 0.0)
    p = tmp_path / "bench.json"
    p.write_text(json.dumps({"results": []}), encoding="utf-8")
    assert bench_lookup("x", p) == (0.0, 0.0)
