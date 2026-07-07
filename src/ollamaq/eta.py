"""Project a full map job's time/token cost from a small sample run.

`ollamaq run <job> --sample N` runs the first N items, measures their real
throughput, and extrapolates to the whole input. Pure functions here (no I/O
beyond an optional bench.json lookup) so the math is unit-testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .worker import RunResult


@dataclass
class Projection:
    sample_items: int
    total_items: int
    mean_completion_tokens: float
    tokens_per_second: float          # measured per-stream decode speed
    max_parallel: int
    est_completion_tokens: int
    est_seconds: float                # includes model load

    def format(self) -> str:
        if self.tokens_per_second <= 0:
            return (f"projection: sampled {self.sample_items} item(s) but measured "
                    f"0 tok/s — cannot estimate (is the model responding?)")
        mins = self.est_seconds / 60
        # ASCII only — this prints to the console, which is cp1252 on Windows.
        return (
            f"projection from {self.sample_items}-item sample -> {self.total_items} items:\n"
            f"  ~{self.est_completion_tokens} completion tokens total "
            f"(~{self.mean_completion_tokens:.0f}/item)\n"
            f"  ~{self.est_seconds:.0f}s (~{mins:.1f} min) at "
            f"{self.tokens_per_second:.0f} tok/s x {self.max_parallel} parallel"
        )


def project(sample: RunResult, total_items: int, max_parallel: int,
            tps_fallback: float = 0.0, load_seconds: float = 0.0) -> Projection:
    """Extrapolate `sample` (a RunResult over the first N items) to `total_items`.

    Time model: total completion tokens / (per-stream tok/s × parallel) + load.
    Uses the sample's measured tok/s; falls back to `tps_fallback` (e.g. bench)
    only if the sample reported no decode time."""
    n = len(sample.results)
    mean_completion = sample.total_completion_tokens / n if n else 0.0
    tps = sample.tokens_per_second or tps_fallback
    est_completion = int(round(mean_completion * total_items))
    if tps > 0:
        est_seconds = est_completion / (tps * max_parallel) + load_seconds
    else:
        est_seconds = 0.0
    return Projection(
        sample_items=n, total_items=total_items,
        mean_completion_tokens=mean_completion, tokens_per_second=tps,
        max_parallel=max_parallel, est_completion_tokens=est_completion,
        est_seconds=round(est_seconds, 1),
    )


def bench_lookup(model: str, path: str | Path = "config/bench.json") -> tuple[float, float]:
    """(tokens_per_second, load_seconds) for `model` from bench.json, or (0, 0)
    if the file or model is missing. Never raises."""
    p = Path(path)
    if not p.is_file():
        return 0.0, 0.0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        for r in data.get("results", []):
            if r.get("model") == model:
                return float(r.get("tokens_per_second", 0)), float(r.get("load_seconds", 0))
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return 0.0, 0.0
