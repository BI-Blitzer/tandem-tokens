"""Benchmark installed models: tokens/sec and cold-load time.

Results are written to config/bench.json and later consumed by the runner to
estimate job ETAs. This module has no CLI concerns - cli.py calls run_bench().
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from .client import OllamaClient
from .config import Config
from .errors import OllamaqError

# Fixed prompt so numbers are comparable across runs and machines.
BENCH_PROMPT = (
    "Count from 1 to 40 in words, one number per line, then stop. "
    "Do not add any commentary."
)
BENCH_SYSTEM = "You are a terse assistant. Follow the instruction exactly."


class BenchResult(BaseModel):
    """One model's benchmark outcome. ok=False carries the error in ``error``."""

    model: str
    ok: bool
    tokens_per_second: float = 0.0
    load_seconds: float = 0.0
    completion_tokens: int = 0
    wall_seconds: float = 0.0
    error: str = ""


class BenchReport(BaseModel):
    generated_at: str
    base_url: str
    results: list[BenchResult]

    def by_model(self) -> dict[str, BenchResult]:
        return {r.model: r for r in self.results}


async def bench_one(client: OllamaClient, model: str) -> BenchResult:
    """Benchmark a single model. Never raises - failures land in BenchResult.error."""
    start = time.perf_counter()
    try:
        res = await client.generate(
            model,
            BENCH_PROMPT,
            system=BENCH_SYSTEM,
            options={"temperature": 0.0},
        )
    except OllamaqError as exc:
        return BenchResult(model=model, ok=False, error=str(exc))
    wall = time.perf_counter() - start
    return BenchResult(
        model=model,
        ok=True,
        tokens_per_second=round(res.tokens_per_second, 2),
        load_seconds=round(res.load_seconds, 3),
        completion_tokens=res.completion_tokens,
        wall_seconds=round(wall, 3),
    )


async def run_bench(config: Config, models: list[str] | None = None) -> BenchReport:
    """Benchmark the given models (default: every enabled model in config).

    Runs strictly sequentially - benchmarking in parallel would contend for VRAM
    and corrupt the tokens/sec numbers.
    """
    targets = models or [name for name, m in config.models.items() if m.enabled]
    results: list[BenchResult] = []
    async with OllamaClient(config.ollama) as client:
        for model in targets:
            results.append(await bench_one(client, model))
    return BenchReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        base_url=config.ollama.base_url,
        results=results,
    )


def write_report(report: BenchReport, path: str | Path) -> Path:
    """Serialize a BenchReport to JSON. Returns the written path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return out
