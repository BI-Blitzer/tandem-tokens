"""Contract tests for the bench module. HTTP mocked; no live server."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx

from ollamaq.bench import BenchReport, run_bench, write_report
from ollamaq.config import Config

BASE = "http://localhost:11434"


def _cfg() -> Config:
    return Config.model_validate(
        {
            "ollama": {"base_url": BASE},
            "defaults": {"model": "gemma2:9b"},
            "models": {
                "gemma2:9b": {"max_parallel": 4, "num_ctx": 8192},
                "ghost": {"enabled": False},
            },
            "routing": {"classify": "gemma2:9b"},
        }
    )


@respx.mock
async def test_run_bench_only_enabled_models() -> None:
    route = respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(
            200,
            json={"response": "one\ntwo", "eval_count": 20, "eval_duration": 1_000_000_000},
        )
    )
    report = await run_bench(_cfg())
    # 'ghost' is disabled -> not benched.
    assert [r.model for r in report.results] == ["gemma2:9b"]
    assert route.call_count == 1
    r = report.results[0]
    assert r.ok is True
    assert r.tokens_per_second == 20.0


@respx.mock
async def test_run_bench_records_failure() -> None:
    respx.post(f"{BASE}/api/generate").mock(return_value=httpx.Response(500, text="boom"))
    report = await run_bench(_cfg(), models=["gemma2:9b"])
    r = report.results[0]
    assert r.ok is False
    assert "500" in r.error


def test_write_report_roundtrip(tmp_path: Path) -> None:
    report = BenchReport(generated_at="now", base_url=BASE, results=[])
    out = write_report(report, tmp_path / "sub" / "bench.json")
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["base_url"] == BASE
