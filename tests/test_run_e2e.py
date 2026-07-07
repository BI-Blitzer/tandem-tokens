"""End-to-end: `ollamaq run` over a CSV, HTTP mocked. Proves the whole
chunk -> fan-out -> digest pipeline through the CLI, including exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx
from typer.testing import CliRunner

from ollamaq.cli import app

runner = CliRunner()
BASE = "http://localhost:11434"


def _job(tmp_path: Path, csv_rows: int = 6) -> tuple[Path, Path]:
    csv = tmp_path / "data.csv"
    lines = ["id,raw"] + [f"{i},value{i}" for i in range(csv_rows)]
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    spec = {
        "name": "clean",
        "model": "gemma2:9b",
        "input": {"kind": "csv", "path": str(csv),
                  "chunk": {"strategy": "rows", "size": 2}},
        "prompt": {"template": "clean {{chunk}}",
                   "schema": {"type": "object", "required": ["cleaned"]}},
    }
    jf = tmp_path / "job.json"
    jf.write_text(json.dumps(spec), encoding="utf-8")
    return jf, csv


def _cfg_file(tmp_path: Path) -> Path:
    p = tmp_path / "models.yaml"
    p.write_text(
        "ollama: {base_url: '%s'}\n"
        "defaults: {model: gemma2:9b}\n"
        "models: {gemma2:9b: {max_parallel: 4, num_ctx: 8192}}\n"
        "routing: {clean: gemma2:9b}\n" % BASE,
        encoding="utf-8",
    )
    return p


@respx.mock
def test_run_success(tmp_path: Path) -> None:
    respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(200, json={
            "response": '{"cleaned": true}', "eval_count": 4, "eval_duration": 1_000_000_000
        })
    )
    jf, _ = _job(tmp_path)
    res = runner.invoke(app, [
        "run", str(jf), "--config", str(_cfg_file(tmp_path)),
        "--results", str(tmp_path / "results"), "--job-id", "clean-1",
    ])
    assert res.exit_code == 0, res.output
    digest = (tmp_path / "results" / "clean-1" / "digest.md").read_text()
    assert "ok: 3" in digest  # 6 rows / 2 = 3 chunks
    merged = (tmp_path / "results" / "clean-1" / "merged.jsonl").read_text().splitlines()
    assert len(merged) == 3


@respx.mock
def test_run_below_threshold_exits_1(tmp_path: Path) -> None:
    respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(200, json={
            "response": "not json", "eval_count": 2, "eval_duration": 1_000_000_000
        })
    )
    jf, _ = _job(tmp_path)
    res = runner.invoke(app, [
        "run", str(jf), "--config", str(_cfg_file(tmp_path)),
        "--results", str(tmp_path / "results"), "--job-id", "clean-2",
        "--min-success", "0.95",
    ])
    assert res.exit_code == 1
    # digest still written even on failure
    assert (tmp_path / "results" / "clean-2" / "digest.md").exists()


def test_run_no_wait_points_to_submit(tmp_path: Path) -> None:
    jf, _ = _job(tmp_path)
    res = runner.invoke(app, [
        "run", str(jf), "--config", str(_cfg_file(tmp_path)), "--no-wait",
    ])
    assert res.exit_code == 2
    out = res.output + (res.stderr or "")
    assert "submit" in out


@respx.mock
def test_run_sample_probes_n_and_stops(tmp_path: Path) -> None:
    route = respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(200, json={
            "response": '{"cleaned": true}', "eval_count": 4, "eval_duration": 1_000_000_000}))
    jf, _ = _job(tmp_path, csv_rows=6)  # 6 rows / 2 = 3 items total
    res = runner.invoke(app, [
        "run", str(jf), "--config", str(_cfg_file(tmp_path)), "--sample", "1",
    ])
    assert res.exit_code == 0, res.output
    assert route.call_count == 1              # only the 1 sampled item ran
    out = res.output
    assert "sampling 1 of 3 items" in out
    assert "projection" in out and "3 items" in out
    # sample must NOT write a full results dir
    assert not (tmp_path / "results").exists()


def test_run_sample_rejects_non_map(tmp_path: Path) -> None:
    spec = {
        "name": "g", "mode": "gen_test", "model": "gemma2:9b",
        "input": {"kind": "inline", "values": [1], "chunk": {"strategy": "rows", "size": 1}},
        "prompt": {"template": "{{chunk}}"},
        "verifier": {"cmd": "python v.py"},
    }
    jf = tmp_path / "g.json"
    jf.write_text(json.dumps(spec), encoding="utf-8")
    res = runner.invoke(app, [
        "run", str(jf), "--config", str(_cfg_file(tmp_path)), "--sample", "2",
    ])
    assert res.exit_code == 2
    assert "only meaningful for map" in (res.output + (res.stderr or ""))


def test_score_command_aggregates_ledger(tmp_path: Path) -> None:
    from ollamaq.ledger import append_entry
    rroot = tmp_path / "results"
    append_entry(rroot, {"job_id": "a", "items": 3, "prompt_tokens": 100,
                         "completion_tokens": 200, "wall_seconds": 10,
                         "estimated_cloud_cost_usd": {"claude-opus-4-8": 0.02}})
    append_entry(rroot, {"job_id": "b", "items": 2, "prompt_tokens": 50,
                         "completion_tokens": 50, "wall_seconds": 5,
                         "estimated_cloud_cost_usd": {"claude-opus-4-8": 0.01}})
    res = runner.invoke(app, ["score", "--results", str(rroot)])
    assert res.exit_code == 0
    out = res.output
    assert "2 job(s), 5 items" in out
    assert "400 total" in out                     # 150 in + 250 out
    assert "claude-opus-4-8: ~$0.0300" in out     # summed


def test_score_empty(tmp_path: Path) -> None:
    res = runner.invoke(app, ["score", "--results", str(tmp_path / "results")])
    assert res.exit_code == 0
    assert "no jobs recorded" in res.output


def test_digest_command_prints(tmp_path: Path) -> None:
    d = tmp_path / "results" / "job-x"
    d.mkdir(parents=True)
    (d / "digest.md").write_text("# hello digest\n", encoding="utf-8")
    res = runner.invoke(app, ["digest", "job-x", "--results", str(tmp_path / "results")])
    assert res.exit_code == 0
    assert "hello digest" in res.output
