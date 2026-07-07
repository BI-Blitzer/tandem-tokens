"""Contract tests for the cumulative savings ledger + scoreboard."""

from __future__ import annotations

from pathlib import Path

from ollamaq.ledger import append_entry, read_entries, summarize


def _entry(**over) -> dict:
    e = {"job_id": "j", "name": "n", "mode": "map", "model": "gemma2:9b",
         "items": 3, "ok": 3, "failed": 0, "cached": 0, "escalated": 0,
         "wall_seconds": 10.0, "prompt_tokens": 100, "completion_tokens": 200,
         "estimated_cloud_cost_usd": {"claude-opus-4-8": 0.006, "claude-haiku-4-5": 0.001}}
    e.update(over)
    return e


def test_append_and_read_roundtrip(tmp_path: Path) -> None:
    append_entry(tmp_path, _entry(job_id="a"))
    append_entry(tmp_path, _entry(job_id="b"))
    entries = read_entries(tmp_path)
    assert [e["job_id"] for e in entries] == ["a", "b"]
    assert all("ts" in e for e in entries)  # timestamp added


def test_read_missing_is_empty(tmp_path: Path) -> None:
    assert read_entries(tmp_path) == []


def test_read_skips_corrupt_line(tmp_path: Path) -> None:
    append_entry(tmp_path, _entry(job_id="a"))
    with (tmp_path / "ledger.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{ half-written line\n")
    append_entry(tmp_path, _entry(job_id="c"))
    assert [e["job_id"] for e in read_entries(tmp_path)] == ["a", "c"]


def test_summarize_totals(tmp_path: Path) -> None:
    entries = [_entry(prompt_tokens=100, completion_tokens=200, items=3, wall_seconds=10),
               _entry(prompt_tokens=50, completion_tokens=50, items=2, wall_seconds=5)]
    sb = summarize(entries)
    assert sb.jobs == 2
    assert sb.items == 5
    assert sb.prompt_tokens == 150 and sb.completion_tokens == 250
    assert sb.total_tokens == 400
    assert sb.wall_seconds == 15.0
    # costs summed per model
    assert round(sb.cost_by_model["claude-opus-4-8"], 4) == 0.012
    assert round(sb.cost_by_model["claude-haiku-4-5"], 4) == 0.002


def test_summarize_empty() -> None:
    assert "no jobs recorded" in summarize([]).format()


def test_format_headlines_model(tmp_path: Path) -> None:
    sb = summarize([_entry()])
    out = sb.format(headline_model="claude-opus-4-8")
    # headline model appears before the cheaper one in the cost list
    assert out.index("claude-opus-4-8") < out.index("claude-haiku-4-5")
    assert "ran on local GPU for $0" in out
