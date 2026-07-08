"""Contract tests for the watch dashboard's snapshot + render (pure logic)."""

from __future__ import annotations

from pathlib import Path

from ollamaq.dashboard import render, snapshot
from ollamaq.ledger import append_entry
from ollamaq.queue import JobQueue, State

SPEC = {"name": "j", "model": "gemma2:9b",
        "input": {"kind": "inline", "values": [1], "chunk": {"strategy": "rows", "size": 1}},
        "prompt": {"template": "{{chunk}}"}}


def test_snapshot_counts_and_running(tmp_path: Path) -> None:
    q = JobQueue(tmp_path / "queue")
    q.submit("a", SPEC)
    q.submit("b", SPEC)
    q.claim_one("b", model="gemma2:9b")  # b -> running
    snap = snapshot(q, tmp_path / "results")
    assert snap.counts["pending"] == 1 and snap.counts["running"] == 1
    assert snap.running == [("b", "gemma2:9b")]


def test_snapshot_scoreboard_from_ledger(tmp_path: Path) -> None:
    rroot = tmp_path / "results"
    append_entry(rroot, {"job_id": "x", "items": 3, "prompt_tokens": 100,
                         "completion_tokens": 200, "wall_seconds": 5,
                         "estimated_cloud_cost_usd": {"claude-opus-4-8": 0.02}})
    snap = snapshot(JobQueue(tmp_path / "queue"), rroot)
    assert snap.scoreboard.jobs == 1 and snap.scoreboard.total_tokens == 300
    assert snap.recent[-1]["job_id"] == "x"


def test_render_is_ascii_and_shows_sections(tmp_path: Path) -> None:
    q = JobQueue(tmp_path / "queue")
    q.submit("a", SPEC)
    q.claim_one("a", model="gemma2:9b")
    append_entry(tmp_path / "results", {"job_id": "done1", "items": 4, "ok": 4,
                 "prompt_tokens": 50, "completion_tokens": 60, "wall_seconds": 2,
                 "model": "gemma2:9b",
                 "estimated_cloud_cost_usd": {"claude-opus-4-8": 0.01}})
    out = render(snapshot(q, tmp_path / "results"), now="12:00:00")
    assert "ollamaq watch  12:00:00" in out
    assert "running now:" in out and "a  (gemma2:9b)" in out
    assert "recent finished:" in out and "done1" in out
    assert "scoreboard: 1 jobs" in out
    assert out.isascii()  # console-safe (cp1252 on Windows)


def test_render_empty_queue(tmp_path: Path) -> None:
    out = render(snapshot(JobQueue(tmp_path / "queue"), tmp_path / "results"))
    assert "pending 0" in out and "scoreboard: 0 jobs" in out
    assert out.isascii()
