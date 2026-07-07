"""Contract tests for the file-based queue: state transitions, claim races,
crash recovery, cancellation."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ollamaq.errors import OllamaqError
from ollamaq.queue import JobQueue, State

SPEC = {"name": "j", "model": "gemma2:9b",
        "input": {"kind": "inline", "values": [1], "chunk": {"strategy": "rows", "size": 1}},
        "prompt": {"template": "{{chunk}}"}}


def _q(tmp_path: Path) -> JobQueue:
    return JobQueue(tmp_path / "queue")


def test_submit_creates_pending(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.submit("a", SPEC)
    assert [j.job_id for j in q.list_state(State.pending)] == ["a"]
    assert (tmp_path / "queue" / "pending" / "a.json").exists()


def test_duplicate_submit_rejected(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.submit("a", SPEC)
    with pytest.raises(OllamaqError, match="already exists"):
        q.submit("a", SPEC)


def test_list_oldest_first(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.submit("a", SPEC)
    time.sleep(0.01)
    q.submit("b", SPEC)
    assert [j.job_id for j in q.list_state(State.pending)] == ["a", "b"]


def test_claim_moves_to_running(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.submit("a", SPEC)
    job = q.claim_one("a", model="gemma2:9b")
    assert job is not None and job.state == State.running
    assert job.model == "gemma2:9b" and job.leased_at > 0
    assert not (tmp_path / "queue" / "pending" / "a.json").exists()
    assert (tmp_path / "queue" / "running" / "a.json").exists()


def test_claim_next_oldest(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.submit("a", SPEC)
    time.sleep(0.01)
    q.submit("b", SPEC)
    assert q.claim_next().job_id == "a"


def test_double_claim_returns_none(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.submit("a", SPEC)
    assert q.claim_one("a") is not None
    assert q.claim_one("a") is None  # already claimed


def test_finish_to_done(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.submit("a", SPEC)
    job = q.claim_one("a")
    q.finish(job, State.done)
    st, refreshed = q.find("a")
    assert st == State.done and refreshed.finished_at > 0
    assert not (tmp_path / "queue" / "running" / "a.json").exists()


def test_finish_to_failed_records_error(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.submit("a", SPEC)
    job = q.claim_one("a")
    q.finish(job, State.failed, error="boom")
    st, refreshed = q.find("a")
    assert st == State.failed and refreshed.error == "boom"


def test_reclaim_stale(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.submit("a", SPEC)
    q.claim_one("a")
    # lease is fresh -> not reclaimed
    assert q.reclaim_stale(timeout_s=3600) == []
    # lease is ancient -> reclaimed
    assert q.reclaim_stale(timeout_s=-1) == ["a"]
    assert q.find("a")[0] == State.pending


def test_cancel_pending(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.submit("a", SPEC)
    assert q.cancel("a") is True
    assert q.find("a") is None


def test_cancel_running_fails(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.submit("a", SPEC)
    q.claim_one("a")
    assert q.cancel("a") is False  # not pending anymore


def test_find_missing(tmp_path: Path) -> None:
    assert _q(tmp_path).find("nope") is None


def test_corrupt_file_skipped(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.submit("a", SPEC)
    (tmp_path / "queue" / "pending" / "bad.json").write_text("{not json", encoding="utf-8")
    # list should skip the corrupt file, not crash
    assert [j.job_id for j in q.list_state(State.pending)] == ["a"]
