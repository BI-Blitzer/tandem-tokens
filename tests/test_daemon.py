"""Contract tests for the daemon drain loop, HTTP mocked via respx."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx

from ollamaq.config import Config
from ollamaq.daemon import pick_next, run_daemon
from ollamaq.queue import JobQueue, State

BASE = "http://localhost:11434"


def _cfg() -> Config:
    return Config.model_validate({
        "ollama": {"base_url": BASE},
        "defaults": {"model": "gemma2:9b"},
        "models": {"gemma2:9b": {"max_parallel": 4, "num_ctx": 8192},
                   "qwen3:8b": {"max_parallel": 4, "num_ctx": 8192}},
        "routing": {"clean": "gemma2:9b", "generate": "qwen3:8b"},
    })


def _spec(model: str = "gemma2:9b") -> dict:
    return {"name": "j", "model": model,
            "input": {"kind": "inline", "values": [1, 2], "chunk": {"strategy": "rows", "size": 1}},
            "prompt": {"template": "{{chunk}}"}}


def _ok_resp() -> httpx.Response:
    return httpx.Response(200, json={"response": "hi", "eval_count": 3,
                                     "eval_duration": 1_000_000_000})


@respx.mock
def test_daemon_drains_to_done(tmp_path: Path) -> None:
    respx.post(f"{BASE}/api/generate").mock(return_value=_ok_resp())
    q = JobQueue(tmp_path / "queue")
    q.submit("a", _spec())
    q.submit("b", _spec())
    n = run_daemon(_cfg(), q, results_root=tmp_path / "results",
                   max_jobs=2, should_stop=lambda: False, log=lambda m: None)
    assert n == 2
    assert q.find("a")[0] == State.done
    assert q.find("b")[0] == State.done
    assert (tmp_path / "results" / "a" / "digest.md").exists()


@respx.mock
def test_daemon_marks_failed_below_threshold(tmp_path: Path) -> None:
    # schema required but model returns non-JSON -> 0% success -> failed
    respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "not json", "eval_count": 2,
                                               "eval_duration": 1_000_000_000}))
    q = JobQueue(tmp_path / "queue")
    spec = _spec()
    spec["prompt"]["schema"] = {"type": "object", "required": ["x"]}
    q.submit("a", spec)
    run_daemon(_cfg(), q, results_root=tmp_path / "results",
               max_jobs=1, should_stop=lambda: False, log=lambda m: None)
    st, job = q.find("a")
    assert st == State.failed
    assert "success" in job.error


@respx.mock
def test_pick_next_prefers_loaded_model(tmp_path: Path) -> None:
    respx.post(f"{BASE}/api/generate").mock(return_value=_ok_resp())
    q = JobQueue(tmp_path / "queue")
    q.submit("gem", _spec("gemma2:9b"))
    q.submit("qwen", _spec("qwen3:8b"))
    # prefer qwen3:8b even though 'gem' was enqueued first
    job = pick_next(q, _cfg(), prefer_model="qwen3:8b")
    assert job.job_id == "qwen" and job.model == "qwen3:8b"


def test_daemon_empty_queue_stops(tmp_path: Path) -> None:
    q = JobQueue(tmp_path / "queue")
    n = run_daemon(_cfg(), q, results_root=tmp_path / "results",
                   should_stop=lambda: False, log=lambda m: None)
    assert n == 0  # empty + should_stop provided -> exits immediately


@respx.mock
def test_daemon_reclaims_stale_on_startup(tmp_path: Path) -> None:
    respx.post(f"{BASE}/api/generate").mock(return_value=_ok_resp())
    q = JobQueue(tmp_path / "queue")
    q.submit("a", _spec())
    q.claim_one("a")  # leaves 'a' in running/ with a fresh lease
    # force the lease to look ancient by reclaiming with a negative timeout via daemon:
    # run_daemon calls reclaim_stale(LEASE_TIMEOUT_S); simulate abandonment by
    # manually aging is covered in test_queue — here just confirm a running job
    # left behind doesn't block draining a fresh one.
    q.submit("b", _spec())
    run_daemon(_cfg(), q, results_root=tmp_path / "results",
               max_jobs=1, should_stop=lambda: False, log=lambda m: None)
    # 'b' should have been processed (a stays running, not abandoned within timeout)
    assert q.find("b")[0] == State.done
