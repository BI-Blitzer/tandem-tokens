"""File-based job queue. Crash-safe via atomic renames, inspectable with `ls`,
zero external dependencies (no Redis/SQLite server).

A job is a single JSON file that moves between state directories:

    queue/pending/<job_id>.json     enqueued, waiting for a worker
    queue/running/<job_id>.json     claimed by the daemon (carries a lease)
    queue/done/<job_id>.json        completed; results in results/<job_id>/
    queue/failed/<job_id>.json      errored or below success threshold

Transitions use os.replace (atomic on Windows and POSIX for same-volume moves),
so a job is never in two states and never half-written into a state dir. The
daemon writes the job file to a temp name then os.replace-s it into place.

Crash recovery: a job left in running/ whose lease is older than LEASE_TIMEOUT_S
(the daemon died mid-run) is reclaimed to pending/ on the next daemon startup.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from .errors import OllamaqError


class State(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


LEASE_TIMEOUT_S = 3600.0  # a running job older than this is presumed abandoned


@dataclass
class QueuedJob:
    """A job file's metadata envelope. The actual job spec is nested under `spec`."""

    job_id: str
    spec: dict
    state: State
    enqueued_at: float
    model: str = ""            # resolved model, filled at claim time for grouping
    leased_at: float = 0.0     # when the daemon claimed it (running state)
    finished_at: float = 0.0
    error: str = ""

    def to_json(self) -> str:
        d = {
            "job_id": self.job_id,
            "state": self.state.value,
            "enqueued_at": self.enqueued_at,
            "model": self.model,
            "leased_at": self.leased_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "spec": self.spec,
        }
        return json.dumps(d, indent=2)

    @staticmethod
    def from_file(path: Path) -> "QueuedJob":
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OllamaqError(f"corrupt queue file {path}: {exc}") from exc
        return QueuedJob(
            job_id=d["job_id"],
            spec=d["spec"],
            state=State(d["state"]),
            enqueued_at=d.get("enqueued_at", 0.0),
            model=d.get("model", ""),
            leased_at=d.get("leased_at", 0.0),
            finished_at=d.get("finished_at", 0.0),
            error=d.get("error", ""),
        )


class JobQueue:
    """Manages queue/<state>/ directories under a root."""

    def __init__(self, root: str | Path = "queue") -> None:
        self.root = Path(root)
        for st in State:
            (self.root / st.value).mkdir(parents=True, exist_ok=True)

    def _path(self, state: State, job_id: str) -> Path:
        return self.root / state.value / f"{job_id}.json"

    def _write_atomic(self, path: Path, job: QueuedJob) -> None:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(job.to_json(), encoding="utf-8")
        os.replace(tmp, path)

    def submit(self, job_id: str, spec: dict) -> QueuedJob:
        """Enqueue a new job into pending/. Raises if the id already exists anywhere."""
        for st in State:
            if self._path(st, job_id).exists():
                raise OllamaqError(f"job id {job_id!r} already exists in {st.value}/")
        job = QueuedJob(job_id=job_id, spec=spec, state=State.pending,
                        enqueued_at=time.time())
        self._write_atomic(self._path(State.pending, job_id), job)
        return job

    def list_state(self, state: State) -> list[QueuedJob]:
        """All jobs in a state, oldest enqueued first."""
        jobs = []
        for p in (self.root / state.value).glob("*.json"):
            try:
                jobs.append(QueuedJob.from_file(p))
            except OllamaqError:
                continue  # skip a corrupt/half-written file rather than crash
        jobs.sort(key=lambda j: j.enqueued_at)
        return jobs

    def find(self, job_id: str) -> Optional[tuple[State, QueuedJob]]:
        for st in State:
            p = self._path(st, job_id)
            if p.exists():
                return st, QueuedJob.from_file(p)
        return None

    def claim_one(self, job_id: str, model: str = "") -> Optional[QueuedJob]:
        """Atomically move a specific pending job to running/ and return it.

        Records `model` (resolved by the caller) and a lease timestamp on the
        envelope. Returns None if the job isn't in pending/ (already claimed by
        another worker, cancelled, etc.). The pending/ source is removed only
        after the running/ file is written, so a crash leaves at most a
        harmless duplicate that reclaim_stale/normal flow tolerates.
        """
        src = self._path(State.pending, job_id)
        if not src.exists():
            return None
        try:
            job = QueuedJob.from_file(src)
        except OllamaqError:
            return None
        job.state = State.running
        job.leased_at = time.time()
        job.model = model
        dst = self._path(State.running, job_id)
        self._write_atomic(dst, job)
        try:
            os.remove(src)
        except FileNotFoundError:
            try:
                os.remove(dst)
            except FileNotFoundError:
                pass
            return None
        return job

    def claim_next(self) -> Optional[QueuedJob]:
        """Claim the oldest pending job (no model preference). Returns None if
        the queue is empty or the claim lost a race."""
        pending = self.list_state(State.pending)
        if not pending:
            return None
        return self.claim_one(pending[0].job_id)

    def finish(self, job: QueuedJob, state: State, error: str = "") -> None:
        """Move a running job to done/ or failed/."""
        assert state in (State.done, State.failed)
        job.state = state
        job.finished_at = time.time()
        job.error = error
        self._write_atomic(self._path(state, job.job_id), job)
        try:
            os.remove(self._path(State.running, job.job_id))
        except FileNotFoundError:
            pass

    def reclaim_stale(self, timeout_s: float = LEASE_TIMEOUT_S) -> list[str]:
        """Move running jobs whose lease is older than timeout_s back to pending/.
        Returns the reclaimed job ids. Call on daemon startup."""
        reclaimed = []
        now = time.time()
        for job in self.list_state(State.running):
            if now - job.leased_at > timeout_s:
                job.state = State.pending
                job.leased_at = 0.0
                self._write_atomic(self._path(State.pending, job.job_id), job)
                try:
                    os.remove(self._path(State.running, job.job_id))
                except FileNotFoundError:
                    pass
                reclaimed.append(job.job_id)
        return reclaimed

    def cancel(self, job_id: str) -> bool:
        """Remove a pending job. Returns True if removed, False if not pending
        (already running/done/failed jobs cannot be cancelled here)."""
        p = self._path(State.pending, job_id)
        try:
            os.remove(p)
            return True
        except FileNotFoundError:
            return False
