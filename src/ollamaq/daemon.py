"""The queue daemon: drains pending jobs continuously, one at a time.

Single-writer model — run ONE daemon per machine (the RTX 5080 holds one large
model at a time; OLLAMA_MAX_LOADED_MODELS=1). It is not designed for multiple
concurrent daemons against the same queue.

Scheduling: to avoid reloading models (the dominant cost on a single-GPU box),
the daemon prefers the next pending job whose resolved model matches the one it
just ran. Within a model, oldest-enqueued first.

The daemon runs synchronous `run_job` per job (the same path as `ollamaq run`),
writes results + digest, and moves the job to done/ or failed/ based on the
success threshold. It sleeps `poll_interval_s` when the queue is empty.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from .config import Config
from .digest import write_results
from .errors import OllamaqError
from .execute import execute_spec
from .jobspec import JobSpec
from .pricing import Pricing
from .queue import JobQueue, QueuedJob, State


def _resolve_model_safe(cfg: Config, spec: JobSpec) -> str:
    try:
        return cfg.resolve_model(spec.model, spec.task_tag)
    except OllamaqError:
        return ""


def pick_next(queue: JobQueue, cfg: Config, prefer_model: str = "") -> Optional[QueuedJob]:
    """Claim the next job, preferring one that matches prefer_model to avoid a
    model reload. Falls back to plain oldest-first. Returns None if queue empty.

    Note: we peek at pending to choose, but the actual claim is still the atomic
    JobQueue.claim_next-style move done here for the chosen id.
    """
    pending = queue.list_state(State.pending)
    if not pending:
        return None
    chosen = None
    chosen_model = ""
    if prefer_model:
        for job in pending:  # oldest-first already
            if _resolve_model_safe(cfg, JobSpec.model_validate(job.spec)) == prefer_model:
                chosen, chosen_model = job, prefer_model
                break
    if chosen is None:
        chosen = pending[0]
        chosen_model = _resolve_model_safe(cfg, JobSpec.model_validate(chosen.spec))
    return queue.claim_one(chosen.job_id, model=chosen_model)


def process_job(cfg: Config, queue: JobQueue, job: QueuedJob, results_root: Path,
                pricing: Optional[Pricing], min_success: float) -> None:
    """Run a claimed job to completion and move it to done/ or failed/."""
    try:
        spec = JobSpec.model_validate(job.spec)
        result = execute_spec(cfg, spec)
        write_results(job.job_id, spec, result, results_root, pricing)
        # gen_test is a search (no pass/fail threshold); map is gated on success.
        if not result.iterations and result.success_rate < min_success:
            queue.finish(job, State.failed,
                         error=f"success {result.success_rate:.1%} < {min_success:.0%}")
        else:
            queue.finish(job, State.done)
    except OllamaqError as exc:
        queue.finish(job, State.failed, error=str(exc))


def run_daemon(
    cfg: Config,
    queue: JobQueue,
    results_root: Path = Path("results"),
    pricing: Optional[Pricing] = None,
    min_success: float = 0.95,
    poll_interval_s: float = 5.0,
    log: Callable[[str], None] = print,
    max_jobs: Optional[int] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> int:
    """Drain the queue until stopped. Returns the number of jobs processed.

    - max_jobs: stop after processing this many (used by tests; None = forever).
    - should_stop: called each loop; return True to exit (used by tests/signals).
    """
    reclaimed = queue.reclaim_stale()
    if reclaimed:
        log(f"reclaimed {len(reclaimed)} stale job(s): {', '.join(reclaimed)}")

    processed = 0
    last_model = ""
    log("daemon started; draining queue (Ctrl-C to stop)")
    while True:
        if should_stop is not None and should_stop():
            break
        if max_jobs is not None and processed >= max_jobs:
            break
        job = pick_next(queue, cfg, prefer_model=last_model)
        if job is None:
            if should_stop is not None:  # tests drive with should_stop; don't sleep
                break
            time.sleep(poll_interval_s)
            continue
        log(f"running {job.job_id} ...")
        process_job(cfg, queue, job, results_root, pricing, min_success)
        state, refreshed = queue.find(job.job_id) or (None, None)
        last_model = refreshed.model if refreshed else ""
        log(f"finished {job.job_id}: {state.value if state else '?'}")
        processed += 1
    log(f"daemon stopped; processed {processed} job(s)")
    return processed
