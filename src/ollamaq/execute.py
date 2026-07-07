"""Single entry point that runs a validated JobSpec to a RunResult, dispatching
on mode. Shared by `ollamaq run` (sync) and the daemon so mode handling lives in
exactly one place.

Async: returns a coroutine-free RunResult by running the async engines under
asyncio.run. Callers are synchronous (CLI command, daemon loop).
"""

from __future__ import annotations

import asyncio

from .chunker import chunk_input
from .config import Config
from .errors import JobSpecError
from .gentest import run_gentest
from .jobspec import JobSpec, Mode
from .pipeline import run_pipeline
from .worker import RunResult, run_job


def execute_spec(cfg: Config, spec: JobSpec) -> RunResult:
    """Run the job to completion, dispatching on mode. Raises JobSpecError for
    empty input."""
    if spec.mode == Mode.map:
        work = chunk_input(spec.input)
        if not work:
            raise JobSpecError("input produced 0 work items")
        items = [(w.index, w.source, w.text, w.images) for w in work]
        return asyncio.run(run_job(cfg, spec, items))
    if spec.mode == Mode.gen_test:
        return asyncio.run(run_gentest(cfg, spec))
    return asyncio.run(run_pipeline(cfg, spec))
