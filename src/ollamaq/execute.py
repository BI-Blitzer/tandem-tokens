"""Single entry point that runs a validated JobSpec to a RunResult, dispatching
on mode. Shared by `ollamaq run` (sync) and the daemon so mode handling lives in
exactly one place.

Async: returns a coroutine-free RunResult by running the async engines under
asyncio.run. Callers are synchronous (CLI command, daemon loop).

Also owns GPU energy metering (nvidia-smi samples around the whole job) so every
mode contributes Wh to the scoreboard when sensors are available.
"""

from __future__ import annotations

import asyncio

from .chunker import chunk_input
from .config import Config
from .embed import run_embed
from .energy import EnergyMeter
from .errors import JobSpecError
from .gentest import run_gentest
from .jobspec import JobSpec, Mode
from .pipeline import run_pipeline
from .worker import RunResult, run_job


def _attach_energy(result: RunResult, meter: EnergyMeter) -> RunResult:
    report = meter.stop()
    if report.available:
        result.energy_wh = round(report.energy_wh, 4)
        result.avg_power_w = round(report.avg_power_w, 2)
        result.peak_power_w = round(report.peak_power_w, 2)
    return result


def execute_spec(cfg: Config, spec: JobSpec) -> RunResult:
    """Run the job to completion, dispatching on mode. Raises JobSpecError for
    empty input."""
    meter = EnergyMeter().start()
    if spec.mode in (Mode.map, Mode.embed):
        work = chunk_input(spec.input)
        if not work:
            raise JobSpecError("input produced 0 work items")
        items = [(w.index, w.source, w.text, w.images) for w in work]
        if spec.mode == Mode.embed:
            result = asyncio.run(run_embed(cfg, spec, items))
        else:
            result = asyncio.run(run_job(cfg, spec, items))
        return _attach_energy(result, meter)
    if spec.mode == Mode.gen_test:
        return _attach_energy(asyncio.run(run_gentest(cfg, spec)), meter)
    return _attach_energy(asyncio.run(run_pipeline(cfg, spec)), meter)
