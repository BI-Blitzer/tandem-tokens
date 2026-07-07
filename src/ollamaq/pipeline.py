"""pipeline mode — chained stages, each a map over the surviving item set.

Stage k runs as a normal map job over the currently-active items. Its output
value becomes stage k+1's {{chunk}} (JSON-encoded). An optional per-stage filter
decides which items continue: items whose output MATCHES the filter proceed to
the next stage; non-matching items are finalized right there with their current
value. Items that fail a stage (schema unrecoverable) are finalized too.

Canonical example (PLAN.md section 3): stage 1 (gemma2:9b) classifies each row
{dirty: bool}; filter keeps dirty==true; stage 2 (qwen2.5-14b) cleans only those.
The final result carries every input item at whatever stage it stopped.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .chunker import chunk_input
from .config import Config
from .jobspec import FilterOp, JobSpec, PipelineStage, StageFilter
from .worker import ItemResult, RunResult, run_job


def _get_path(value: Any, path: str) -> Any:
    cur = value
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def matches(f: StageFilter, value: Any) -> bool:
    """True if `value` matches the filter (i.e. the item should CONTINUE)."""
    v = _get_path(value, f.field)
    if f.op == FilterOp.truthy:
        return bool(v)
    if f.op == FilterOp.falsy:
        return not bool(v)
    if f.op == FilterOp.eq:
        return v == f.value
    if f.op == FilterOp.ne:
        return v != f.value
    try:
        return {
            FilterOp.gt: v > f.value,
            FilterOp.ge: v >= f.value,
            FilterOp.lt: v < f.value,
            FilterOp.le: v <= f.value,
        }[f.op]
    except TypeError:
        return False  # non-comparable (e.g. None vs number) -> does not match


def _stage_spec(parent: JobSpec, stage: PipelineStage) -> JobSpec:
    """A single-stage map JobSpec that run_job can execute. Reuses the parent's
    input only to satisfy validation — run_job reads items, not input."""
    return JobSpec.model_validate({
        "name": f"{parent.name}:{stage.name}",
        "mode": "map",
        "model": stage.model,
        "task_tag": stage.task_tag,
        "input": parent.input.model_dump(mode="json"),
        "prompt": stage.prompt.model_dump(mode="json", by_alias=True),
        "params": stage.params.model_dump(mode="json"),
        "retry": stage.retry.model_dump(mode="json"),
    })


async def run_pipeline(config: Config, spec: JobSpec) -> RunResult:
    """Run all stages in order; return a RunResult whose `results` are every
    input item at its final stage, plus a per-stage trace in `stages`."""
    assert spec.stages  # guaranteed by JobSpec validator
    stages = spec.stages

    work = chunk_input(spec.input)
    if not work:
        raise ValueError("pipeline input produced 0 work items")
    active = [(w.index, w.source, w.text) for w in work]

    current: dict[int, ItemResult] = {}
    stage_meta: list[dict] = []
    prompt_tokens = completion_tokens = eval_ns = 0
    start = time.perf_counter()

    for i, stage in enumerate(stages):
        is_last = i == len(stages) - 1
        sub = _stage_spec(spec, stage)
        res = await run_job(config, sub, active)
        prompt_tokens += res.total_prompt_tokens
        completion_tokens += res.total_completion_tokens
        eval_ns += sum(r.eval_duration_ns for r in res.results)

        next_active: list[tuple[int, str, str]] = []
        continued = 0
        for r in res.results:
            current[r.index] = r
            proceed = r.ok and (stage.filter is None or matches(stage.filter, r.value))
            if proceed and not is_last:
                continued += 1
                next_active.append(
                    (r.index, f"{r.source}>{stage.name}",
                     json.dumps(r.value, ensure_ascii=False))
                )
        stage_meta.append({
            "stage": stage.name,
            "model": res.model,
            "in": len(active),
            "ok": res.ok_count,
            "failed": res.fail_count,
            "continued": continued,
        })
        active = next_active
        if not active:
            break  # nothing left to feed the remaining stages

    results = [current[i] for i in sorted(current)]
    return RunResult(
        model=" -> ".join(m["model"] for m in stage_meta),
        results=results,
        wall_seconds=round(time.perf_counter() - start, 3),
        stages=stage_meta,
        prompt_tokens_total=prompt_tokens,
        completion_tokens_total=completion_tokens,
        eval_ns_total=eval_ns,
    )
