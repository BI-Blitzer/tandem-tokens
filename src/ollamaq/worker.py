"""Async fan-out: run one job's WorkItems through a local model, honoring the
per-model concurrency cap and the schema repair-retry policy.

This is the orchestration layer that ties client + chunker + enforce together.
It returns ItemResults in input order; it does NOT write files or build digests
(digest.py owns that) and never returns bulk output to a caller's context beyond
the ItemResult list, which run() persists to disk.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .cache import Cache, make_key as cache_make_key
from .client import GenerateResult, OllamaClient
from .config import Config
from .enforce import build_repair_prompt, process_attempt
from .errors import OllamaqError
from .jobspec import JobSpec, OnSchemaFail


@dataclass
class ItemResult:
    index: int
    source: str
    ok: bool
    value: Any = None
    error: str = ""
    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    eval_duration_ns: int = 0
    score: Optional[float] = None   # set by gen_test (verifier score); None for map
    verdict_notes: str = ""         # verifier notes, gen_test only
    cached: bool = False            # served from the dedupe cache (0 new tokens)
    escalated: bool = False         # recovered by retry.escalate_to on a stronger model


@dataclass
class RunResult:
    model: str
    results: list[ItemResult] = field(default_factory=list)
    wall_seconds: float = 0.0
    # Per-iteration trace, gen_test only (empty otherwise). Each dict carries
    # {iteration, candidates, best_score, global_best}.
    iterations: list[dict] = field(default_factory=list)
    # Per-stage trace, pipeline only (empty otherwise). Each dict carries
    # {stage, model, in, ok, failed, continued}.
    stages: list[dict] = field(default_factory=list)
    # Explicit token/latency totals. gen_test sets these because its cost spans
    # ALL candidates generated, not just the survivors kept in `results`. map
    # leaves them None and the properties fall back to summing `results`.
    prompt_tokens_total: Optional[int] = None
    completion_tokens_total: Optional[int] = None
    eval_ns_total: Optional[int] = None

    @property
    def ok_count(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.ok)

    @property
    def success_rate(self) -> float:
        return self.ok_count / len(self.results) if self.results else 0.0

    @property
    def cached_count(self) -> int:
        return sum(1 for r in self.results if r.cached)

    @property
    def escalated_count(self) -> int:
        return sum(1 for r in self.results if r.escalated)

    @property
    def total_prompt_tokens(self) -> int:
        if self.prompt_tokens_total is not None:
            return self.prompt_tokens_total
        return sum(r.prompt_tokens for r in self.results)

    @property
    def total_completion_tokens(self) -> int:
        if self.completion_tokens_total is not None:
            return self.completion_tokens_total
        return sum(r.completion_tokens for r in self.results)

    @property
    def tokens_per_second(self) -> float:
        ns = (self.eval_ns_total if self.eval_ns_total is not None
              else sum(r.eval_duration_ns for r in self.results))
        return self.total_completion_tokens / (ns / 1e9) if ns > 0 else 0.0


def render_prompt(template: str, chunk_text: str) -> str:
    return template.replace("{{chunk}}", chunk_text)


# Base backoff (seconds) between transient-error retries; grows exponentially.
# A module constant so tests can zero it out (see conftest).
TRANSIENT_BACKOFF_BASE = 0.5


async def _generate_retrying(client: OllamaClient, model: str, prompt: str, *,
                             system, schema, options, images,
                             transient_retries: int) -> GenerateResult:
    """client.generate with bounded retry on TRANSPORT errors (connection resets,
    HTTP 5xx) — common while a model cold-loads. Schema failures are NOT retried
    here; they flow up to the repair-retry loop. Re-raises after the last retry."""
    delay = TRANSIENT_BACKOFF_BASE
    for attempt in range(transient_retries + 1):
        try:
            return await client.generate(model, prompt, system=system, schema=schema,
                                         options=options, images=images)
        except OllamaqError:
            if attempt >= transient_retries:
                raise
            if delay > 0:
                await asyncio.sleep(delay)
            delay *= 2


async def _run_item(
    client: OllamaClient,
    model: str,
    spec: JobSpec,
    index: int,
    source: str,
    chunk_text: str,
    cache: "Optional[Cache]" = None,
    escalate_model: "Optional[str]" = None,
    escalate_num_ctx: "Optional[int]" = None,
    images: "Optional[list[str]]" = None,
) -> ItemResult:
    """Run one work item through the model with the repair-retry loop, then — if
    `escalate_model` is set — one final attempt on that stronger model before
    giving up (only for schema failures; transport errors short-circuit earlier).
    `images` (base64) are sent to the model for vision jobs."""
    schema = spec.prompt.schema_
    options = spec.params.to_options()
    initial_prompt = render_prompt(spec.prompt.template, chunk_text)
    prompt = initial_prompt
    last_error = ""
    attempt = 0
    # Token/latency totals accumulate across retries: every attempt genuinely
    # consumed local compute, and would have cost cloud tokens too.
    prompt_tokens = completion_tokens = eval_ns = 0

    # Dedupe cache: keyed on the ORIGINAL request. A hit skips generation and
    # re-validation entirely (schema is part of the key, so no stale hits).
    cache_key = None
    if cache is not None:
        cache_key = cache_make_key(model, spec.prompt.system, initial_prompt, schema, options)
        hit = cache.get(cache_key)
        if hit is not None:
            return ItemResult(index=index, source=source, ok=True, value=hit,
                              cached=True, attempts=0)

    for attempt in range(1, spec.retry.max_attempts + 1):
        try:
            gen = await _generate_retrying(
                client, model, prompt, system=spec.prompt.system, schema=schema,
                options=options, images=images,
                transient_retries=spec.retry.transient_retries,
            )
        except OllamaqError as exc:
            # transport/HTTP error persisted through all transient retries.
            return ItemResult(index=index, source=source, ok=False,
                              error=f"generate failed: {exc}", attempts=attempt,
                              prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                              eval_duration_ns=eval_ns)
        prompt_tokens += gen.prompt_tokens
        completion_tokens += gen.completion_tokens
        eval_ns += gen.eval_duration_ns
        outcome = process_attempt(gen.text, schema)
        if outcome.ok:
            if cache is not None and cache_key is not None:
                cache.put(cache_key, outcome.value)  # cache validated successes only
            return ItemResult(index=index, source=source, ok=True, value=outcome.value,
                              attempts=attempt, prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens, eval_duration_ns=eval_ns)
        last_error = outcome.error

        # decide whether to keep going
        if spec.retry.on_schema_fail == OnSchemaFail.fail:
            break
        if attempt < spec.retry.max_attempts:
            if spec.retry.on_schema_fail == OnSchemaFail.repair_then_retry and schema:
                prompt = build_repair_prompt(
                    spec.prompt.template, chunk_text, gen.text, last_error, schema
                )
            # OnSchemaFail.retry re-sends the original prompt unchanged

    # Escalation: the primary model couldn't satisfy the schema. Try once on a
    # stronger local model (with its own context size) before surfacing to Claude.
    if escalate_model is not None:
        esc_options = dict(options)
        if escalate_num_ctx is not None:
            esc_options["num_ctx"] = escalate_num_ctx
        try:
            gen = await _generate_retrying(
                client, escalate_model, initial_prompt, system=spec.prompt.system,
                schema=schema, options=esc_options, images=images,
                transient_retries=spec.retry.transient_retries)
            prompt_tokens += gen.prompt_tokens
            completion_tokens += gen.completion_tokens
            eval_ns += gen.eval_duration_ns
            outcome = process_attempt(gen.text, schema)
            if outcome.ok:
                if cache is not None and cache_key is not None:
                    cache.put(cache_key, outcome.value)
                return ItemResult(index=index, source=source, ok=True,
                                  value=outcome.value, escalated=True,
                                  attempts=attempt + 1, prompt_tokens=prompt_tokens,
                                  completion_tokens=completion_tokens,
                                  eval_duration_ns=eval_ns)
            last_error = (f"{last_error}; escalation to {escalate_model} also failed: "
                          f"{outcome.error}")
        except OllamaqError as exc:
            last_error = f"{last_error}; escalation to {escalate_model} errored: {exc}"

    return ItemResult(
        index=index, source=source, ok=False, error=last_error or "schema not satisfied",
        attempts=attempt, prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens, eval_duration_ns=eval_ns,
    )


async def run_job(
    config: Config,
    spec: JobSpec,
    items: list[tuple],
) -> RunResult:
    """Fan items out across the resolved model, bounded by that model's
    max_parallel from models.yaml. Each item is (index, source, chunk_text) or
    (index, source, chunk_text, images) for vision jobs. Results come back in
    input order.
    """
    model = config.resolve_model(spec.model, spec.task_tag)
    max_parallel = config.models[model].max_parallel
    sem = asyncio.Semaphore(max_parallel)
    cache = None
    if spec.cache:
        try:
            cap = int(os.environ.get("OLLAMAQ_CACHE_MAX_ENTRIES", "10000"))
        except ValueError:
            cap = 10000
        cache = Cache(os.environ.get("OLLAMAQ_CACHE", "cache"), max_entries=cap)

    # Resolve the escalation target up front (fail fast if undefined/disabled).
    # A no-op when it equals the primary model.
    escalate_model: Optional[str] = None
    escalate_num_ctx: Optional[int] = None
    if spec.retry.escalate_to:
        esc = config.resolve_model(spec.retry.escalate_to, None)
        if esc != model:
            escalate_model = esc
            escalate_num_ctx = config.models[esc].num_ctx

    start = time.perf_counter()

    async with OllamaClient(config.ollama) as client:
        async def _guarded(item: tuple) -> ItemResult:
            # items are (index, source, text) or (index, source, text, images).
            idx, source, text = item[0], item[1], item[2]
            imgs = list(item[3]) if len(item) > 3 and item[3] else None
            async with sem:
                return await _run_item(client, model, spec, idx, source, text, cache,
                                       escalate_model, escalate_num_ctx, imgs)

        tasks = [_guarded(it) for it in items]
        results = await asyncio.gather(*tasks)

    results.sort(key=lambda r: r.index)
    return RunResult(model=model, results=results,
                     wall_seconds=round(time.perf_counter() - start, 3))
