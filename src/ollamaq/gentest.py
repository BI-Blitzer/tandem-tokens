"""gen_test mode — the continuous generate-and-test loop.

Each iteration: generate N candidates locally, score every candidate with the
user's verifier subprocess, keep the global top-K, feed the survivors back as
few-shot exemplars, and repeat until `stop_when` fires or `max_iterations` is
reached. Only the final scored leaderboard reaches Claude (via the digest).

The prompt template's {{chunk}} is the fixed task/seed (first work item of the
input). An optional {{exemplars}} placeholder is filled with the current top
survivors so the model can learn from what scored well; if the template has no
{{exemplars}} placeholder, feedback is simply not injected.

Verifier calls are blocking subprocesses; they run in a thread pool so they
don't stall the async generation loop.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

from .chunker import chunk_input
from .client import OllamaClient
from .config import Config
from .enforce import process_attempt
from .jobspec import JobSpec, LoopSpec
from .verify import Verdict, run_verifier
from .worker import ItemResult, RunResult, render_prompt

# gen_test wants diverse candidates; if the job sets no temperature, use this.
DEFAULT_GENTEST_TEMPERATURE = 0.8
_EXEMPLAR_PLACEHOLDER = "{{exemplars}}"


@dataclass
class Candidate:
    value: Any
    score: float
    passed: bool
    notes: str
    iteration: int


def _format_exemplars(top: list[Candidate]) -> str:
    """Render survivors as a compact scored block for the next prompt."""
    if not top:
        return "(none yet)"
    lines = []
    for c in top:
        payload = json.dumps(c.value, ensure_ascii=False)
        if len(payload) > 300:
            payload = payload[:300] + "…"
        lines.append(f"- score {c.score:.3f}: {payload}")
    return "\n".join(lines)


def _merge_leaderboard(board: list[Candidate], new: list[Candidate],
                       keep_top: int) -> list[Candidate]:
    board = board + new
    board.sort(key=lambda c: c.score, reverse=True)
    return board[:keep_top]


async def run_gentest(config: Config, spec: JobSpec) -> RunResult:
    """Run the gen_test loop and return a RunResult whose `results` are the
    final leaderboard (highest score first) and whose `iterations` traces each
    round. Token totals cover every candidate generated."""
    assert spec.verifier is not None  # guaranteed by JobSpec validator
    model = config.resolve_model(spec.model, spec.task_tag)
    max_parallel = config.models[model].max_parallel
    loop: LoopSpec = spec.loop or LoopSpec()
    keep_top = spec.verifier.keep_top
    schema = spec.prompt.schema_

    options = spec.params.to_options()
    options.setdefault("temperature", DEFAULT_GENTEST_TEMPERATURE)

    work = chunk_input(spec.input)
    if not work:
        raise ValueError("gen_test input produced 0 work items")
    seed = work[0].text

    board: list[Candidate] = []
    best_history: list[float] = []
    iterations_meta: list[dict] = []
    global_best = float("-inf")
    prompt_tokens = completion_tokens = eval_ns = 0

    sem = asyncio.Semaphore(max_parallel)
    start = time.perf_counter()

    from .stopcond import should_stop  # local import: stopcond has no heavy deps

    async with OllamaClient(config.ollama) as client:

        async def _one_generation(prompt: str):
            async with sem:
                return await client.generate(model, prompt, system=spec.prompt.system,
                                             schema=schema, options=options)

        for iteration in range(1, loop.max_iterations + 1):
            exemplars = _format_exemplars(board) if loop.feedback == "top_k" else ""
            prompt = render_prompt(spec.prompt.template, seed)
            if _EXEMPLAR_PLACEHOLDER in prompt:
                prompt = prompt.replace(_EXEMPLAR_PLACEHOLDER, exemplars)

            gens = await asyncio.gather(
                *[_one_generation(prompt) for _ in range(loop.candidates)]
            )

            # Parse candidates, then score them off the event loop (blocking subprocess).
            parsed: list[Any] = []
            for g in gens:
                prompt_tokens += g.prompt_tokens
                completion_tokens += g.completion_tokens
                eval_ns += g.eval_duration_ns
                outcome = process_attempt(g.text, schema)
                parsed.append(outcome.value if outcome.ok else {"text": g.text})

            verdicts: list[Verdict] = await asyncio.gather(
                *[asyncio.to_thread(run_verifier, spec.verifier.cmd, v,
                                    spec.verifier.timeout_s)
                  for v in parsed]
            )

            new = [
                Candidate(value=v, score=verd.score, passed=verd.passed,
                          notes=verd.notes, iteration=iteration)
                for v, verd in zip(parsed, verdicts)
            ]
            board = _merge_leaderboard(board, new, keep_top)
            iter_best = max((c.score for c in new), default=float("-inf"))
            global_best = max(global_best, iter_best)
            best_history.append(global_best)
            iterations_meta.append({
                "iteration": iteration,
                "candidates": loop.candidates,
                "best_score": round(iter_best, 4) if iter_best > float("-inf") else None,
                "global_best": round(global_best, 4),
            })

            if should_stop(loop.stop_when, global_best, best_history):
                break

    results = [
        ItemResult(index=rank, source=f"iter{c.iteration}", ok=c.passed,
                   value=c.value, score=c.score, verdict_notes=c.notes)
        for rank, c in enumerate(board)
    ]
    return RunResult(
        model=model, results=results,
        wall_seconds=round(time.perf_counter() - start, 3),
        iterations=iterations_meta,
        prompt_tokens_total=prompt_tokens,
        completion_tokens_total=completion_tokens,
        eval_ns_total=eval_ns,
    )
