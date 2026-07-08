"""embed mode — turn each input item's text into a vector via a local embedding
model (Ollama /api/embed). The classic bulk data-engineering task: build a vector
store for RAG, dedup, clustering, or semantic search without paying a cloud
embedding API per document.

Simpler than the generation engines: no template, schema, repair, or verifier —
the chunk text is embedded as-is. Transport errors get the same bounded transient
retry as generation. Each ItemResult.value is `{"dim": N, "embedding": [...]}`;
the vectors live on disk in merged.jsonl, never in Claude's context (the digest
reports counts and dimensions, not the vectors).
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from .client import OllamaClient
from .config import Config
from .errors import OllamaqError
from .jobspec import JobSpec
from .worker import ItemResult, RunResult, TRANSIENT_BACKOFF_BASE


async def _embed_one(client: OllamaClient, model: str, index: int, source: str,
                     text: str, transient_retries: int) -> ItemResult:
    delay = TRANSIENT_BACKOFF_BASE
    for attempt in range(transient_retries + 1):
        try:
            res = await client.embed(model, text)
        except OllamaqError as exc:
            if attempt >= transient_retries:
                return ItemResult(index=index, source=source, ok=False,
                                  error=f"embed failed: {exc}", attempts=attempt + 1)
            if delay > 0:
                await asyncio.sleep(delay)
            delay *= 2
            continue
        if not res.embedding:
            return ItemResult(index=index, source=source, ok=False,
                              error="embed returned an empty vector", attempts=attempt + 1)
        return ItemResult(index=index, source=source, ok=True,
                          value={"dim": res.dim, "embedding": res.embedding},
                          attempts=attempt + 1, prompt_tokens=res.prompt_tokens)
    return ItemResult(index=index, source=source, ok=False,
                      error="embed exhausted retries", attempts=transient_retries + 1)


async def run_embed(config: Config, spec: JobSpec, items: list[tuple]) -> RunResult:
    """Embed every item's text, bounded by the model's max_parallel. Items are
    (index, source, text[, images]) — images are ignored for embedding."""
    model = config.resolve_model(spec.model, spec.task_tag)
    max_parallel = config.models[model].max_parallel
    sem = asyncio.Semaphore(max_parallel)
    tr = spec.retry.transient_retries
    start = time.perf_counter()

    async with OllamaClient(config.ollama) as client:
        async def _guarded(item: tuple) -> ItemResult:
            async with sem:
                return await _embed_one(client, model, item[0], item[1], item[2], tr)

        results = await asyncio.gather(*[_guarded(it) for it in items])

    results.sort(key=lambda r: r.index)
    return RunResult(model=model, results=results,
                     wall_seconds=round(time.perf_counter() - start, 3),
                     prompt_tokens_total=sum(r.prompt_tokens for r in results),
                     completion_tokens_total=0, eval_ns_total=0)
