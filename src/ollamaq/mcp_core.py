"""Pure logic behind the MCP tools, kept separate from the MCP wiring so it can
be unit-tested without an MCP client. mcp_server.py is a thin adapter over these.

The token-economy contract still holds: `queue_submit` returns only a job id,
`queue_status` returns compact state, and `queue_digest` returns the ≤8 KB digest
file — bulk worker output never flows back through these tools. `ollama_quick` is
the ONE tool that returns model text to Claude, so it hard-caps its output.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

from .client import OllamaClient
from .config import Config
from .errors import OllamaqError
from .jobspec import JobSpec
from .queue import JobQueue, State

# ollama_quick is the only tool whose result enters Claude's context, so bound it.
QUICK_MAX_TOKENS = 2048
QUICK_MAX_CHARS = 8000


async def quick(cfg: Config, prompt: str, model: Optional[str] = None,
                task_tag: Optional[str] = None,
                schema: Optional[dict] = None) -> str:
    """One-shot generation for a small delegation. Output is capped so it can
    never blow up Claude's context. Returns the (possibly truncated) text."""
    resolved = cfg.resolve_model(model, task_tag) if (model or task_tag) \
        else cfg.defaults.model
    options: dict[str, Any] = {"num_predict": QUICK_MAX_TOKENS}
    if cfg.defaults.temperature is not None:
        options["temperature"] = cfg.defaults.temperature
    async with OllamaClient(cfg.ollama) as client:
        res = await client.generate(resolved, prompt, schema=schema, options=options)
    text = res.text
    if len(text) > QUICK_MAX_CHARS:
        text = text[:QUICK_MAX_CHARS] + "\n…(truncated by ollama_quick cap)"
    return text


def submit(queue: JobQueue, cfg: Config, jobspec: dict,
           job_id: Optional[str] = None) -> str:
    """Validate a job spec dict and enqueue it. Returns the job id. Raises
    OllamaqError on an invalid spec (the MCP wrapper converts to a tool error).

    Security: a job spec's `verifier.cmd` (map/gen_test) or a pipeline stage runs
    an arbitrary local subprocess. That's fine from the trusted CLI, but the MCP
    server may be reachable by a less-trusted client, so verifier execution over
    MCP is BLOCKED unless OLLAMAQ_MCP_ALLOW_VERIFIER=1 is set. See MCP.md."""
    try:
        spec = JobSpec.model_validate(jobspec)
    except Exception as exc:  # pydantic ValidationError or other
        raise OllamaqError(f"invalid job spec: {exc}") from exc
    if spec.verifier is not None and os.environ.get("OLLAMAQ_MCP_ALLOW_VERIFIER") != "1":
        raise OllamaqError(
            "job spec has a verifier command, which runs an arbitrary subprocess; "
            "refusing to enqueue it over MCP. Set OLLAMAQ_MCP_ALLOW_VERIFIER=1 to "
            "allow verifier jobs from this server (only if you trust the client)."
        )
    jid = job_id or f"{spec.name}-{int(time.time())}"
    queue.submit(jid, spec.model_dump(mode="json", by_alias=True))
    return jid


def status(queue: JobQueue, job_id: Optional[str] = None) -> dict:
    """State counts, or one job's detail when job_id is given."""
    if job_id:
        found = queue.find(job_id)
        if not found:
            return {"job_id": job_id, "found": False}
        st, job = found
        return {"job_id": job_id, "found": True, "state": st.value,
                "model": job.model, "error": job.error}
    return {st.value: len(queue.list_state(st)) for st in State}


def digest(results_root: Path, job_id: str) -> str:
    """Return a finished job's digest.md text, or a not-found message."""
    p = results_root / job_id / "digest.md"
    if not p.is_file():
        return f"(no digest yet for {job_id}: not finished, or unknown job)"
    return p.read_text(encoding="utf-8")


async def recommend_models(cfg: Config) -> str:
    """GPU-aware model-fit report: which configured models run fully on this
    machine's GPU vs. spill to CPU. Used by the MCP `recommend_models` tool."""
    from .gpu import detect_nvidia_gpu, recommend as build_recommendation

    gpu = detect_nvidia_gpu()
    sizes: dict = {}
    async with OllamaClient(cfg.ollama) as client:
        if await client.ping():
            sizes = await client.model_sizes()
    return build_recommendation(cfg, sizes, gpu, bench=None).format()
