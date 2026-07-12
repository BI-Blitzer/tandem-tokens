"""Grok-oriented MCP server facade for ollamaq.

Same underlying logic as mcp_server (mcp_core), but:
  - Server name `ollamaq-grok` so Grok client configs can pin it distinctly.
  - Tools registered under hybrid_* aliases (and the standard names) with
    descriptions written for Grok as the premium orchestrator.
  - Extra tools: hybrid_score, hybrid_energy, hybrid_policy.

Launch (Grok / any stdio MCP client):
  python -m ollamaq.mcp_grok
  # or: ollamaq-mcp-grok

Config snippets live under clients/ — see MCP.md.
"""

from __future__ import annotations

import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import mcp_core
from .errors import OllamaqError
from .queue import JobQueue

_HOME_PATH = mcp_core.resolve_home()
_QUEUE_ROOT = mcp_core.resolve_root("OLLAMAQ_QUEUE", "queue", _HOME_PATH)
_RESULTS_ROOT = mcp_core.resolve_root("OLLAMAQ_RESULTS", "results", _HOME_PATH)

mcp = FastMCP("ollamaq-grok")


def _cfg():
    return mcp_core.load_cfg_for_mcp()


# ---- core tools (standard names — any client) --------------------------------

@mcp.tool()
async def ollama_quick(prompt: str, model: Optional[str] = None,
                       task_tag: Optional[str] = None,
                       schema: Optional[dict] = None) -> str:
    """One-shot local Ollama generation (output capped ~2k tokens). Prefer
    queue_submit / hybrid_offload for bulk work so results stay on disk."""
    try:
        return await mcp_core.quick(_cfg(), prompt, model=model,
                                    task_tag=task_tag, schema=schema)
    except OllamaqError as exc:
        return f"error: {exc}"


@mcp.tool()
def queue_submit(jobspec: dict, job_id: Optional[str] = None) -> str:
    """Enqueue a validated ollamaq job for the local daemon. Returns job id.
    Bulk map/gen_test/pipeline work must go here — never pull bulk output into
    Grok context."""
    try:
        jid = mcp_core.submit(JobQueue(_QUEUE_ROOT), _cfg(), jobspec, job_id)
        return f"submitted {jid}"
    except OllamaqError as exc:
        return f"error: {exc}"


@mcp.tool()
def queue_status(job_id: Optional[str] = None) -> str:
    """Queue counts, or one job's state when job_id is set."""
    return json.dumps(mcp_core.status(JobQueue(_QUEUE_ROOT), job_id))


@mcp.tool()
def queue_digest(job_id: str) -> str:
    """Finished job digest.md only (≤8 KB). Never request merged.jsonl via MCP."""
    return mcp_core.digest(_RESULTS_ROOT, job_id)


@mcp.tool()
async def recommend_models() -> str:
    """GPU-aware local model fit report (nvidia-smi VRAM + Ollama sizes)."""
    try:
        return await mcp_core.recommend_models(_cfg())
    except OllamaqError as exc:
        return f"error: {exc}"


# ---- hybrid_* aliases (Grok-as-orchestrator naming) --------------------------

@mcp.tool()
async def hybrid_quick(prompt: str, model: Optional[str] = None,
                       task_tag: Optional[str] = None,
                       schema: Optional[dict] = None) -> str:
    """Alias of ollama_quick — small local delegation under the hybrid workflow."""
    return await ollama_quick(prompt, model=model, task_tag=task_tag, schema=schema)


@mcp.tool()
def hybrid_offload(jobspec: dict, job_id: Optional[str] = None) -> str:
    """Alias of queue_submit — offload bulk map/gen_test/pipeline to local GPU."""
    return queue_submit(jobspec, job_id)


@mcp.tool()
def hybrid_status(job_id: Optional[str] = None) -> str:
    """Alias of queue_status."""
    return queue_status(job_id)


@mcp.tool()
def hybrid_digest(job_id: str) -> str:
    """Alias of queue_digest — harvest step 1 after a job is done."""
    return queue_digest(job_id)


@mcp.tool()
async def hybrid_recommend() -> str:
    """Alias of recommend_models."""
    return await recommend_models()


@mcp.tool()
def hybrid_score() -> str:
    """Cumulative savings scoreboard: local tokens, cloud $ avoided, GPU energy
    (Wh) when jobs were metered via nvidia-smi."""
    return mcp_core.scoreboard(_RESULTS_ROOT)


@mcp.tool()
def hybrid_energy() -> str:
    """Live GPU power sample (nvidia-smi power.draw). Use before/after bulk runs
    to sanity-check energy metering; per-job Wh lands in the scoreboard."""
    return mcp_core.energy_now()


@mcp.tool()
def hybrid_policy() -> str:
    """When to offload to local Ollama vs. keep in Grok premium context."""
    return mcp_core.hybrid_policy_text()


def main() -> None:  # pragma: no cover
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
