"""Thin MCP (stdio) server exposing ollamaq to Claude Code and other clients.

Tools:
  ollama_quick / queue_submit / queue_status / queue_digest / recommend_models
  hybrid_score / hybrid_energy / hybrid_policy  (shared with the Grok facade)

For Grok-oriented tool aliases (hybrid_offload, …) use `python -m ollamaq.mcp_grok`
or the `ollamaq-mcp-grok` console script. Logic lives in mcp_core (unit-tested).
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

mcp = FastMCP("ollamaq")


def _cfg():
    return mcp_core.load_cfg_for_mcp()


@mcp.tool()
async def ollama_quick(prompt: str, model: Optional[str] = None,
                       task_tag: Optional[str] = None,
                       schema: Optional[dict] = None) -> str:
    """Run a single small prompt on a local Ollama model and return its text.
    Output is capped (~2k tokens) — use this for quick one-offs, not bulk work
    (enqueue bulk work with queue_submit instead)."""
    try:
        return await mcp_core.quick(_cfg(), prompt, model=model,
                                    task_tag=task_tag, schema=schema)
    except OllamaqError as exc:
        return f"error: {exc}"


@mcp.tool()
def queue_submit(jobspec: dict, job_id: Optional[str] = None) -> str:
    """Enqueue a job spec (see PLAN.md section 3) for the ollamaq daemon to run.
    Returns the job id. The daemon must be running to process it."""
    try:
        jid = mcp_core.submit(JobQueue(_QUEUE_ROOT), _cfg(), jobspec, job_id)
        return f"submitted {jid}"
    except OllamaqError as exc:
        return f"error: {exc}"


@mcp.tool()
def queue_status(job_id: Optional[str] = None) -> str:
    """Queue state — counts per state, or one job's detail when job_id is given."""
    return json.dumps(mcp_core.status(JobQueue(_QUEUE_ROOT), job_id))


@mcp.tool()
def queue_digest(job_id: str) -> str:
    """Return a finished job's digest.md (the compact, premium-agent summary)."""
    return mcp_core.digest(_RESULTS_ROOT, job_id)


@mcp.tool()
async def recommend_models() -> str:
    """Recommend which local models fit this machine's GPU (vs. spill to CPU),
    from nvidia-smi VRAM + installed model sizes. Use before choosing a model for
    a job on an unfamiliar machine."""
    try:
        return await mcp_core.recommend_models(_cfg())
    except OllamaqError as exc:
        return f"error: {exc}"


@mcp.tool()
def hybrid_score() -> str:
    """Cumulative savings scoreboard: local tokens, estimated cloud $ avoided,
    and GPU energy (Wh) when nvidia-smi metering was available."""
    return mcp_core.scoreboard(_RESULTS_ROOT)


@mcp.tool()
def hybrid_energy() -> str:
    """Live GPU power sample via nvidia-smi (watts / limit / temp)."""
    return mcp_core.energy_now()


@mcp.tool()
def hybrid_policy() -> str:
    """When to offload bulk work to local Ollama vs. keep in premium context."""
    return mcp_core.hybrid_policy_text()


def main() -> None:  # pragma: no cover - entry point
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
