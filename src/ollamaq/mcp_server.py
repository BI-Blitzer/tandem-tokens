"""Thin MCP (stdio) server exposing ollamaq to Claude Code.

Tools:
  ollama_quick(prompt, model?, task_tag?, schema?)  one-shot local generation,
      output hard-capped so it never floods Claude's context.
  queue_submit(jobspec, job_id?)   enqueue a job for the daemon; returns job id.
  queue_status(job_id?)            state counts, or one job's detail.
  queue_digest(job_id)             the finished job's ≤8 KB digest.md.

Config, queue root, and results root are resolved from the process CWD (or the
OLLAMAQ_* env vars) at startup — register this server with a `cwd` at the project
root in .mcp.json. Run: `python -m ollamaq.mcp_server`.

The logic lives in mcp_core (unit-tested); this module is only the MCP wiring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import mcp_core
from .config import load_config
from .errors import OllamaqError
from .queue import JobQueue

def _auto_home() -> Optional[Path]:
    """Locate the project root from this file's location (editable install:
    <root>/src/ollamaq/mcp_server.py), so the server works from any client and
    any working directory with no configuration. None if config/ isn't there
    (e.g. a wheel install into site-packages) — then env vars / cwd take over."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "config" / "models.yaml").is_file() else None


# Precedence for where the server finds config/queue/results:
#   explicit OLLAMAQ_* env var  ->  $OLLAMAQ_HOME  ->  auto-detected project root
#   ->  the current working directory.
# So a project-scoped .mcp.json needs no hardcoded path (auto-detect handles it),
# while Codex/Grok/Desktop launching from elsewhere can set one OLLAMAQ_HOME.
_HOME = os.environ.get("OLLAMAQ_HOME")
_HOME_PATH = Path(_HOME) if _HOME else _auto_home()


def _root(env_key: str, sub: str) -> Path:
    explicit = os.environ.get(env_key)
    if explicit:
        return Path(explicit)
    if _HOME_PATH:
        return _HOME_PATH / sub
    return Path(sub)


_QUEUE_ROOT = _root("OLLAMAQ_QUEUE", "queue")
_RESULTS_ROOT = _root("OLLAMAQ_RESULTS", "results")

mcp = FastMCP("ollamaq")


def _cfg():
    # Load per-call so a config edit doesn't require a server restart. Prefer the
    # resolved home's config; else fall back to load_config's own env/cwd search.
    if _HOME_PATH:
        return load_config(_HOME_PATH / "config" / "models.yaml")
    return load_config()


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
    """Return a finished job's digest.md (the compact, Claude-facing summary)."""
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


def main() -> None:  # pragma: no cover - entry point
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
