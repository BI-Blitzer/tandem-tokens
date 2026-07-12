"""Pure logic behind the MCP tools, kept separate from the MCP wiring so it can
be unit-tested without an MCP client. mcp_server / mcp_grok are thin adapters.

The token-economy contract still holds: `queue_submit` returns only a job id,
`queue_status` returns compact state, and `queue_digest` returns the ≤8 KB digest
file — bulk worker output never flows back through these tools. `ollama_quick` is
the ONE tool that returns model text to the premium client, so it hard-caps output.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from .client import OllamaClient
from .config import Config, load_config
from .errors import OllamaqError
from .jobspec import JobSpec
from .queue import JobQueue, State

# ollama_quick is the only tool whose result enters premium context, so bound it.
QUICK_MAX_TOKENS = 2048
QUICK_MAX_CHARS = 8000

HYBRID_POLICY = """\
# hybrid offload policy (ollamaq / tandem-tokens)

You are the premium orchestrator (Grok / Claude / Codex). Local Ollama does map work.

OFFLOAD when ALL of:
  - >= ~20 similar independent items, AND
  - mechanical (clean/extract/classify/summarize/boilerplate) OR gen_test OR staged pipeline

DO NOT offload:
  - single hard reasoning problems
  - tasks needing full repo context the local model lacks
  - silent-corruption risk with no verifier
  - fewer than ~20 items (setup cost > savings)

CONTRACT:
  - Worker output stays on disk under results/<job_id>/
  - You read ONLY digest.md (and scoreboard/energy tools) — never bulk merged.jsonl
    unless deliberately running a reduce pass on a small residue
  - Prefer queue_submit / hybrid_offload over ollama_quick for batches
  - After done: hybrid_digest / queue_digest, then reduce only failures/samples

ROUTING (model: auto + task_tag):
  clean/extract/code -> workhorse (qwen2.5-14b class)
  summarize -> long-context local
  classify -> fast small local
  generate -> gen_test model
  Optional escalate_to: local bigger model OR cloud:grok[/model] (needs XAI_API_KEY)

METERING:
  hybrid_score / ollamaq score -> tokens + cloud $ avoided + GPU Wh
  hybrid_energy -> live nvidia-smi power draw
"""


def auto_project_home() -> Optional[Path]:
    """Project root from package location (editable install), else None."""
    # src/ollamaq/mcp_core.py -> parents[2] == project root
    root = Path(__file__).resolve().parents[2]
    return root if (root / "config" / "models.yaml").is_file() else None


def resolve_home() -> Optional[Path]:
    """OLLAMAQ_HOME env, else auto-detected project root."""
    home = os.environ.get("OLLAMAQ_HOME")
    if home:
        return Path(home)
    return auto_project_home()


def resolve_root(env_key: str, sub: str, home: Optional[Path] = None) -> Path:
    """Path for queue/results/config: env override -> $HOME/sub -> cwd/sub."""
    explicit = os.environ.get(env_key)
    if explicit:
        return Path(explicit)
    home = home if home is not None else resolve_home()
    if home:
        return home / sub
    return Path(sub)


def load_cfg_for_mcp() -> Config:
    """Config for MCP processes: prefer project home config, else search path."""
    home = resolve_home()
    if home:
        return load_config(home / "config" / "models.yaml")
    return load_config()


async def quick(cfg: Config, prompt: str, model: Optional[str] = None,
                task_tag: Optional[str] = None,
                schema: Optional[dict] = None) -> str:
    """One-shot generation for a small delegation. Output is capped so it can
    never blow up premium context. Returns the (possibly truncated) text."""
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
    from .gpu import detect_gpu, recommend as build_recommendation

    gpu = detect_gpu()
    sizes: dict = {}
    async with OllamaClient(cfg.ollama) as client:
        if await client.ping():
            sizes = await client.model_sizes()
    return build_recommendation(cfg, sizes, gpu, bench=None).format()


def scoreboard(results_root: Path) -> str:
    """Cumulative ledger scoreboard (tokens, cloud $ avoided, energy $ / CO2)."""
    from .energy_cost import load_energy_rates
    from .ledger import read_entries, summarize
    from .pricing import load_pricing

    entries = read_entries(results_root)
    sb = summarize(entries)
    headline = None
    try:
        pricing = load_pricing()
        if pricing is not None:
            headline = pricing.default_compare
    except OllamaqError:
        pass
    rates = None
    try:
        rates = load_energy_rates()
    except OllamaqError:
        pass
    return sb.format(headline_model=headline, energy_rates=rates)


def energy_now() -> str:
    """Live GPU power sample as JSON text."""
    from .energy import sample_power

    s = sample_power()
    if s is None:
        return json.dumps({"available": False, "reason": "nvidia-smi unavailable"})
    return json.dumps({
        "available": True,
        "watts": round(s.watts, 2),
        "limit_w": round(s.limit_w, 2),
        "temp_c": round(s.temp_c, 1),
    })


def hybrid_policy_text() -> str:
    """Static policy text for premium orchestrators."""
    return HYBRID_POLICY
