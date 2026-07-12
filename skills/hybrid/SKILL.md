---
name: hybrid
description: Unified hybrid orchestrator skill for Claude, Grok, and Codex. Offload bulk map-phase work to local ollamaq/Ollama; keep premium tokens for orchestration, reduce/clean, and high-reasoning only. Use when bulk/mechanical/gen_test work appears, or when starting a tandem-tokens session.
---

# Hybrid orchestrator (Claude / Grok / Codex)

You are the **premium orchestrator**. Local **ollamaq** (tandem-tokens) runs map work on Ollama. Energy and cloud $ are scarce — maximize local/offline.

## Core contract

1. Worker output stays on disk under `results/<job_id>/`.
2. You read **only** `digest.md` (via `ollamaq digest` / MCP `queue_digest` / `hybrid_digest`) — hard-capped ~8 KB.
3. Pull `merged.jsonl` only for a deliberate reduce on residue; prefer a slice.
4. Never echo bulk worker output into the conversation.

## When to offload

Offload when **≥ ~20 similar independent items** AND one of:

| Mode | Use for |
|------|---------|
| `map` | clean / normalize / extract / classify / summarize / boilerplate codegen |
| `gen_test` | generate candidates + scriptable verifier |
| `pipeline` | cheap triage → stronger model on flagged subset |

## When NOT to offload

- Single hard reasoning problem
- Needs repo-wide context the local model lacks
- Wrong answers silently corrupt data with no verifier
- Fewer than ~20 items (overhead > savings)

## Routing (`config/models.yaml`)

`model: "auto"` + `task_tag`:

- `clean` / `extract` / `code` → workhorse (qwen2.5-14b class)
- `summarize` → long-context local
- `classify` → fast small local
- `generate` → gen_test model

Optional residual escalate: job `retry.escalate_to` or config `fallbacks.escalate_to` —
local bigger model, or `cloud:grok` / `cloud:grok/grok-3-mini` (needs `cloud.grok.enabled` + `XAI_API_KEY`).

## Workflow

1. **Offload** — write a validated job spec → `ollamaq submit` / MCP `queue_submit` / `hybrid_offload`.
2. Ensure daemon: `ollamaq daemon` (one per machine). Don’t poll tightly.
3. **Harvest** — when `done`, read digest only; reduce residue; write final artifact.
4. **Score** — `ollamaq score` / `hybrid_score` for tokens, cloud $ avoided, GPU Wh.

## MCP tools

| Standard | Grok alias | Purpose |
|----------|------------|---------|
| `ollama_quick` | `hybrid_quick` | tiny one-shot (capped) |
| `queue_submit` | `hybrid_offload` | enqueue batch |
| `queue_status` | `hybrid_status` | queue state |
| `queue_digest` | `hybrid_digest` | finished digest |
| `recommend_models` | `hybrid_recommend` | VRAM fit |
| `hybrid_score` | same | scoreboard + energy |
| `hybrid_energy` | same | live nvidia-smi power |
| `hybrid_policy` | same | this policy text |

- **Claude Code**: `.mcp.json` → `python -m ollamaq.mcp_server` (+ `CLAUDE.md` / `.claude/skills/`)
- **Codex**: `clients/codex.config.toml` → `~/.codex/config.toml`; policy in `AGENTS.md`
- **Grok**: `python -m ollamaq.mcp_grok` (hybrid_* aliases) — `clients/grok.mcp.json`
- **Cursor / VS Code / Continue / Cline / Windsurf**: `clients/cursor.mcp.json` or `clients/generic.mcp.json`
- **No MCP**: shell only — `ollamaq submit|run|status|digest|score` (works from any agent that can run commands)

Client matrix: [clients/README.md](../../clients/README.md).

## Example map spec

```json
{
  "name": "clean-contacts",
  "mode": "map",
  "model": "auto",
  "task_tag": "clean",
  "input": {
    "kind": "csv",
    "path": "data/contacts.csv",
    "chunk": { "strategy": "rows", "size": 50 }
  },
  "prompt": {
    "system": "You are a data cleaning engine. Output ONLY JSON.",
    "template": "Normalize these rows... return {\"rows\": [...]}. Input:\\n{{chunk}}",
    "schema": { "type": "object", "required": ["rows"] }
  },
  "params": { "temperature": 0.1 },
  "retry": {
    "max_attempts": 3,
    "on_schema_fail": "repair_then_retry",
    "escalate_to": "auto"
  }
}
```

## Kickoff checklist

1. `ollamaq doctor` — config + Ollama OK  
2. `ollamaq recommend` — models fit GPU  
3. `hybrid_policy` or re-read this skill  
4. Offload eligible bulk work; harvest digests; report savings  
