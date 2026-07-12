# AGENTS.md — hybrid orchestrator (Codex, Cursor, Grok CLI, others)

This project is **ollamaq / tandem-tokens**: a local Ollama harness that does
map-phase work so premium agents (Codex, Claude, Grok, …) spend tokens only on
orchestration and reduce/clean.

**Full policy skill:** [skills/hybrid/SKILL.md](skills/hybrid/SKILL.md)  
**Claude-oriented copy:** [CLAUDE.md](CLAUDE.md) + `.claude/skills/{offload,harvest,hybrid}/`  
**MCP setup by client:** [clients/README.md](clients/README.md) · [MCP.md](MCP.md)

## Contract (do not break)

1. Worker output stays on disk under `results/<job_id>/`.
2. Read **only** `results/<job_id>/digest.md` (or MCP `queue_digest` /
   `hybrid_digest`) — not bulk `merged.jsonl` unless you are deliberately
   reducing a small residue.
3. Never paste bulk worker output into the chat.

## Offload bar

Offload when **≥ ~20 similar independent items** and the work is mechanical
(`map`), generate-and-test (`gen_test`), or staged (`pipeline`). Otherwise do
it inline — setup cost exceeds savings.

## Commands (CLI — works without MCP)

```text
ollamaq doctor
ollamaq recommend
ollamaq submit <job.json> --job-id <name>
ollamaq run <job.json> --wait          # small batches
ollamaq status
ollamaq digest <job_id>
ollamaq score                          # tokens + $ avoided + GPU Wh
ollamaq daemon                         # one per machine
```

## MCP tools (Codex / Cursor / generic)

Server: `python -m ollamaq.mcp_server`  
Config snippet: [clients/codex.config.toml](clients/codex.config.toml) or
[clients/cursor.mcp.json](clients/cursor.mcp.json).

| Tool | Use |
|------|-----|
| `queue_submit` | Bulk offload |
| `queue_status` | Poll done/failed |
| `queue_digest` | Harvest summary only |
| `ollama_quick` | Tiny one-shot (capped; not for bulk) |
| `recommend_models` | VRAM fit |
| `hybrid_score` | Savings scoreboard |
| `hybrid_energy` | Live GPU watts |
| `hybrid_policy` | This policy as tool text |

Grok clients may use `python -m ollamaq.mcp_grok` for `hybrid_*` aliases of the
same tools.

## Routing

`model: "auto"` + `task_tag` in the job spec → `config/models.yaml`. Default
workhorse is the local 14B-class model. Optional residual escalate:
`retry.escalate_to: "auto"` or `cloud:grok` (needs key + `cloud.grok.enabled`).

## Workflow

1. Confirm offload-eligible → write `jobs/<name>.job.json`.
2. `ollamaq submit` / MCP `queue_submit` (daemon running).
3. When done → `digest` only → reduce residue → final artifact.
4. Report `ollamaq score` savings (tokens, $, Wh when metered).
