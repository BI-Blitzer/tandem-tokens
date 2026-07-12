# ollamaq — delegation policy for Claude Code

This project is `ollamaq`: a harness that offloads bulk, mechanical, or
generate-and-test work to **local Ollama models** so premium Claude tokens are
spent only on orchestration and the final reduce/cleaning pass. Read
[PLAN.md](PLAN.md) for the architecture; [README.md](README.md) for commands.

## The core contract (do not break)

Worker output stays on disk. **You read only `results/<job_id>/digest.md`** (hard-
capped at 8 KB: stats, samples, failures, cloud cost avoided when pricing is
configured, and GPU energy Wh / $ / CO₂ when metered). Pull
`results/<job_id>/merged.jsonl` into context **only** when you are deliberately
running a reduce/clean pass over it, and prefer a slice. Never echo bulk worker
output back into the conversation. After harvest, prefer `ollamaq score` for
cumulative savings (cloud $ avoided − local energy $).

## When to offload

Offload when the task is **(a) ≥ ~20 similar independent items**, AND one of:
- **mechanical** — clean / normalize / extract / classify / summarize / boilerplate
  codegen over many rows, files, or records (use `map` mode), or
- **generate-and-test** — produce candidates in volume and score them with a
  scriptable verifier (use `gen_test` mode), or
- **staged** — a cheap model triages, an expensive model handles only the flagged
  subset (use `pipeline` mode).

## When NOT to offload

- A single hard reasoning problem (no map phase to parallelize) — do it yourself.
- Anything needing repo-wide context the local model won't have.
- Anything where a wrong answer silently corrupts data and there is no verifier.
- Fewer than ~20 items — the setup/queue overhead exceeds the savings; just do it.

## Routing (from config/models.yaml)

`model: "auto"` routes by `task_tag`: `clean`/`extract`/`code` → qwen2.5-14b-16k,
`summarize` → hermes3-8b-32k (long context), `classify` → gemma2:9b (fast),
`generate` → qwen3:8b. Default model is qwen2.5-14b-16k. **Never** target
`gemma2:27b` (spills to CPU on the 16 GB card) unless the user explicitly opts in.

## Workflow

1. **`/offload`** — turn the request into a validated job spec, submit it, get a
   job id + rough ETA. For a small batch use `ollamaq run <spec> --wait`; for a
   large or long one use `ollamaq submit <spec>` and let the daemon drain it.
2. If using the daemon, make sure one is running: `ollamaq daemon` (one per
   machine). After submitting, **don't poll tightly** — use `/loop` or a scheduled
   wake-up (≥20 min) to check `ollamaq status`.
3. **`/harvest`** — when jobs are `done`, read their digests, and run the
   Opus/Sonnet reduce/clean pass over the residue the local model got wrong
   (guided by the digest's failures/samples), then write the final artifact.
   Cross-client policy also lives in `skills/hybrid/SKILL.md` and `AGENTS.md`
   (Codex/Grok/Cursor). Report energy lines from the digest when present.

## Auto-offload nudge (hook)

A `UserPromptSubmit` hook (`hooks/auto_offload.py`, registered in
`.claude/settings.json`) scans each prompt and, when it looks like a bulk
mechanical data task, injects a `[tandem-tokens]` suggestion naming the likely
mode. It surfaces the opportunity — you still decide whether to `/offload`. It
stays silent on ordinary prompts. (A hook can't auto-*execute* a job: a prompt
isn't a validated job spec.)

## MCP tools (alternative to the CLI)

An MCP server (`.mcp.json`) exposes `ollama_quick` (one-shot, output-capped —
for tiny delegations only), `queue_submit`, `queue_status`, `queue_digest`.
Prefer the CLI for batch work; `ollama_quick` is for a single small prompt whose
short result is fine to bring back into context.
