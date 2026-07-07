---
name: offload
description: Turn a bulk/mechanical data task into a validated ollamaq job spec, submit it to the local Ollama harness, and report the job id + ETA. Use when the user wants to clean/extract/classify/summarize many items, generate-and-test candidates, or run a staged pipeline — offloading the work off Claude's token budget.
---

# /offload — delegate bulk work to local Ollama

Follow these steps. The goal is a **validated** job spec that runs on a local
model, so Claude tokens are spent only on the later reduce pass. See
[../../../PLAN.md](../../../PLAN.md) §3 for the full spec and [CLAUDE.md] for the
when-to-offload policy.

## 1. Confirm it's worth offloading

Check the policy in `CLAUDE.md`: ≥ ~20 similar independent items AND mechanical,
generate-and-test, or staged. If it fails the bar, say so and just do it inline.

## 2. Pick the mode and task_tag

- **map** — one transformation per item (clean / extract / classify / summarize /
  codegen). `task_tag` ∈ clean, extract, summarize, classify, code.
- **gen_test** — generate N candidates, score with a verifier, keep the best.
  `task_tag: generate`. Requires a `verifier` command and usually a `loop`.
- **pipeline** — ordered `stages`; a per-stage `filter` decides which items
  continue. Remember: **each stage must carry forward any payload the next stage
  needs** (a classifier should output `{flag, ...the data...}`, not just `{flag}`).

## 3. Describe the input

`input.kind`: `csv`/`jsonl` (chunk `rows`), `text` (chunk `lines`/`bytes`),
`glob` (chunk `files`, one file per item), or `inline` (`values`, chunk `rows`).
Pick a `chunk.size` that fits the model context (start ~50 rows for cleaning,
fewer for wide rows).

## 4. Write the prompt + schema

`prompt.template` MUST contain `{{chunk}}`. Add a JSON `schema` whenever the
output is structured — Ollama constrains decoding to it, and the harness
validates + repairs against it. Put "output ONLY JSON, no prose, no fences" in
`prompt.system`. For gen_test, an optional `{{exemplars}}` placeholder gets the
current top survivors fed back.

## 5. Emit and validate the spec

Write the spec to `jobs/<name>.job.json` (create `jobs/` if needed). Then:

- **Small batch (< ~2 min):** run it synchronously and go straight to results:
  `ollamaq run jobs/<name>.job.json --config config/models.yaml --job-id <id>`
- **Large / long / unattended:** enqueue it for the daemon:
  `ollamaq submit jobs/<name>.job.json --config config/models.yaml --job-id <id>`
  then ensure a daemon is running (`ollamaq status` shows the queue; start one
  with `ollamaq daemon` if needed).

If the spec is invalid, the command exits 2 with the exact validation error —
fix the spec and retry. Do NOT hand-wave past a validation failure.

## 6. Report

Give the user the **job id**, the resolved model, and a rough **ETA** from
`config/bench.json` (that model's tok/s and load time — if bench.json is missing,
suggest `ollamaq bench`). Then tell them to run `/harvest` (or check
`ollamaq status <id>`) once it's `done`. Remind them that the map phase runs for
$0 in Claude tokens; only the harvest/reduce pass will cost tokens.

## Example map spec

```json
{
  "name": "clean-contacts",
  "mode": "map",
  "model": "auto",
  "task_tag": "clean",
  "input": { "kind": "csv", "path": "data/contacts.csv",
             "chunk": { "strategy": "rows", "size": 50 } },
  "prompt": {
    "system": "You are a data cleaning engine. Output ONLY JSON.",
    "template": "Normalize these rows... return {\"rows\": [...]}. Input:\n{{chunk}}",
    "schema": { "type": "object", "required": ["rows"] }
  },
  "params": { "temperature": 0.1 },
  "retry": { "max_attempts": 3, "on_schema_fail": "repair_then_retry" }
}
```
