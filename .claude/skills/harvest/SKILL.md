---
name: harvest
description: Collect finished ollamaq jobs — read their compact digests, then run the Opus/Sonnet reduce/clean pass over only the residue the local model got wrong, and write the final artifact. Use after /offload jobs reach the "done" state.
---

# /harvest — reduce and finalize offloaded work

The map phase already ran locally for $0. Harvest is where Claude tokens get
spent — so spend them **only on the residue**, guided by each digest. See
[CLAUDE.md] for the token-economy contract.

## 1. Find finished jobs

`ollamaq status` — shows counts per state and the ids in each. Note the ids in
`done` (and `failed`). For one job: `ollamaq status <job_id>`.

## 2. Read the digest FIRST (cheap)

`ollamaq digest <job_id>` — prints `results/<job_id>/digest.md` (≤ 8 KB): success
rate, sample outputs, the full list of failures, and the estimated cloud cost
avoided. This is usually all you need to decide what to do next. **Do not read
`merged.jsonl` yet.**

## 3. Decide the reduce strategy

- **Clean pass (all good):** if the digest shows high success and the samples
  look right, the local output is the deliverable. Point the user at
  `results/<job_id>/merged.jsonl` (and `items/`). No token-heavy pass needed.
- **Fix the residue:** if the digest shows a systematic local error (e.g. a
  format the small model didn't follow) or lists failures, prefer a **mechanical
  fix** — write a small script/transform that corrects that specific error across
  `merged.jsonl`, and run it. This keeps the reduce at O(digest), not O(n).
- **Per-row review (last resort):** only read all of `merged.jsonl` into context
  when the task genuinely needs Claude to inspect every row. This is the
  expensive path — say so, and confirm it's warranted.
- **Escalate failures:** re-run just the `failed_items.jsonl` items as a new
  small job on a stronger local model, or handle them inline.

## 4. Write the final artifact

Produce the cleaned/aggregated output the user asked for (a file under their
project, not under `results/`). State plainly what you did: how many items came
from the local run untouched, how many you corrected and how, and how many
failed. Include the digest's "cost avoided" figure so the savings are visible.

## 5. gen_test / pipeline jobs

- **gen_test:** the digest is a scored **leaderboard** — the deliverable is the
  top candidate(s). Report the best and its score; no reduce pass usually needed.
- **pipeline:** the digest shows a **stage funnel**; `merged.jsonl` holds each
  item at its final stage. Verify the funnel matches intent (right counts
  continued vs finalized) before finalizing.
