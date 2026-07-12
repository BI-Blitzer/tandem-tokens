# Hybrid campaign use cases (local Ollama + premium orchestrator)

Three complementary workloads that exercise **map**, **pipeline**, and
**long-context summarize** on an RTX 5080 (16 GB). Designed for
**model-grouped execution** so VRAM is not thrashed by constant swaps.

Premium agent (Grok / Claude / Codex) only **orchestrates** and **harvests digests**.
Map-phase work stays on disk under `results/<job_id>/`.

---

## Resource policy (this machine)

| Constraint | Implication |
|------------|-------------|
| 16 GB VRAM, one large model resident | Prefer **one model family at a time** |
| Daemon drains queue | Submit in **strategic order** (below) or run sync sequential |
| Energy metering (nvidia-smi) | Longer jobs → measurable Wh + `$` via `config/energy.yaml` |
| Cloud $ avoided | Token-heavy map phases dominate the scoreboard |

### Recommended execution order

```text
1) UC3  hermes3-8b-32k     summarize     (~5 GB)   — warm small long-ctx model
2) UC2  gemma2:9b → qwen2.5-14b-16k  pipeline — triage then draft (one swap)
3) UC1  qwen2.5-14b-16k    map clean     (~9 GB)  — reuses 14b after pipeline stage 2
```

Why this order:

1. **UC3 first** isolates the long-context 8B model; no 14B load yet.
2. **UC2 pipeline** loads **gemma2:9b** for cheap triage, then **qwen 14B** only for
   tickets that need a human-facing draft (`needs_human == true`).
3. **UC1 last** keeps **qwen 14B** resident (no thrash after pipeline stage 2).

Avoid interleaving unrelated 14B and 8B jobs; each swap costs load time and energy.

---

## UC1 — Bulk contact clean (`map` · workhorse)

| | |
|--|--|
| **Job** | `jobs/uc1_map_clean_contacts.job.json` |
| **Model** | `qwen2.5-14b-16k` (`task_tag: clean`) |
| **Input** | `data/hybrid_campaign/contacts_messy.csv` (300 messy rows) |
| **Chunks** | 20 rows → **15** work items |
| **Output** | Normalized `{id,name,email,phone}` per row |

**Use case:** CRM / marketing list hygiene. Mechanical, high volume, schema-enforced.
Premium model should not touch every row — only harvest failures from the digest.

**Success criteria:** high success_rate; phones exactly `XXX-XXX-XXXX`; emails lowercased.

**Harvest:** `ollamaq digest uc1-map-clean-contacts`. Phone format is finished mechanically:

```pwsh
py scripts/normalize_contacts_from_source.py data/hybrid_campaign/contacts_messy.csv results/uc1-map-clean-contacts/merged.jsonl
# → merged.normalized.jsonl (100% phones from source digits)
```

LLM map stays for name/email structure; deterministic phone rules do not re-burn GPU.

---

## UC2 — Support triage + draft (`pipeline` · staged)

| | |
|--|--|
| **Job** | `jobs/uc2_pipeline_support.job.json` |
| **Stage 1** | `gemma2:9b` classify (category, priority, `needs_human`) |
| **Filter** | continue only if `needs_human == true` |
| **Stage 2** | `qwen2.5-14b-16k` draft reply + internal note |
| **Input** | `data/hybrid_campaign/support_tickets.jsonl` (48 tickets) |

**Use case:** Helpdesk intake. Cheap model triages everyone; expensive local model
only drafts replies for tickets that need a human. Matches the hybrid energy goal:
**don’t run 14B on praise/how-to that can auto-close.**

**Success criteria:** stage funnel in digest (stage1 in → continued → stage2);
drafts stay concrete (no invented refunds).

**Harvest:** inspect funnel counts; premium agent reviews only high/urgent drafts.

---

## UC3 — Doc summarization (`map` · long-context)

| | |
|--|--|
| **Job** | `jobs/uc3_summarize_docs.job.json` |
| **Model** | `hermes3-8b-32k` (`task_tag: summarize`) |
| **Input** | `data/hybrid_campaign/docs.jsonl` (36 docs, ~1.4k chars each) |
| **Chunks** | 1 doc → **36** work items |
| **Output** | summary, key_points, campaign_refs |

**Use case:** Knowledge base / runbook compression before a premium reduce pass
that clusters themes (energy, ops, security, …).

**Success criteria:** schema-valid summaries; `campaign_refs` captures `CAMPAIGN-…` ids.

**Harvest:** cluster digests by `domain` offline; optional premium synthesis of themes.

---

## Commands

### Sequential (recommended for a controlled demo)

```pwsh
# 1) long-context summarize
ollamaq run jobs/uc3_summarize_docs.job.json --config config/models.yaml --job-id uc3-summarize-docs

# 2) pipeline triage + draft
ollamaq run jobs/uc2_pipeline_support.job.json --config config/models.yaml --job-id uc2-pipeline-support

# 3) workhorse clean
ollamaq run jobs/uc1_map_clean_contacts.job.json --config config/models.yaml --job-id uc1-map-clean-contacts

ollamaq score
ollamaq history --limit 10
```

### Queued (daemon)

```pwsh
# Submit in the same strategic order so the daemon sees UC3, then UC2, then UC1
ollamaq submit jobs/uc3_summarize_docs.job.json --job-id uc3-summarize-docs
ollamaq submit jobs/uc2_pipeline_support.job.json --job-id uc2-pipeline-support
ollamaq submit jobs/uc1_map_clean_contacts.job.json --job-id uc1-map-clean-contacts
ollamaq daemon   # one per machine
```

### Via MCP (Grok / Claude)

Tools: `queue_submit` → `queue_status` → `queue_digest` → `hybrid_score` / `hybrid_energy`.  
Daemon must be running for `queue_submit` to complete jobs.

---

## What the scoreboard should show after the campaign

- **Local tokens** from all three jobs (map-phase cloud API $0)
- **Energy Wh** + **$** (`config/energy.yaml`) + optional **CO₂**
- **Cloud $ avoided** vs Opus/Sonnet/etc. (estimate)
- **Net** ≈ cloud avoided − local energy $

Per-job digests under `results/uc*/digest.md` (≤ 8 KB each).

---

## File map

```text
data/hybrid_campaign/
  contacts_messy.csv      # UC1
  support_tickets.jsonl   # UC2
  docs.jsonl              # UC3
jobs/
  uc1_map_clean_contacts.job.json
  uc2_pipeline_support.job.json
  uc3_summarize_docs.job.json
docs/HYBRID_USE_CASES.md  # this file
```

---

## Live results (2026-07-12, RTX 5080)

Executed in strategic order UC3 → UC2 → UC1. All digests under `results/uc*/`.

| Job | Model | Items | Wall | Tokens (in/out) | Energy | Cloud avoided (Opus est.) |
|-----|--------|------:|-----:|----------------:|-------:|--------------------------:|
| **UC3** summarize | hermes3-8b-32k | 36/36 | 49 s | 15505 / 5342 | **1.63 Wh** | ~$0.21 |
| **UC2** pipeline | gemma2:9b → qwen 14b | 48/48 | 100 s | 13054 / 5613 | **4.24 Wh** | ~$0.21 |
| **UC1** map clean | qwen2.5-14b-16k | 15/15* | 133 s | 11051 / 10643 | **5.47 Wh** | ~$0.32 |
| **Campaign total** | — | 99 | ~4.7 min | ~40k / ~22k | **~11.3 Wh** | **~$0.74** |

\*15 chunks × 20 rows = 300 contacts cleaned.

**UC2 funnel:** triage in 48 → continued **45** (`needs_human`) → draft_reply **45** ok  
(only 3 tickets finalized at triage without a draft — cheap model filter working).

**Local energy $ (campaign):** ~11.3 Wh × $0.16/kWh ≈ **$0.0018**  
**Net vs Opus (campaign map phase):** ~$0.74 − ~$0.002 ≈ **~$0.74**  

Cumulative scoreboard after campaign (includes earlier smoke/history jobs):  
`ollamaq score` / MCP `hybrid_score` — see live output for full ledger.
