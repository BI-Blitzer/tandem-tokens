# Hybrid campaign — premium reduce findings

**Date:** 2026-07-12 · **Host:** Scion-Zwei (RTX 5080 16 GB)  
**Orchestration:** Grok + ollamaq MCP · **Map phase:** local Ollama only  

This is the **premium reduce** over digests only (no bulk re-read of every row).
Artifacts: `results/uc{1,2,3}-*/digest.md`, UC1 clean list
`results/uc1-map-clean-contacts/merged.normalized.jsonl`.

---

## Executive summary

Three complementary workloads ran in VRAM-aware order
(**summarize → pipeline → map clean**). All jobs finished **100% schema-ok**.
Map-phase cloud API cost was **$0**; estimated Opus-class map spend avoided for
the campaign alone was **~$0.74** at **~11.3 Wh** local energy (**~$0.002** at
$0.16/kWh). Net map-phase savings ≈ **~$0.74**.

| Job | Mode | Model | Items | Wall | Wh | Opus avoided (est.) |
|-----|------|--------|------:|-----:|---:|--------------------:|
| UC3 docs | map summarize | hermes3-8b-32k | 36/36 | 49 s | 1.63 | ~$0.21 |
| UC2 support | pipeline | gemma2:9b → qwen2.5-14b | 48/48 | 100 s | 4.24 | ~$0.21 |
| UC1 contacts | map clean | qwen2.5-14b-16k | 15×20 rows | 133 s | 5.47 | ~$0.32 |

---

## UC3 — Knowledge compression (hermes3-8b-32k)

**Intent:** Compress runbook-style docs before a premium theme cluster.

**Digest signal:** 36/36 ok; summaries carry domain labels (energy, ops, security,
data, product) and `CAMPAIGN-…` refs.

**Reduce takeaway (premium):** Themes already align with the hybrid policy itself:

1. **Energy** — measure tokens + wall + Wh together; offload map work.  
2. **Ops** — one model resident; queue by model family to avoid thrash.  
3. **Security** — MCP verifiers are trust-boundary; don’t expose untrusted.  
4. **Data** — schema + repair-retry is the map-phase quality lever.  
5. **Product** — one harness for Claude / Grok / Codex.

**Action:** No row-level rework. Optional: keep a static “theme card” on the
landing page (done in `docs/index.html` use-cases section).

---

## UC2 — Support triage funnel (pipeline)

**Intent:** Cheap triage for all tickets; 14B drafts only when `needs_human`.

**Digest funnel:**

```text
triage (gemma2:9b):     in 48, ok 48, continued 45
draft_reply (qwen 14b): in 45, ok 45
```

**Filter worked:** ~6% of tickets (3/48) finalized at triage without a draft —
exactly the energy/token win of pipeline mode.

**Reduce takeaway:**

- Priority/`needs_human` from stage 1 is good enough for routing.  
- Drafts are usable first responses; a human should still approve refunds,
  security lockouts, and production-down claims (sample drafts sometimes
  over-committed on billing adjustments — **policy: never auto-send drafts**).  
- Premium agent role = review **high/urgent** drafts only (~subset of 45), not all 48.

**Action:** Ship pipeline pattern as the default helpdesk template; gate send on human.

---

## UC1 — Contact clean (qwen2.5-14b-16k) + quality pass

**Intent:** CRM list hygiene at volume.

**Digest:** 15/15 chunks ok (300 contacts). Model format adherence on phones was
**~83%** exact `XXX-XXX-XXXX` (often dropped a digit or kept punctuation).

**Quality pass (mechanical residue — preferred over re-burning 14B):**

1. Tightened job prompt (digits-only → last-10 → `ddd-ddd-dddd`).  
2. Dogfood 60-row sample: model improved punctuation but still dropped digits (~85%).  
3. **Source-grounded normalizer** `scripts/normalize_contacts_from_source.py`:
   rebuilds phones from the original CSV digits (authoritative), keeps model
   name/email when present.

**Deliverable:** `results/uc1-map-clean-contacts/merged.normalized.jsonl`  
with **~100% phones** reformatted from source digits.

**Lesson for hybrid policy:** structured fields with deterministic rules
(phone E.164/US display) should finish in a **script**, not a second LLM pass.

---

## Scoreboard snapshot (ledger, post-campaign)

From MCP `hybrid_score` after campaign (+ earlier smoke jobs):

- Local tokens ~197k cumulative; map phase cloud API **$0**  
- Metered energy ~13 Wh · local **~$0.002** · CO₂e ~5 g  
- Cloud avoided (Opus headline) ~**$2.82** cumulative  
- Net vs Opus ~**$2.82**

---

## Recommended next operations

1. **Daemon always-on** so Claude + Grok MCP `queue_submit` drains without a
   foreground shell.  
2. **Scale** contacts/tickets with the same jobs + post-normalize phones.  
3. Point UC1/UC2 at **production** CSV/JSONL when ready.  

See also: [HYBRID_USE_CASES.md](HYBRID_USE_CASES.md).
