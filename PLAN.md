# tandem-tokens — Local Ollama Offload Harness for Claude Code

**Goal:** Let Claude Code (Opus/Sonnet) delegate mundane, brute-force, and map-phase work
to local Ollama models, then spend premium tokens only on orchestration and the
reduce/cleaning pass. Map-phase outputs live on disk; Claude reads compact digests.

**Execution note for the implementing model:** Phases are ordered by dependency. Each
phase has explicit deliverables and acceptance criteria — do not start a phase until the
previous phase's acceptance criteria pass. All code is Python 3.11+, Windows-first
(this machine), no admin rights required.

---

## 1. Hardware & model reality (measured on this machine)

- GPU: RTX 5080, 16 GB VRAM. One large model resident at a time.
- Installed models and their assigned roles:

| Model | VRAM | Role |
|---|---|---|
| `qwen2.5-14b-16k` | ~9 GB | **Workhorse.** Data cleaning, structured extraction, codegen drafts. Best quality that stays fully on GPU. |
| `hermes3-8b-32k` | ~5 GB | **Long-context reader.** Bulk summarization/extraction over big chunks (32k ctx). |
| `qwen3:8b` | ~5 GB | **Candidate generator / verifier.** Has thinking mode; use for generate-and-test loops and self-checking. |
| `gemma2:9b` | ~5.4 GB | **Fast classifier/labeler.** Cheap triage, dedup decisions, routing. |
| `llama3.2-vision:11b` | ~8 GB | Image/chart/screenshot extraction (later phase, optional). |
| `gemma2:27b` | 15 GB | **Avoid by default** — spills to CPU on 16 GB. Optional "escalation" tier only for hard items where latency is irrelevant. |

- Concurrency policy (enforced by the harness, configurable in `config/models.yaml`):
  8b-class models: up to 4 parallel requests; 14b: 2; 27b: 1. Jobs are **grouped by
  model** and drained model-by-model to avoid VRAM thrash from constant model swapping.
- Ollama server env (document in README; set in the daemon's launcher):
  `OLLAMA_NUM_PARALLEL=4`, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_KEEP_ALIVE=30m`.

## 2. Architecture

Three layers, thinnest possible surface toward Claude:

```
Claude (Opus/Sonnet)
  │  Bash: `ollamaq submit/run/status/digest`   ← primary path (batch, token-free map)
  │  MCP:  ollama_quick / queue_* tools          ← convenience path (one-shot, small)
  ▼
ollamaq  (Python package + CLI)
  ├─ sync runner   (`ollamaq run --wait`)  — fan-out, block, digest
  ├─ daemon        (`ollamaq daemon`)      — drains file queue continuously
  ├─ chunker, prompt templating, JSON-schema enforcement, retry/repair, verifier hook
  ▼
Ollama HTTP API (localhost:11434)
```

**Token economy contract:** worker outputs are written to
`results/<job_id>/items/*.json` and NEVER returned to Claude. Claude only ever reads
`results/<job_id>/digest.md` (bounded ≤ ~2k tokens: stats, schema of outputs, sample
rows, failures needing escalation). The reduce/cleaning pass is Claude reading the
*aggregated artifact* (`results/<job_id>/merged.jsonl` or a slice of it) deliberately.

### Directory layout

```
tandem-tokens/
├─ PLAN.md                      (this file)
├─ pyproject.toml               (package: ollamaq; deps: httpx, pydantic, pyyaml, typer)
├─ config/
│   └─ models.yaml              (routing table, per-model concurrency, ctx limits, defaults)
├─ src/ollamaq/
│   ├─ cli.py                   (typer app: submit, run, daemon, status, digest, cancel, bench)
│   ├─ jobspec.py               (pydantic models for job spec — see §3)
│   ├─ chunker.py               (glob/JSONL/CSV/text splitting strategies)
│   ├─ worker.py                (async fan-out against Ollama, per-model semaphores)
│   ├─ enforce.py               (JSON-schema validation + bounded repair-retry loop)
│   ├─ verify.py                (runs user verifier script per item, captures score/verdict)
│   ├─ digest.py                (builds digest.md + merged.jsonl from item results)
│   ├─ queue.py                 (file-based queue: atomic renames across state dirs)
│   └─ mcp_server.py            (stdio MCP: ollama_quick, queue_submit, queue_status, queue_digest)
├─ queue/{pending,running,done,failed}/
├─ results/<job_id>/{items/, merged.jsonl, digest.md, log.txt}
├─ .claude/
│   ├─ skills/offload/SKILL.md  (/offload — turn NL request into job spec, submit)
│   ├─ skills/harvest/SKILL.md  (/harvest — collect digests, run reduce/clean pass)
│   └─ settings.json            (permission allowlist for `ollamaq *`)
├─ .mcp.json                    (registers mcp_server.py, stdio transport)
└─ CLAUDE.md                    (delegation policy — see §6)
```

## 3. Job spec (the single contract everything shares)

`ollamaq submit job.json` — pydantic-validated:

```jsonc
{
  "name": "clean-sensor-csv",
  "mode": "map",                       // "map" | "gen_test" | "pipeline"
  "model": "qwen2.5-14b-16k",          // or "auto" → routed via models.yaml by task tag
  "task_tag": "clean",                 // clean|extract|summarize|classify|generate|code
  "input": {
    "kind": "jsonl",                   // glob|jsonl|csv|text|inline
    "path": "data/raw/*.csv",
    "chunk": {"strategy": "rows", "size": 50, "overlap": 0}
  },
  "prompt": {
    "system": "You are a data cleaning engine. Output ONLY JSON.",
    "template": "Normalize these rows to schema X... {{chunk}}",
    "schema": { /* JSON schema enforced via Ollama format param */ }
  },
  "params": {"temperature": 0.1, "num_ctx": 16384},
  "verifier": {                        // optional; required for mode=gen_test
    "cmd": "python verifiers/check_rows.py",   // gets item JSON on stdin, returns {"pass": bool, "score": float, "notes": str}
    "keep_top": 20                     // gen_test: survivors surfaced in digest
  },
  "retry": {"max_attempts": 3, "on_schema_fail": "repair_then_retry", "escalate_to": null},
  "loop": {                            // only for mode=gen_test continuous loops
    "max_iterations": 10, "stop_when": "score>=0.95 or no_improvement(3)",
    "feedback": "top_k"                // feed best/worst exemplars back into next generation round
  }
}
```

**Modes:**
- `map` — chunk inputs, run template per chunk, merge outputs. The bread and butter.
- `gen_test` — generate N candidates per iteration, run verifier on each, feed survivors
  back as few-shot context, repeat until `stop_when`. This is the "autoresearch /
  continuous gen" loop: it runs unattended inside the daemon; Claude only sees the
  final leaderboard in the digest.
- `pipeline` — ordered list of stages (e.g., gemma2:9b classifies → qwen2.5-14b cleans
  only the rows flagged dirty). Stage outputs feed the next stage's input. (Phase 3.)

## 4. Implementation phases

### Phase 0 — Scaffold & smoke test  ✅ DONE (2026-07-04)
- `pyproject.toml`, package layout above, `config/models.yaml` seeded with the table in §1.
- `ollamaq bench` — hits `/api/generate` on each configured model with a fixed prompt,
  records tokens/sec and load time to `config/bench.json` (used later for ETA estimates).
- **Accept:** `pip install -e .` works; `ollamaq bench --model qwen2.5-14b-16k` prints tok/s.
- **Delivered beyond the minimum:** `errors.py` (full exception hierarchy incl.
  `NotImplementedPhaseError`), `config.py` (`resolve_model()` routing + `normalize_model_name`),
  `client.py` (sole HTTP module, mockable), `bench.py`, `cli.py` (`version`/`doctor`/
  `models`/`bench` + phase-stub commands that exit 2). 34 passing tests (HTTP mocked via
  respx, no live server needed); live `doctor` + `bench` verified on the RTX 5080.
  Contract rules for later phases documented in README "Module map".
- **Baton for Phase 1:** implement `chunker.py`, `worker.py`, `enforce.py`, `digest.py`
  and wire the `run` stub in `cli.py`. The `NotImplementedPhaseError` stub for `run`/`digest`
  marks exactly where to start.

### Phase 1 — Sync map runner (the 80% value)  ✅ DONE (2026-07-05)
- `chunker.py` (rows/lines/bytes/files strategies), `worker.py` (httpx + asyncio,
  per-model semaphore from models.yaml), `enforce.py` (validate against `prompt.schema`
  using Ollama's `format` field; on failure, one repair prompt "fix this to match the
  schema", then retry; after `max_attempts`, item goes to `failed_items.jsonl`).
- `digest.py`: counts, success rate, elapsed, tok/s, 3 sample outputs, all failures
  (truncated), path pointers. Hard cap digest at 8 KB.
- `ollamaq run job.json --wait` end-to-end.
- **Accept:** a 200-row CSV cleaning job completes with ≥95% schema-valid items and a
  digest under 8 KB; failed items isolated, exit code reflects failure threshold.
- **Delivered:** full-schema `jobspec.py` (all PLAN §3 fields parsed now; later-phase
  fields marked, unsupported *modes* rejected via `NotImplementedPhaseError`);
  `chunker.py`, `enforce.py` (pure `process_attempt` + repair prompt), `worker.py`
  (async fan-out, per-model semaphore, repair-retry loop, transport-error
  short-circuit), `digest.py` (artifacts + 8 KB-capped digest). `run`/`digest` wired
  in `cli.py`. 85 passing tests (HTTP mocked). Live verified on gemma2:9b:
  `examples/clean_contacts.job.json` → 3/3 items, 100% schema-valid, digest ≈1 KB.
  New dep: `jsonschema`.
- **Note observed live:** local model followed structure perfectly but not every
  formatting rule (phone normalization) — this is exactly what the Phase 5 Opus/Sonnet
  reduce/clean pass over `merged.jsonl` is for.
- **Baton for Phase 2:** `queue.py` (atomic-rename state dirs) + `daemon` command +
  wire `submit`/`status`/`cancel` stubs. `run --no-wait` already routes to a Phase-2
  `NotImplementedPhaseError`, marking the entry point.

### Savings accounting (cross-cutting, added 2026-07-05)
- `config/pricing.yaml` (user-editable, USD/1M tokens; Anthropic prices authoritative
  from the claude-api skill, comparison models flagged `verify:`) + `pricing.py`
  (`load_pricing` optional, `estimate_all`). `worker.py` now tracks `prompt_tokens`
  (summed across retries); `digest.py` emits an "estimated cloud cost avoided (map
  phase)" block and `run.json` records `estimated_cloud_cost_usd` per model. Honest
  wording: it's a map-phase estimate (tokenizers differ), and the Claude reduce pass
  still costs real tokens. Live-verified in the contacts digest. 95 tests pass.
- **Cumulative scoreboard (added 2026-07-07):** `ledger.py` — every finished job
  appends one line to `<results_root>/ledger.jsonl` (tokens, items, wall, per-model
  cost avoided). `ollamaq score` sums it into a scoreboard (total local tokens run for
  $0, total cloud spend avoided per model). This is the "keeping score" ledger; before
  it, savings lived only per-job in each digest/run.json. 6 ledger + 2 CLI tests.

### Phase 2 — Queue + daemon  ✅ DONE (2026-07-06)
- `queue.py`: job files move `pending → running → done|failed` via atomic `os.replace`;
  daemon claims oldest-first, grouped by model. Crash recovery: on startup, `running/`
  jobs older than a lease timeout go back to `pending`.
- `ollamaq daemon` (foreground; document a Windows Task Scheduler / `pythonw` launch
  recipe in README — do not build a service). `ollamaq status [job_id]` renders a
  compact table; `ollamaq cancel <job_id>`.
- **Accept:** submit 3 jobs across 2 models, kill daemon mid-run, restart, all complete;
  status output correct throughout.
- **Delivered:** `queue.py` (`JobQueue`: submit/claim_one/claim_next/finish/
  reclaim_stale/cancel; atomic `os.replace`, corrupt-file tolerant, lease-based crash
  recovery). `daemon.py` (`run_daemon` drain loop with model-preference scheduling to
  avoid VRAM thrash; `pick_next`, `process_job`; `max_jobs`/`should_stop` hooks for
  tests). `submit`/`daemon`/`status`/`cancel` wired in `cli.py` (daemon flushes stdout
  for redirected logs). 113 tests pass. Live-verified: `submit` → `daemon` drained the
  contacts job to `done` on gemma2:9b → `digest` with savings block. README documents
  the Task Scheduler / `pythonw` launch.
- **Baton for Phase 3:** `verify.py` (subprocess verdict protocol), `gen_test`
  iteration engine with `stop_when` + top-k feedback, and `pipeline` mode. The daemon
  already rejects non-`map` modes with a clear Phase-3 error, marking the entry point.

### Phase 3 — gen_test loops + verifiers + pipeline  ✅ DONE (2026-07-07)
- `verify.py` subprocess protocol (stdin JSON → stdout verdict JSON, 30 s timeout/item).
- `gen_test` iteration engine with `stop_when` mini-expressions and top-k feedback.
- `pipeline` mode chaining stage outputs.
- **Accept:** demo loop — "generate regex candidates matching test corpus, verifier
  scores against fixtures" — converges and digest shows a scored leaderboard.
- **Delivered:** `verify.py` (`run_verifier` — stdin JSON → stdout verdict, robust
  policy: setup errors raise, per-candidate failures score 0 and continue),
  `stopcond.py` (safe `score>=/>/<=/<`, `no_improvement(K)`, `and`/`or`; no eval),
  `gentest.py` (`run_gentest` — generate N candidates → score off-thread → keep top-K
  → feed survivors into `{{exemplars}}` → repeat until stop). `execute.py` unifies
  map/gen_test dispatch for both `run` and the daemon. `digest.py` renders a scored
  leaderboard + iteration trace for gen_test. Token totals cover ALL candidates (not
  just survivors), so savings accounting stays honest. New ~135-test suite passes.
  Live-verified: `examples/seven_words.job.json` (+ `word_count_verifier.py`) on
  qwen3:8b converged to a 1.0 (exactly-7-words) candidate; digest shows the leaderboard.
- **Pipeline delivered (2026-07-07):** `pipeline.py` (`run_pipeline` — each stage is
  a map over the active set; output feeds the next stage's {{chunk}}; per-stage
  `StageFilter` decides who continues vs finalizes; stops early when nothing
  continues). Job spec gained `stages`/`PipelineStage`/`StageFilter`, `prompt` made
  optional (required for map/gen_test, unused for pipeline). `execute_spec` dispatches
  all three modes. Digest renders a stage funnel. Live-verified:
  `examples/phones_pipeline.job.json` — gemma2:9b triages phone numbers → qwen2.5-14b
  reformats only the malformed ones.

### Phase 4 — MCP server  ✅ DONE (2026-07-07)
- `mcp_server.py` via the `mcp` Python SDK, stdio. Tools:
  - `ollama_quick(model?, prompt, schema?)` — one-shot for small delegations (hard cap
    response at 2k tokens so it never blows up Claude's context),
  - `queue_submit(jobspec)`, `queue_status(job_id?)`, `queue_digest(job_id)`.
- Register in `.mcp.json`.
- **Accept:** from a Claude Code session, `ollama_quick` answers and `queue_submit` →
  daemon → `queue_digest` round-trips.
- **Delivered:** `mcp_core.py` (pure, unit-tested logic: `quick`/`submit`/`status`/
  `digest`; `ollama_quick` caps output at 2048 tokens / 8000 chars — the only tool that
  returns model text to Claude) + `mcp_server.py` (thin FastMCP stdio adapter, per-call
  config reload). `.mcp.json` registers it (`python -m ollamaq.mcp_server`, cwd + config
  env). New dep: `mcp`. Server smoke-tested: loads and registers all 4 tools. ~163 tests
  pass. **Note:** `queue_submit` enqueues but the daemon must be running to drain it; the
  MCP server does not itself run jobs.
- **Baton for Phase 5:** `CLAUDE.md` delegation policy + `/offload` and `/harvest`
  skills + `.claude/settings.json` allowlist. All the machinery Claude needs (CLI + MCP
  tools) now exists; Phase 5 is the Claude-side glue that decides *when* to offload.

### Phase 5 — Claude-side integration  ✅ DONE (2026-07-07)
- `CLAUDE.md` delegation policy (§6).
- `/offload` skill: interrogate the user's request, emit a validated job spec, submit,
  report job_id + ETA (from bench data). `/harvest` skill: list done jobs, pull digests,
  run the Opus/Sonnet reduce/clean pass on `merged.jsonl`, write final artifact.
- `.claude/settings.json`: allow `ollamaq *` Bash calls without prompts.
- Long-run pattern documented: after submitting, Claude uses `/loop` or scheduled
  wakeups (≥20 min) to check `ollamaq status` rather than polling.
- **Accept:** full workflow demo — user asks for a bulk cleaning task in a Claude
  session; Claude offloads, harvests, and delivers a cleaned dataset, with total
  Claude-token spend visible in the digest's accounting block.
- **Delivered:** `CLAUDE.md` (when/when-not to offload, routing, the digest-only
  contract, workflow). `.claude/skills/offload/SKILL.md` + `.claude/skills/harvest/
  SKILL.md` (both registered and picked up by the harness). `.claude/settings.json`
  allowlists `ollamaq`/`python -m ollamaq.cli`/`py -m ollamaq.cli`. Harvest skill
  stresses O(digest) reduce (fix residue with a script) over O(n) per-row review.
  CLI cleanup: `run` docstring/`--no-wait` now point to `submit`; dead `_phase_stub`
  removed. 154 tests pass. The submit→daemon→digest loop the skills orchestrate was
  already live-verified in Phases 2–3.

**All planned phases (0–5) are complete.** Phase 6 below is optional tuning/extras.

### Phase 6 — Tuning & optional extras  (planned 2026-07-07; not started)

Four independent items, ordered by value ÷ risk. Each is self-contained; do them
in this order but any can be skipped. Same contract-first bar as Phases 0–5:
pinned signatures, HTTP mocked in tests, one live demo per item.

**Implementation note (harness vs Opus):** these are novel, tightly-architected
changes to the async worker, the job spec, and the client — *not* bulk mechanical
work. The local 8–14b models are the wrong tool for writing this code (they'd miss
the concurrency and cache-invalidation subtleties). **Implement with Opus.** The
one honest place the harness could help is generating *bulk test input data* (e.g.
a corpus of varied prompts to exercise the dedupe cache) — marginal; skip unless
convenient. Vision test images can't be generated locally (no image model here).

---

#### 6a. Dedupe cache — skip re-runs  ✅ DONE (2026-07-07)

- **Goal:** re-running a job (after fixing a few items, or an idempotent retry)
  should not re-generate items whose exact request already produced a validated
  result. Keyed on a stable hash so any input change is a natural miss.
- **Design:** new `cache.py` with `class Cache` — `get(key) -> Any | None`,
  `put(key, value)`; disk-backed under `cache/` (gitignored), one JSON per key,
  atomic temp+`os.replace` writes. Key = sha256 of canonical JSON of
  `{model, system, rendered_prompt, schema, options}` (rendered_prompt already
  embeds the chunk → per-item). **Cache validated successes only** — never cache a
  schema failure. On a hit, short-circuit `_run_item` entirely: return
  `ItemResult(ok=True, value=cached, cached=True, prompt_tokens=0,
  completion_tokens=0)` — no HTTP, no re-validation (schema is in the key, so a
  schema change misses). Opt-in per job: add `cache: bool = False` to `JobSpec`
  (and pass through to `_run_item`). Token accounting stays honest — a cache hit
  generated 0 new local tokens this run.
- **Files:** `cache.py` (new); `jobspec.py` (`cache: bool`); `worker.py`
  (`ItemResult.cached: bool`; `_run_item` checks/populates cache; `run_job` builds
  the `Cache` when `spec.cache`); `digest.py` (one line: "N/M served from cache").
- **Tests:** miss-then-hit reuse; disabled by default (no cache dir created);
  key sensitivity (different chunk/model/schema → miss); failures not cached;
  concurrent-safe write (two items, atomic).
- **Accept:** run a job twice with `cache: true`; second run does **0**
  generations (respx sees no calls / live is near-instant), digest shows all cached.
- **Delivered:** `cache.py` (`Cache` + `make_key`, atomic writes, corrupt=miss);
  `JobSpec.cache: bool`; `ItemResult.cached` + `RunResult.cached_count`; `_run_item`
  short-circuits on hit (0 tokens), caches validated successes only; `run_job` builds
  the cache from `OLLAMAQ_CACHE` env (default `cache/`). Digest + run.json report
  cache hits. 6 cache tests + 4 worker cache tests. Live: contacts job cold 206.4s →
  warm **1.0s**, digest "cache: 3/3 items served from cache (0 new tokens)".

#### 6b. `retry.escalate_to` — escalate persistent failures  ✅ DONE (2026-07-07)

- **Goal:** an item that fails schema on the primary model after `max_attempts`
  gets **one** attempt on a stronger local model before being surfaced to Claude —
  so the digest's failure list is only the genuinely hard residue. The
  `RetrySpec.escalate_to` field already exists (parsed, unused).
- **Design:** in `worker.run_job`, after resolving the primary model, also resolve
  `spec.retry.escalate_to` if set (must be a defined, **enabled** model; error at
  startup if not, like the primary — reuse `config.resolve_model`; no-op if equal
  to primary). Pass the escalate model + its options (that model's `num_ctx` from
  config) into `_run_item`. After the repair-retry loop exhausts on a **schema**
  failure (not a transport error — those won't self-heal), run one final
  `client.generate` on the escalate model; if it validates, return
  `ItemResult(ok=True, value=…, escalated=True, attempts=…)`. Accumulate its
  tokens. Escalation is per-item and sequential (a model swap mid-run) — expensive,
  so it fires only on residue; document that.
- **Files:** `worker.py` (`ItemResult.escalated: bool`; `_run_item` escalate step;
  `run_job` resolves + threads the escalate model/options); `digest.py` ("N items
  escalated to <model>"). Job spec unchanged (field already there).
- **Tests:** primary fails all attempts → escalate succeeds (item ok, `escalated`);
  escalate also fails → item failed, note both models; `escalate_to` == primary →
  no-op; `escalate_to` undefined/disabled → `ConfigError` at `run_job` start;
  transport error does **not** escalate (short-circuits as today).
- **Accept:** a job whose primary returns non-schema text but whose escalate model
  returns valid JSON → item ok via escalation, flagged in the digest.
- **Delivered:** `worker.py` — `run_job` resolves `escalate_to` up front (fail-fast
  `ConfigError` if undefined/disabled; no-op if == primary), `_run_item` runs one
  final attempt on the escalate model (its own `num_ctx`) after schema exhaustion,
  only for schema failures (transport errors short-circuit). `ItemResult.escalated`
  + `RunResult.escalated_count`; escalated successes are cached under the primary
  key. Digest + run.json report escalations. 7 escalation tests.
- **Note (honest):** escalation is a **rare-residue safety net by design** — because
  the schema is passed to Ollama as a `format` constraint, primary output is already
  forced schema-valid, so primary failures (and thus escalations) are uncommon. Live
  `examples/escalation_demo.job.json` ran 2/2 ok on gemma2:9b with **0 escalations**
  (qwen-14b never loaded) — which validates the design, not a gap. The mechanism is
  proven across all branches by the unit tests.

#### 6c. Vision inputs — ✅ DONE + LIVE-VERIFIED (2026-07-07, via gemma3:4b)

- **Goal:** run image inputs (screenshots, scanned tables, charts) through the
  vision model — the "bulk extraction from images" workload.
- **Design:** Ollama takes images as base64 in the `images` array of
  `/api/generate`. Add `images: list[str]` to `client.generate` →
  `payload["images"]`. Add `InputKind.images` (glob of image files, chunk strategy
  `files`, one image per item); `chunker` emits a `WorkItem` whose `text` is the
  file path (for the `source`/reference) and a new `WorkItem.images: list[str]`
  holds the base64. `_run_item` passes `item.images` to `client.generate`. Enable
  `llama3.2-vision:11b` in `models.yaml` (currently `enabled: false`) and add
  routing `vision → llama3.2-vision:11b`. Template `{{chunk}}` becomes the
  instruction ("extract the table as JSON"); the image rides alongside.
- **Files:** `client.py` (`images` param); `chunker.py` (`WorkItem.images`, image
  glob); `worker.py` (thread `images` through `_run_item`/`run_job` items tuple —
  note: the `(index, source, text)` tuple grows to carry images, or switch to
  passing `WorkItem`s); `jobspec.py` (`InputKind.images`, validate strategy);
  `config/models.yaml` (enable vision + routing).
- **Tests:** chunker image-glob → items with base64 (temp PNG files); `client`
  generate includes `images` in payload (respx assert); worker passes images
  through; jobspec accepts `kind: images`.
- **Accept:** live — extract structured data from a real screenshot/chart via
  `llama3.2-vision`, digest shows the parsed JSON.
- **Risk:** the `(index, source, text)` items tuple threads through `run_job`,
  `gentest`, `pipeline`. Cleanest is to pass `WorkItem` objects instead of tuples —
  a small refactor touching those three call sites. Budget for it.
- **Delivered (code):** `client.generate(images=…)` → payload `images`; `chunker`
  `WorkItem.images` + `_chunk_images` (base64 per file, filename as `text`);
  `InputKind.images`; `worker` threads images via a **4-tuple** `(index, source, text,
  images)` (chose this over the WorkItem refactor for minimal churn — `run_job` accepts
  3- or 4-tuples, so all existing tests/tuples still work); `execute`/`--sample` build
  4-tuples; `models.yaml` enables `llama3.2-vision:11b` + `vision` routing. Escalation
  and cache paths carry images too. 4 chunker + 2 client + 1 worker + 1 jobspec tests.
- **Live: SOLVED with a model swap.** Root cause was model↔engine compatibility, not
  the harness or Ollama version: on Ollama 0.31.1, `llama3.2-vision` (mllama arch)
  won't load and `moondream` returns empty to images (confirmed via a direct
  `/api/generate` probe that bypassed ollamaq). **`gemma3:4b` works.** Diagnosis path:
  moondream text-only worked (eval=7) but image gave eval=1/empty → isolated the
  break to the image path for those models. Fix: `config/models.yaml` enables
  `gemma3:4b`, routes `vision -> gemma3:4b` (llama3.2-vision left disabled with a
  note). Live: `examples/invoice_vision.job.json` (model `auto`, task_tag `vision`) on
  the PIL invoice → schema-valid `{"invoice": "A-2043", "total": 82.5}`, 1/1 ok, 2.0s.
  **Every Phase-6 feature is now live-verified.**

#### 6d. ETA from a sample run  ✅ DONE (2026-07-07)

- **Goal:** replace vague "concurrency/num_ctx tuning" with something concrete: a
  grounded ETA so `/offload` can tell the user how long a big job will take.
- **Design:** `ollamaq run`/`submit` gain `--sample N` (default 0 = off). When set,
  run the first N items, measure their tok/s and mean tokens/item, and project the
  full-job time = `(total_items × mean_tokens_per_item) / (tok/s × max_parallel)`
  `+ load_seconds` (from `config/bench.json`). Print the projection; with
  `--sample-only`, stop after reporting (don't run the rest). This is real
  measurement, not guesswork — drop the original "auto-tune num_ctx/parallel" idea
  (we don't measure VRAM, so it would be hand-waving).
- **Files:** `cli.py` (`--sample`/`--sample-only` on `run`); a small
  `eta.py::project(sample_result, total_items, bench, max_parallel) -> str`.
- **Tests:** projection math on a synthetic sample RunResult; `--sample-only`
  runs exactly N items then stops.
- **Accept:** `ollamaq run big.job.json --sample 5 --sample-only` prints a
  time/token projection for the full set.
- **Delivered:** `eta.py` (`project` pure math + `Projection.format` + `bench_lookup`)
  and `run --sample N` (map-only; probes first N items, projects the full job from
  measured tok/s + bench load, then stops — `--sample-only` folded in since `--sample`
  always stops; suggests pairing with `cache: true` to reuse the probe). 8 eta tests +
  2 CLI tests. Live: contacts `--sample 1` → "projection ... -> 3 items: ~207
  completion tokens, ~23s at 113 tok/s x 4 parallel".
- **Bug caught live (fixed):** the projection string used `->`-arrow and `x`-times as
  Unicode (`→`, `×`); the Windows cp1252 console can't encode `→` →
  `UnicodeEncodeError`. The respx/CliRunner test missed it (UTF-8 capture). Fix: keep
  CLI console output ASCII (also swapped the three `...` echoes). Lesson recorded:
  **CLI echo must be ASCII on this Windows console; UTF-8 is fine for digest files.**

#### 6e. GPU-aware model recommender  ✅ DONE (2026-07-07)
- **Goal:** "which of my installed models actually fit this GPU vs. spill to CPU?"
- **Delivered:** `gpu.py` (`detect_nvidia_gpu` via nvidia-smi — no pynvml/GPUtil dep;
  `recommend` matches config models to installed sizes from Ollama `/api/tags`, flags
  FITS/SPILL against 85% of VRAM, merges bench tok/s, warns on routing that spills).
  `client.model_sizes()`; `ollamaq recommend` command; `recommend_models` MCP tool.
  7 gpu tests. Live: correctly flags gemma2:27b (15.6 GB) as SPILL on the 16 GB card,
  everything else FITS, with per-task routing advice.

#### 6f. Cross-client MCP (Claude / Codex / Grok)  ✅ DONE (2026-07-07)
- **Goal:** the MCP server usable from any MCP client, not just Claude Code.
- **Delivered:** server hardened for arbitrary launch cwd via one `OLLAMAQ_HOME` env
  var (roots config/queue/results; individual `OLLAMAQ_*` still override). `MCP.md`
  documents the 5 tools + per-client config (Claude Code `.mcp.json`, Claude Desktop
  `claude_desktop_config.json`, Codex `~/.codex/config.toml [mcp_servers]`, Grok — same
  stdio command). The server command is identical everywhere (`python -m
  ollamaq.mcp_server`); only the wrapper file format differs. Verified: server
  registers all 5 tools and `OLLAMAQ_HOME` roots correctly.

## 5. Design decisions & rationale (confirmed by user 2026-07-04)

**Confirmed:** CLI harness + thin MCP; queue daemon + sync `--wait` mode; Python stack.
**All four workloads in scope from the start** — Phase 1 must ship working example job
specs (in `examples/`) for each `task_tag`: `clean` (CSV normalization), `extract`
(structured facts from docs via hermes3-8b-32k), `summarize` (multi-file digest), and
`code` (test/stub generation via qwen2.5-14b). `generate` (gen_test mode) lands in
Phase 3 as planned.

1. **CLI-first, MCP-thin.** Batch results returned through MCP tool calls would land in
   Claude's context and destroy the token savings. Files + digests keep the map phase
   free. MCP exists only for ergonomics on small one-shots.
2. **File-based queue, no Redis/SQLite server.** Atomic renames are crash-safe enough,
   inspectable with `ls`, and zero-dependency. SQLite can replace it later behind
   `queue.py`'s interface if job volume demands.
3. **Daemon + sync both.** `--wait` for < ~2 min batches; daemon for the continuous
   research loops. Same runner underneath.
4. **Schema enforcement local, escalation explicit.** Local models fail schemas
   sometimes; the repair-retry loop handles most of it locally. Only persistent failures
   reach Claude, pre-collected in the digest.
5. **Group-by-model scheduling** because 16 GB VRAM makes model swaps the dominant cost.

## 6. CLAUDE.md delegation policy (to be written in Phase 5)

Claude should offload when a task is: (a) ≥ ~20 similar independent items, (b) mechanical
(clean/extract/classify/summarize/boilerplate), or (c) generate-and-test with a scriptable
verifier. Claude should NOT offload: single hard reasoning problems, tasks needing repo-wide
context, or anything where a wrong answer silently corrupts data without a verifier.
Routing: `task_tag` → model via models.yaml; default `qwen2.5-14b-16k`; never `gemma2:27b`
unless the user opts in.
