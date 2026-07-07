# ollamaq

Local Ollama offload harness for Claude Code. Claude (Opus/Sonnet) delegates
map-phase and brute-force work to local models; results stay on disk; Claude
reads only compact digests. See [PLAN.md](PLAN.md) for the full architecture and
phase roadmap.

**Status: complete (Phases 0–6, all live-verified).** Sync map runner (`run`), file
queue + daemon (`submit`/`daemon`/`status`/`cancel`), all three job modes (`map`,
`gen_test` generate-and-verify loop, `pipeline` chained stages), vision inputs, digests
with cloud-cost savings accounting + a cumulative `score` scoreboard, a dedupe cache,
model escalation, `--sample` ETA, a GPU-aware `recommend`, a cross-client MCP server,
and Claude-side integration (`CLAUDE.md` policy + `/offload` & `/harvest` skills).

## Documentation

| Doc | What |
|---|---|
| [PLAN.md](PLAN.md) | Full architecture + the phase-by-phase build log (design, decisions, live results). |
| [CLAUDE.md](CLAUDE.md) | Delegation policy — when Claude should/shouldn't offload, routing, the digest-only contract. |
| [MCP.md](MCP.md) | Using the MCP server from Claude Code/Desktop, Codex, Grok, and other MCP clients. |
| this README | Install, commands, workflow. |

## Claude workflow

In a Claude Code session in this repo: **`/offload`** turns a bulk task into a validated
job spec and submits it; the daemon drains it locally; **`/harvest`** reads the digest
and runs the reduce/clean pass only on the residue. See [CLAUDE.md](CLAUDE.md) for the
when-to-offload policy.

## Install

```pwsh
py -m pip install -e ".[dev]"
```

## Configure

All model routing lives in [config/models.yaml](config/models.yaml) — the single
source of truth. No module hardcodes model names. Resolution order for the config
file: `--config` flag → `$OLLAMAQ_CONFIG` → `$OLLAMAQ_HOME/config/models.yaml` →
`./config/models.yaml`.

## Commands

```pwsh
ollamaq version                 # print version
ollamaq models                  # show routing table + per-model settings
ollamaq doctor                  # check config, Ollama reachability, model coverage
ollamaq bench                   # benchmark all enabled models -> config/bench.json

# Synchronous: run one job now, block until done, write the digest
ollamaq run examples/clean_contacts.job.json      # map mode (bulk clean)
ollamaq run examples/seven_words.job.json         # gen_test mode (generate + verify loop)
ollamaq run examples/phones_pipeline.job.json     # pipeline mode (triage -> clean flagged)

# Queue + daemon: enqueue jobs, drain them 24/7 in the background
ollamaq submit examples/clean_contacts.job.json --job-id nightly-clean
ollamaq daemon                  # foreground; drains the queue (run ONE per machine)
ollamaq status                  # counts per state (pending/running/done/failed)
ollamaq status nightly-clean    # one job's state + resolved model + error
ollamaq cancel nightly-clean    # remove a still-pending job
ollamaq digest nightly-clean    # print results/<job_id>/digest.md
ollamaq score                   # cumulative scoreboard: local tokens run for $0
                                #   and total cloud spend avoided across all jobs
ollamaq recommend               # which installed models fit this GPU vs. spill to CPU
```

Every finished job appends to `results/ledger.jsonl`; `ollamaq score` sums it into
a running scoreboard (total local tokens, per-model cloud cost avoided). A good
end-to-end test job to try: `examples/support_triage.job.json` (24 support tickets
→ category/priority/needs-human classification, with `cache: true`).

`doctor` exit codes: 0 healthy · 1 Ollama unreachable or an enabled model is not
installed · 2 config error. `run` exits 1 when the success rate falls below
`--min-success` (default 0.95).

Add `"cache": true` to a job spec to reuse prior validated outputs on re-runs
(keyed on model+prompt+schema+options; validated successes only). Cached items
cost 0 tokens and skip generation — a re-run of an unchanged job is near-instant.
Cache lives under `cache/` (override with `OLLAMAQ_CACHE`).

`retry.escalate_to: <model>` gives an item that keeps failing schema on the
primary model one attempt on a stronger local model before it's surfaced as a
failure. `--sample N` on `run` probes the first N map items and prints a full-job
time/token projection (then stops). `input.kind: images` (chunk `files`) sends
each matched image to a vision model (`task_tag: vision` → `gemma3:4b`); see
`examples/invoice_vision.job.json` for schema-constrained extraction from an image.

## MCP — use from Claude, Codex, Grok, and other MCP clients

The MCP server (`python -m ollamaq.mcp_server`) exposes `ollama_quick`,
`queue_submit`, `queue_status`, `queue_digest`, and `recommend_models` to any MCP
client. Point one `OLLAMAQ_HOME` env var at the project and it works regardless of
launch directory. Per-client config (Claude Code/Desktop, Codex, Grok) is in
[MCP.md](MCP.md).

### Running the daemon under Windows Task Scheduler

The daemon is a foreground process; to run it unattended, launch it with `pythonw`
(no console window) and redirect its log:

```pwsh
# One-off / at logon. Runs ONE daemon per machine (single GPU, one model resident).
pythonw -m ollamaq.cli daemon --config config/models.yaml *> C:\_Work\Claude_Coded\endless-code\daemon.log
```

Register it as a Scheduled Task set to "Run whether user is logged on or not" with
trigger "At log on". The daemon reclaims any job left mid-run (its lease older than
1 h) back to `pending/` on startup, so a reboot loses no work.

## Ollama server env (recommended)

Set before launching `ollama serve` so behavior matches the concurrency policy in
models.yaml:

```pwsh
$env:OLLAMA_NUM_PARALLEL = "4"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_KEEP_ALIVE = "30m"
```

## Development

```pwsh
py -m pytest              # full suite, no live server needed (HTTP is mocked)
py -m pytest -m integration   # opt-in; requires a live Ollama (set OLLAMAQ_LIVE=1)
```

### Module map (what a contributor / lower model may touch)

| Module | Responsibility | May raise |
|---|---|---|
| `errors.py` | All exception types | — |
| `config.py` | Load/validate models.yaml; `resolve_model()` routing | `ConfigError` family |
| `client.py` | Only HTTP path to Ollama; `generate/ping/list_models` | `OllamaConnectionError`, `OllamaHTTPError` |
| `bench.py` | Sequential model benchmark → `bench.json` | never (folds errors into results) |
| `chunker.py` / `worker.py` | Split input into items; async fan-out with repair-retry | `JobSpecError` / `OllamaqError` |
| `enforce.py` / `digest.py` | Schema validate + repair; write artifacts + ≤8 KB digest | — |
| `queue.py` / `daemon.py` | File queue (atomic renames) + drain loop | `OllamaqError` |
| `verify.py` / `stopcond.py` / `gentest.py` | Verifier subprocess; `stop_when` eval; gen_test loop | `VerifierError` / `ValueError` |
| `execute.py` | One `execute_spec` dispatch (map / gen_test) for `run` + daemon | `NotImplementedPhaseError` (pipeline) |
| `pricing.py` | Estimate cloud $ avoided, per model, into the digest | `ConfigError` |
| `cli.py` | Typer commands; thin wrappers over the above | exits with codes above |

Contract rules that later phases must not break: (1) model names appear only in
`models.yaml`; (2) `client.py` is the sole HTTP module; (3) `bench.bench_one`
never raises; (4) worker/job output must never be returned through Claude's
context — only digests (enforced from Phase 1 on).
