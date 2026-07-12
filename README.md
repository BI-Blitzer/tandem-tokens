# ollamaq

[![CI](https://github.com/BI-Blitzer/tandem-tokens/actions/workflows/ci.yml/badge.svg)](https://github.com/BI-Blitzer/tandem-tokens/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-0f8c68.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab.svg)](pyproject.toml)
[![Platforms](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-6b6a64.svg)](#deploy)
[![MCP server](https://img.shields.io/badge/MCP-server-17b58f.svg)](MCP.md)

**Site: [bi-blitzer.github.io/tandem-tokens](https://bi-blitzer.github.io/tandem-tokens/)** · `pip`-installable, Docker + Pinokio, cross-client MCP.

**Local Ollama offload harness for any premium agent.** Claude, Grok, Codex, Cursor,
or the CLI alone can orchestrate map-phase work on your GPU; results stay on disk;
the premium model reads only compact digests. Same contract everywhere — only the
client config file format differs. See [PLAN.md](PLAN.md) for architecture;
[docs/HYBRID_USE_CASES.md](docs/HYBRID_USE_CASES.md) for a live three-job campaign.

**Status: complete (Phases 0–6 + hybrid/energy pack).** Sync map runner (`run`), file
queue + daemon (`submit`/`daemon`/`status`/`cancel`), job modes `map` / `gen_test` /
`pipeline` / `embed`, vision inputs, digests with cloud-cost savings + cumulative
`score` scoreboard, dedupe cache, model escalation (local or optional `cloud:grok`),
`--sample` ETA, GPU-aware `recommend`, cross-client MCP (`clients/`), optional Grok
facade (`ollamaq-mcp-grok`), and hybrid skills (`skills/hybrid`, `AGENTS.md`, plus
Claude-oriented `/offload` · `/harvest` under `.claude/skills/`).

**Energy:** nvidia-smi meters GPU Wh; `config/energy.yaml` turns that into local `$`
and optional CO₂. `ollamaq score` shows cloud $ avoided, energy cost, and net vs the
headline cloud model. Smoke: `examples/energy_smoke.job.json`.

## Documentation

| Doc | What |
|---|---|
| [PLAN.md](PLAN.md) | Full architecture + the phase-by-phase build log (design, decisions, live results). |
| [AGENTS.md](AGENTS.md) | Hybrid policy for Codex, Cursor, Grok CLI, and other agents. |
| [CLAUDE.md](CLAUDE.md) | Same policy, Claude Code–oriented (skills + hook). |
| [skills/hybrid/SKILL.md](skills/hybrid/SKILL.md) | Portable skill — works across clients. |
| [MCP.md](MCP.md) | MCP setup for Claude, Grok, Codex, Cursor, and generic hosts. |
| [clients/README.md](clients/README.md) | Drop-in MCP configs (JSON/TOML) per client. |
| [docs/index.html](docs/index.html) | Public landing page (GitHub Pages `/docs`): calculator, use cases, MCP. |
| [docs/HYBRID_USE_CASES.md](docs/HYBRID_USE_CASES.md) | Live three-job campaign (map / pipeline / summarize). |
| [docs/CAMPAIGN_FINDINGS.md](docs/CAMPAIGN_FINDINGS.md) | Premium reduce write-up from campaign digests. |
| this README | Install, commands, workflow. |

## Security

A job spec's `verifier.cmd` (gen_test mode) runs an **arbitrary local subprocess** —
powerful, but it means submitting a spec is submitting code to run. From the CLI
that's you on your own machine. Over MCP the server may be reachable by a
less-trusted client, so **verifier jobs are refused over MCP by default**; set
`OLLAMAQ_MCP_ALLOW_VERIFIER=1` to allow them only if you trust the client. Do not
expose the MCP server to an untrusted network.

Licensed under the [MIT License](LICENSE).

## Hybrid workflow (any premium agent)

1. **Offload** bulk work via CLI (`ollamaq submit` / `run`) or MCP `queue_submit`.
2. Ensure one **daemon** is running (`ollamaq daemon` or `scripts/start_daemon.ps1`).
3. **Harvest** only the digest (`ollamaq digest` / MCP `queue_digest`) — never bulk
   worker output into the chat.
4. **Score** with `ollamaq score` / MCP `hybrid_score` (tokens, cloud $, GPU Wh).

| Client | How to wire |
|--------|-------------|
| **CLI only** | No MCP required — full `ollamaq` surface |
| **Claude Code** | `.mcp.json` + skills `/offload` · `/harvest` · [CLAUDE.md](CLAUDE.md) |
| **Grok** | `.grok/config.toml` (project) or `clients/grok.mcp.json` |
| **Codex** | `clients/codex.config.toml` → `~/.codex/config.toml` + [AGENTS.md](AGENTS.md) |
| **Cursor / others** | `clients/cursor.mcp.json` or `clients/generic.mcp.json` |

Portable policy: [skills/hybrid/SKILL.md](skills/hybrid/SKILL.md). Claude-only extras:
optional `UserPromptSubmit` hook (`hooks/auto_offload.py`) and `.claude/skills/`.

## Install

From source (development, with test deps):

```pwsh
py -m pip install -e ".[dev]"
```

Or build and install a wheel (what a release ships):

```pwsh
py -m build                                  # -> dist/ollamaq-<ver>-py3-none-any.whl
py -m pip install dist/ollamaq-*.whl
```

Console scripts: **`ollamaq`** (CLI), **`ollamaq-mcp`** (`python -m ollamaq.mcp_server`),
and **`ollamaq-mcp-grok`** (optional Grok-oriented aliases).

## Configure

All model routing lives in [config/models.yaml](config/models.yaml) — the single
source of truth. No module hardcodes model names. Resolution order for the config
file: `--config` flag → `$OLLAMAQ_CONFIG` → `$OLLAMAQ_HOME/config/models.yaml` →
`./config/models.yaml`.

**Ollama endpoint:** `base_url` comes from `models.yaml`, but `$OLLAMAQ_BASE_URL`
overrides it at runtime without editing the file — used by the Docker setup to
reach the host's Ollama (`http://host.docker.internal:11434`), and equally good
for pointing at a **remote Ollama** (a VPS, another machine on your LAN or
tailnet). `https://` URLs work as-is.

**Remote auth:** Ollama's API has no built-in authentication — never expose the
raw port to the internet. Either tunnel (SSH/VPN/Tailscale) and keep `base_url`
private, or put an authenticating reverse proxy in front and set
`$OLLAMAQ_API_KEY` (or `ollama.api_key` in models.yaml): every request then
carries `Authorization: Bearer <key>`. Prefer the env var so the secret never
lands in a committed file.

## Deploy

Two turnkey ways to run the queue daemon beyond a bare `ollamaq daemon`:

- **Docker** — [`Dockerfile`](Dockerfile) + [`docker-compose.yml`](docker-compose.yml)
  run the daemon in a container against an Ollama server on the host. `docker compose
  up --build` starts it; it mounts `./config` (read-only) and `./queue` / `./results`
  / `./cache` so you still `ollamaq submit` from the host. The image bundles only the
  worker — not Ollama or any models. A `HEALTHCHECK` uses `ollamaq doctor`.
- **Pinokio** — root-level [`pinokio.js`](pinokio.js) + `install.js` / `start.js` /
  `stop.js` / `update.js` / `reset.js` give a 1-click GUI install and daemon launcher
  for non-CLI users. Point Pinokio at the repo URL; the menu is state-aware
  (Install → Start/Update/Reset → Stop). Needs a local Ollama with the configured
  models pulled.

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
ollamaq run examples/embed_docs.job.json          # embed mode (vectors for RAG/search)
ollamaq run big.job.json --stream                 # constant-memory streaming for huge inputs

# Queue + daemon: enqueue jobs, drain them 24/7 in the background
ollamaq submit examples/clean_contacts.job.json --job-id nightly-clean
ollamaq daemon                  # foreground; drains the queue (run ONE per machine)
ollamaq status                  # counts per state (pending/running/done/failed)
ollamaq watch                   # live dashboard: queue state + savings scoreboard
ollamaq status nightly-clean    # one job's state + resolved model + error
ollamaq cancel nightly-clean    # remove a still-pending job
ollamaq digest nightly-clean    # print results/<job_id>/digest.md
ollamaq score                   # scoreboard: local tokens, cloud $ avoided, GPU Wh/$
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
Cache lives under `cache/` (override with `OLLAMAQ_CACHE`); it's bounded to
`OLLAMAQ_CACHE_MAX_ENTRIES` entries (default 10000, oldest evicted; 0 = unbounded).

`retry.escalate_to` gives one attempt on a stronger local model (or optional
`cloud:grok`) after schema failures. `--sample N` projects full-job ETA from the
first N items. `input.kind: images` sends files to a vision model (`task_tag: vision`
→ `gemma3:4b`); see `examples/invoice_vision.job.json`.

## MCP — use from Claude, Codex, Grok, and other MCP clients

```text
python -m ollamaq.mcp_server     # standard tools (Claude, Codex, Cursor, …)
python -m ollamaq.mcp_grok       # + hybrid_* aliases (Grok-oriented)
```

Tools: `ollama_quick`, `queue_submit`, `queue_status`, `queue_digest`,
`recommend_models`, `hybrid_score`, `hybrid_energy`, `hybrid_policy` (Grok also
gets `hybrid_offload` / `hybrid_digest` / …). Per-client drop-ins:
[clients/README.md](clients/README.md) · full write-up [MCP.md](MCP.md).

## Energy scoreboard

```pwsh
# optional rates (defaults ship in config/energy.yaml)
# usd_per_kwh / g_co2_per_kwh / label

ollamaq run examples/energy_smoke.job.json --job-id energy-smoke
ollamaq score          # tokens + cloud $ avoided + Wh + local $ + CO2 + net
ollamaq history        # per-run Wh column when metered
```

### Running the daemon under Windows Task Scheduler

The daemon is a foreground process; to run it unattended, launch it with `pythonw`
(no console window) and redirect its log:

```pwsh
# One-off / at logon. Runs ONE daemon per machine (single GPU, one model resident).
pythonw -m ollamaq.cli daemon --config config/models.yaml *> daemon.log
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

Contract rules: (1) model names appear only in `models.yaml`; (2) `client.py` is
the sole HTTP path to Ollama; (3) `bench.bench_one` never raises; (4) worker
output must never flood a premium agent's context — only digests.
