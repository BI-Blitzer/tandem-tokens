# Changelog

## 0.2.0 — 2026-07-12

Hybrid multi-agent + energy scoreboard release.

### Highlights
- **Multi-client**: Claude, Grok, Codex, Cursor — MCP configs under `clients/`, project `.grok/config.toml`, portable `skills/hybrid` + `AGENTS.md`
- **Energy metering**: nvidia-smi Wh → digest/ledger; `config/energy.yaml` for $/kWh + optional CO₂; `ollamaq score` net vs cloud
- **Optional cloud residual**: `cloud:grok` escalate path (`XAI_API_KEY`)
- **Live campaign**: map / pipeline / summarize jobs (`jobs/uc*`) + scale (1k contacts, 120 tickets)
- **Landing page**: use-cases section, energy calculator, hybrid MCP tools; OG card “Premium models do the thinking”
- **Real-data drop zone**: `data/real/` + `jobs/real_uc*.job.json`
- **Daemon autostart**: `scripts/install_daemon_startup.ps1` (user Startup folder)

### Docs
- `docs/HYBRID_USE_CASES.md`, `docs/CAMPAIGN_FINDINGS.md`
- GitHub About: multi-agent description, homepage, topics

## 0.1.0

Initial public package: map / gen_test / pipeline / embed, queue daemon, digests, MCP, Claude skills, Docker + Pinokio.
