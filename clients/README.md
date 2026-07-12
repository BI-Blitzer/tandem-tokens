# Cross-client MCP configs

**One server, many wrappers.** Every client speaks the same MCP stdio protocol.
Only the *config file shape* changes. Pick a row, paste the snippet, restart the
client, call `recommend_models` or `hybrid_policy` to verify.

| Client | Config file | Command | Drop-in |
|--------|-------------|---------|---------|
| **Claude Code** | project `.mcp.json` | `python -m ollamaq.mcp_server` | [claude.mcp.json](claude.mcp.json) (repo already has `.mcp.json`) |
| **Claude Desktop** | `%APPDATA%\Claude\claude_desktop_config.json` | same | [claude.mcp.json](claude.mcp.json) |
| **OpenAI Codex** | `~/.codex/config.toml` | same | [codex.config.toml](codex.config.toml) |
| **xAI Grok (Build TUI)** | project `.grok/config.toml` | `mcp_server` + `mcp_grok` | already in-repo; `/mcps` refresh |
| **xAI Grok (other)** | client JSON MCP block | `python -m ollamaq.mcp_grok` | [grok.mcp.json](grok.mcp.json) |
| **Cursor** | `.cursor/mcp.json` or Cursor Settings → MCP | `python -m ollamaq.mcp_server` | [cursor.mcp.json](cursor.mcp.json) |
| **VS Code / Continue / Cline / Windsurf** | each product’s MCP JSON | same as Claude | [generic.mcp.json](generic.mcp.json) |
| **Any CLI / custom** | N/A | `ollamaq` CLI or `ollamaq-mcp` | no MCP required |

## Hybrid skill (shared prompt)

Same policy for every premium orchestrator:

- Portable: [`../skills/hybrid/SKILL.md`](../skills/hybrid/SKILL.md)
- Claude Code: [`.claude/skills/hybrid/SKILL.md`](../.claude/skills/hybrid/SKILL.md)
- Codex / multi-agent: [`../AGENTS.md`](../AGENTS.md) (points at the skill + CLI)

Paste the skill body into a system/custom instruction if your client has no
skill loader — the contract is identical.

## Which entry point?

| Entry | When |
|-------|------|
| `python -m ollamaq.mcp_server` / `ollamaq-mcp` | Claude, Codex, Cursor, generic — classic tool names |
| `python -m ollamaq.mcp_grok` / `ollamaq-mcp-grok` | Grok (or anyone who prefers `hybrid_*` aliases) |

Both expose `hybrid_score`, `hybrid_energy`, `hybrid_policy`. The Grok facade
adds aliases (`hybrid_offload` = `queue_submit`, etc.).

## Zero-config rule

Editable install (`pip install -e .` from this repo): **omit `env`**. The server
auto-detects the project root. Set `OLLAMAQ_HOME` only for wheel installs or
when the client launches with a cwd outside the project.

## Verify any client

```text
1. pip install -e .          # package + ollamaq / ollamaq-mcp on PATH
2. ollamaq doctor            # Ollama + models.yaml OK
3. ollamaq daemon            # one per machine, drains the queue
4. From the client: recommend_models  (or hybrid_policy)
5. queue_submit a tiny job → queue_digest when done
```

Full write-up: [MCP.md](../MCP.md).
