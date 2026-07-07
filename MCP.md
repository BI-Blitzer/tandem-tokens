# ollamaq MCP server — cross-client setup

The ollamaq MCP server is a **standard stdio MCP server** (built on the official
`mcp` Python SDK / FastMCP). Any MCP-compatible client can use it — Claude Code,
Claude Desktop, OpenAI Codex, xAI Grok, and others — because they all speak the
same protocol. Only the *config file format* differs per client; the server
command is identical everywhere:

```
command: python
args:    ["-m", "ollamaq.mcp_server"]
```

## Tools exposed

| Tool | What it does |
|---|---|
| `ollama_quick(prompt, model?, task_tag?, schema?)` | One-shot local generation, output capped (~2k tokens) so it never floods context. |
| `queue_submit(jobspec, job_id?)` | Enqueue a batch job for the daemon; returns the job id. |
| `queue_status(job_id?)` | Queue state — counts, or one job's detail. |
| `queue_digest(job_id)` | A finished job's ≤8 KB digest (the Claude-facing summary). |
| `recommend_models()` | GPU-aware advice: which local models fit this machine (vs. spill to CPU). |

Bulk work goes through `queue_submit` (run `ollamaq daemon` to drain it); only
`ollama_quick`'s small capped output and the digests ever return to the model's
context — the token-economy contract holds across every client.

## Security — verifier commands over MCP

A job spec's `verifier.cmd` (gen_test mode) runs an **arbitrary local subprocess**.
That's fine from the trusted CLI, but the MCP server may be reachable by a
less-trusted client, so `queue_submit` **refuses any spec with a verifier by
default**. To allow verifier (gen_test) jobs over MCP, set
`OLLAMAQ_MCP_ALLOW_VERIFIER=1` in the server's `env` — and only if you trust the
client. Never expose this server to an untrusted network.

## Finding config, queue, results — usually zero config

The server resolves those roots in this order:

1. explicit `OLLAMAQ_CONFIG` / `OLLAMAQ_QUEUE` / `OLLAMAQ_RESULTS`
2. `$OLLAMAQ_HOME/{config/models.yaml,queue,results}`
3. **auto-detected project root** — from the installed source location (editable
   install: `<root>/src/ollamaq/`), so it works with no configuration at all
4. the current working directory

Because of (3), an **editable install** (`pip install -e .`) needs **no path
anywhere** — the server finds its own project. Set `OLLAMAQ_HOME` only for a
non-editable/wheel install or when you want config/results somewhere specific.

The examples below use `<PROJECT>` for the project path; drop the whole `env`
block entirely if you did an editable install.

---

## Claude Code

Project-scoped `.mcp.json` (already in this repo) — no path needed for an editable
install. Restart Claude Code after adding:

```json
{
  "mcpServers": {
    "ollamaq": { "command": "python", "args": ["-m", "ollamaq.mcp_server"] }
  }
}
```

## Claude Desktop

Edit `claude_desktop_config.json`
(`%APPDATA%\Claude\claude_desktop_config.json` on Windows), same shape. Add an
`env` with `OLLAMAQ_HOME` only if the server can't auto-detect (non-editable install):

```json
{
  "mcpServers": {
    "ollamaq": {
      "command": "python",
      "args": ["-m", "ollamaq.mcp_server"],
      "env": { "OLLAMAQ_HOME": "<PROJECT>" }
    }
  }
}
```

## OpenAI Codex (CLI)

Codex reads `~/.codex/config.toml`. MCP servers go under `[mcp_servers.<name>]`
(TOML, not JSON):

```toml
[mcp_servers.ollamaq]
command = "python"
args = ["-m", "ollamaq.mcp_server"]

# only if not auto-detected:
# [mcp_servers.ollamaq.env]
# OLLAMAQ_HOME = "<PROJECT>"
```

## xAI Grok

Grok clients that support MCP use the same stdio launch. For a JSON-config client
(the common shape), use the Claude-style block above verbatim; for a TOML-config
client, use the Codex block. The server command/args never change — only the
wrapper file format does, and the `env` is optional for editable installs.

---

## Verifying any client

1. Ensure the package is importable in the Python the client will launch:
   `pip install -e .` in the project (so `python -m ollamaq.mcp_server` works).
2. Start a daemon so queued jobs actually run: `ollamaq daemon` (one per machine).
3. From the client, call `recommend_models` (no args) — it should return the GPU
   report. Then `queue_submit` a small job and `queue_digest` it once done.

A standalone smoke test (no client needed) that drives the server as an MCP
client is in the repo history under `scratchpad/mcp_extension_test.py`.
