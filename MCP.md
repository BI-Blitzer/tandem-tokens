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

## One env var configures everything: `OLLAMAQ_HOME`

Because different clients launch the server from different working directories,
point everything at the project with **`OLLAMAQ_HOME`**. The server then uses
`OLLAMAQ_HOME/config/models.yaml`, `OLLAMAQ_HOME/queue`, `OLLAMAQ_HOME/results`.
(Individual `OLLAMAQ_CONFIG` / `OLLAMAQ_QUEUE` / `OLLAMAQ_RESULTS` override if set.)

Assumed project path below: `C:\_Work\Claude_Coded\endless-code`.

---

## Claude Code

Project-scoped `.mcp.json` (already in this repo) or user config. Restart Claude
Code after adding:

```json
{
  "mcpServers": {
    "ollamaq": {
      "command": "python",
      "args": ["-m", "ollamaq.mcp_server"],
      "env": { "OLLAMAQ_HOME": "C:\\_Work\\Claude_Coded\\endless-code" }
    }
  }
}
```

## Claude Desktop

Edit `claude_desktop_config.json`
(`%APPDATA%\Claude\claude_desktop_config.json` on Windows), same shape:

```json
{
  "mcpServers": {
    "ollamaq": {
      "command": "python",
      "args": ["-m", "ollamaq.mcp_server"],
      "env": { "OLLAMAQ_HOME": "C:\\_Work\\Claude_Coded\\endless-code" }
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

[mcp_servers.ollamaq.env]
OLLAMAQ_HOME = "C:\\_Work\\Claude_Coded\\endless-code"
```

## xAI Grok

Grok clients that support MCP use the same stdio launch. For a JSON-config client
(the common shape), use the Claude-style block above verbatim. For a TOML-config
client, use the Codex block. The server command/args/env never change — only the
wrapper file format does.

---

## Verifying any client

1. Ensure the package is importable in the Python the client will launch:
   `pip install -e .` in the project (so `python -m ollamaq.mcp_server` works).
2. Start a daemon so queued jobs actually run: `ollamaq daemon` (one per machine).
3. From the client, call `recommend_models` (no args) — it should return the GPU
   report. Then `queue_submit` a small job and `queue_digest` it once done.

A standalone smoke test (no client needed) that drives the server as an MCP
client is in the repo history under `scratchpad/mcp_extension_test.py`.
