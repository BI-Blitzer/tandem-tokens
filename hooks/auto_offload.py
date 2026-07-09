"""UserPromptSubmit hook: nudge Claude to offload bulk-data tasks to tandem-tokens.

Claude Code runs this before each user prompt is processed, passing a JSON event
on stdin ({"prompt": "...", ...}). If the prompt looks like a bulk mechanical
data task, we emit `additionalContext` so Claude sees a suggestion to use
/offload. Otherwise we stay silent (exit 0, no output) — the prompt proceeds
unchanged.

Register in .claude/settings.json (see that file). Fully transparent auto-EXECUTION
isn't possible via hooks (a prompt isn't a validated job spec), so this surfaces
the opportunity rather than silently running it — Claude still decides.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # never block on malformed input
    prompt = event.get("prompt", "") if isinstance(event, dict) else ""

    try:
        from ollamaq.detect import detect_offloadable
    except Exception:
        return 0  # harness not importable -> stay silent
    suggestion = detect_offloadable(prompt)
    if not suggestion:
        return 0

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": f"[tandem-tokens] {suggestion}",
        }
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
