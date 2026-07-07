"""Cumulative savings ledger — the running scoreboard across all jobs.

Each finished job appends one JSON line to `<results_root>/ledger.jsonl` (token
counts, item counts, wall time, and the per-model estimated cloud cost avoided).
`ollamaq score` reads the ledger and prints totals: how many local tokens ran for
$0 and how much cloud spend that would have been.

Append is atomic-enough for the single-daemon model (one writer); each line is
self-contained so a partial final line never corrupts earlier entries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

LEDGER_NAME = "ledger.jsonl"


def append_entry(results_root: str | Path, entry: dict) -> None:
    """Append one job's summary to the ledger (adds a UTC timestamp)."""
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    with (root / LEDGER_NAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_entries(results_root: str | Path) -> list[dict]:
    """All ledger entries, skipping any unparseable (e.g. partial) line."""
    p = Path(results_root) / LEDGER_NAME
    if not p.is_file():
        return []
    entries = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


@dataclass
class Scoreboard:
    jobs: int = 0
    items: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0
    cost_by_model: dict[str, float] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def format(self, headline_model: str | None = None) -> str:
        if self.jobs == 0:
            return "no jobs recorded yet — run some jobs, then `ollamaq score`."
        # ASCII only — printed to the console (cp1252 on Windows).
        lines = [
            f"ollamaq scoreboard - {self.jobs} job(s), {self.items} items",
            f"  local tokens: {self.prompt_tokens} in / {self.completion_tokens} out "
            f"({self.total_tokens} total) - ran on local GPU for $0",
            f"  local wall time: {self.wall_seconds/60:.1f} min",
            "  estimated cloud cost avoided (map-phase, approx):",
        ]
        # headline model first, then the rest by cost desc
        models = sorted(self.cost_by_model,
                        key=lambda m: (m != headline_model, -self.cost_by_model[m]))
        for m in models:
            lines.append(f"    {m}: ~${self.cost_by_model[m]:.4f}")
        return "\n".join(lines)


def summarize(entries: list[dict]) -> Scoreboard:
    """Aggregate ledger entries into a Scoreboard."""
    sb = Scoreboard()
    for e in entries:
        sb.jobs += 1
        sb.items += int(e.get("items", 0))
        sb.prompt_tokens += int(e.get("prompt_tokens", 0))
        sb.completion_tokens += int(e.get("completion_tokens", 0))
        sb.wall_seconds += float(e.get("wall_seconds", 0.0))
        for model, usd in (e.get("estimated_cloud_cost_usd") or {}).items():
            sb.cost_by_model[model] = sb.cost_by_model.get(model, 0.0) + float(usd)
    return sb
