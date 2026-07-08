"""Live daemon dashboard — a read-only view of the queue and the savings
scoreboard, refreshed in place. `ollamaq watch`.

Data gathering (`snapshot`) and rendering (`render`) are pure and testable; the
CLI just loops snapshot -> clear screen -> render -> sleep. ASCII only (the
console is cp1252 on Windows)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .ledger import Scoreboard, read_entries, summarize
from .queue import JobQueue, State


@dataclass
class Snapshot:
    counts: dict[str, int] = field(default_factory=dict)
    running: list[tuple[str, str]] = field(default_factory=list)  # (job_id, model)
    recent: list[dict] = field(default_factory=list)              # last finished entries
    scoreboard: Scoreboard = field(default_factory=Scoreboard)


def snapshot(queue: JobQueue, results_root: str | Path) -> Snapshot:
    """Read current queue state + ledger totals. Never raises on missing dirs."""
    counts = {st.value: len(queue.list_state(st)) for st in State}
    running = [(j.job_id, j.model or "?") for j in queue.list_state(State.running)]
    entries = read_entries(results_root)
    return Snapshot(counts=counts, running=running,
                    recent=entries[-5:], scoreboard=summarize(entries))


def render(snap: Snapshot, now: str | None = None) -> str:
    now = now or datetime.now().strftime("%H:%M:%S")
    c = snap.counts
    lines = [
        f"ollamaq watch  {now}",
        "-" * 52,
        f"queue   pending {c.get('pending',0):<4} running {c.get('running',0):<4} "
        f"done {c.get('done',0):<4} failed {c.get('failed',0):<4}",
    ]
    if snap.running:
        lines.append("running now:")
        for jid, model in snap.running[:8]:
            lines.append(f"  - {jid}  ({model})")
    lines.append("")
    if snap.recent:
        lines.append("recent finished:")
        for e in reversed(snap.recent):
            lines.append(f"  - {e.get('job_id','?'):20} {e.get('ok',0)}/"
                         f"{e.get('items',0)} ok  {e.get('model','?')}")
        lines.append("")
    sb = snap.scoreboard
    lines.append(f"scoreboard: {sb.jobs} jobs, {sb.total_tokens} local tokens run for $0")
    if sb.cost_by_model:
        top = max(sb.cost_by_model.items(), key=lambda kv: kv[1])
        lines.append(f"  cloud spend avoided (top): {top[0]} ~${top[1]:.4f}")
    lines.append("")
    lines.append("(Ctrl-C to stop)")
    return "\n".join(lines) + "\n"
