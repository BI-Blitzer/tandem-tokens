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
    energy_wh: float = 0.0
    energy_jobs: int = 0  # jobs that reported energy metering
    avg_power_samples: float = 0.0  # running sum of avg_power_w for mean

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def energy_kwh(self) -> float:
        return self.energy_wh / 1000.0

    def format(
        self,
        headline_model: str | None = None,
        energy_rates: object | None = None,
    ) -> str:
        """Render scoreboard. ``energy_rates`` is optional EnergyRates for $/CO2."""
        if self.jobs == 0:
            return "no jobs recorded yet — run some jobs, then `ollamaq score`."
        # ASCII only — printed to the console (cp1252 on Windows).
        lines = [
            f"ollamaq scoreboard - {self.jobs} job(s), {self.items} items",
            f"  local tokens: {self.prompt_tokens} in / {self.completion_tokens} out "
            f"({self.total_tokens} total) - cloud API $0 on map phase",
            f"  local wall time: {self.wall_seconds/60:.1f} min",
        ]
        energy_usd = None
        co2_g = None
        if self.energy_jobs:
            mean_w = (self.avg_power_samples / self.energy_jobs
                      if self.energy_jobs else 0.0)
            lines.append(
                f"  local energy: {self.energy_wh:.3f} Wh "
                f"({self.energy_kwh:.5f} kWh) across {self.energy_jobs} metered job(s)"
            )
            lines.append(f"  mean GPU power (metered jobs): {mean_w:.0f} W")
            if energy_rates is not None:
                from .energy_cost import EnergyRates, cost_for_wh
                if isinstance(energy_rates, EnergyRates):
                    line = cost_for_wh(energy_rates, self.energy_wh)
                    energy_usd = line.usd
                    co2_g = line.co2_g
                    lines.append(
                        f"  local energy cost: ~${line.usd:.6f} "
                        f"@ ${line.usd_per_kwh:.4f}/kWh ({line.label})"
                    )
                    if co2_g is not None:
                        lines.append(f"  local CO2e: ~{co2_g:.2f} g "
                                     f"({co2_g/1000.0:.5f} kg)")
        else:
            lines.append(
                "  local energy: (no nvidia-smi samples yet — "
                "power metering starts on the next run)"
            )
        lines.append("  estimated cloud cost avoided (map-phase, approx):")
        # headline model first, then the rest by cost desc
        models = sorted(self.cost_by_model,
                        key=lambda m: (m != headline_model, -self.cost_by_model[m]))
        headline_usd = None
        for m in models:
            usd = self.cost_by_model[m]
            lines.append(f"    {m}: ~${usd:.4f}")
            if m == headline_model:
                headline_usd = usd
        if headline_usd is None and models:
            headline_usd = self.cost_by_model[models[0]]
        if headline_usd is not None and energy_usd is not None:
            net = headline_usd - energy_usd
            lines.append(
                f"  net vs {headline_model or 'top cloud'} "
                f"(cloud avoided - local energy): ~${net:.4f}"
            )
        return "\n".join(lines)


def format_history(entries: list[dict], limit: int = 20) -> str:
    """Readable run history: when / where (host + input) / what (mode+model) /
    token counts / energy. Most recent last. ASCII (console is cp1252 on Windows)."""
    rows = entries[-limit:] if limit > 0 else entries
    if not rows:
        return "no runs recorded yet."
    header = (f"{'when (UTC)':20} {'host':12} {'input':22} {'mode/model':24} "
              f"{'tok in/out':>12} {'Wh':>8}")
    lines = [header, "-" * len(header)]
    for e in rows:
        when = str(e.get("ts", ""))[:19]
        host = str(e.get("host", "") or "-")[:11]
        inp = str(e.get("input", "") or "-")[:21]
        what = f"{e.get('mode','?')}/{e.get('model','?')}"[:23]
        tok = f"{e.get('prompt_tokens',0)}/{e.get('completion_tokens',0)}"
        if e.get("energy_wh") is not None:
            wh = f"{float(e['energy_wh']):.2f}"
        else:
            wh = "-"
        lines.append(f"{when:20} {host:12} {inp:22} {what:24} {tok:>12} {wh:>8}")
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
        if e.get("energy_wh") is not None:
            sb.energy_wh += float(e["energy_wh"])
            sb.energy_jobs += 1
            if e.get("avg_power_w") is not None:
                sb.avg_power_samples += float(e["avg_power_w"])
    return sb
