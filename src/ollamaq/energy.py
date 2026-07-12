"""GPU energy metering via nvidia-smi.

Samples instantaneous power draw at job start/end (and optionally mid-run) and
estimates watt-hours consumed. Used by the scoreboard so hybrid runs show both
cloud $ avoided and local kWh used — the dual cost of "ran it here instead."

No pynvml dependency: reuses the same nvidia-smi locator as gpu.py. Non-NVIDIA
machines get None samples; energy fields are simply omitted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .gpu import _find_nvidia_smi, _run


@dataclass
class PowerSample:
    """One nvidia-smi power reading."""

    watts: float
    limit_w: float = 0.0
    temp_c: float = 0.0
    ts: float = 0.0  # time.perf_counter() at sample


@dataclass
class EnergyReport:
    """Estimated energy for a wall-clock interval."""

    samples: int = 0
    avg_power_w: float = 0.0
    peak_power_w: float = 0.0
    energy_wh: float = 0.0
    wall_seconds: float = 0.0
    available: bool = False

    def as_dict(self) -> dict:
        if not self.available:
            return {"available": False}
        return {
            "available": True,
            "samples": self.samples,
            "avg_power_w": round(self.avg_power_w, 2),
            "peak_power_w": round(self.peak_power_w, 2),
            "energy_wh": round(self.energy_wh, 4),
            "wall_seconds": round(self.wall_seconds, 3),
        }


def parse_power_csv(stdout: str) -> Optional[PowerSample]:
    """Parse `nvidia-smi --query-gpu=power.draw,power.limit,temperature.gpu` CSV."""
    first = stdout.strip().splitlines()[0] if stdout.strip() else ""
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 1:
        return None
    try:
        watts = float(parts[0])
    except ValueError:
        # nvidia-smi sometimes prints "[N/A]" when power sensors are unavailable
        return None
    limit = 0.0
    temp = 0.0
    try:
        if len(parts) >= 2 and parts[1] and parts[1].upper() != "[N/A]":
            limit = float(parts[1])
        if len(parts) >= 3 and parts[2] and parts[2].upper() != "[N/A]":
            temp = float(parts[2])
    except ValueError:
        pass
    return PowerSample(watts=watts, limit_w=limit, temp_c=temp, ts=time.perf_counter())


def sample_power() -> Optional[PowerSample]:
    """Current GPU power draw, or None if nvidia-smi / sensors unavailable."""
    exe = _find_nvidia_smi()
    if not exe:
        return None
    out = _run([
        exe,
        "--query-gpu=power.draw,power.limit,temperature.gpu",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return None
    sample = parse_power_csv(out)
    if sample is not None:
        sample.ts = time.perf_counter()
    return sample


def estimate_energy(samples: list[PowerSample], wall_seconds: float) -> EnergyReport:
    """Trapezoid-ish estimate: average of sample watts × wall hours."""
    if not samples or wall_seconds <= 0:
        return EnergyReport(wall_seconds=max(0.0, wall_seconds), available=False)
    watts = [s.watts for s in samples]
    avg = sum(watts) / len(watts)
    peak = max(watts)
    return EnergyReport(
        samples=len(samples),
        avg_power_w=avg,
        peak_power_w=peak,
        energy_wh=avg * (wall_seconds / 3600.0),
        wall_seconds=wall_seconds,
        available=True,
    )


class EnergyMeter:
    """Sample power at start/stop (and optional mid ticks) around a job."""

    def __init__(self) -> None:
        self._samples: list[PowerSample] = []
        self._t0: float = 0.0
        self._t1: float = 0.0

    def start(self) -> "EnergyMeter":
        self._samples.clear()
        self._t0 = time.perf_counter()
        s = sample_power()
        if s is not None:
            self._samples.append(s)
        return self

    def tick(self) -> None:
        """Optional mid-run sample (e.g. between batches)."""
        s = sample_power()
        if s is not None:
            self._samples.append(s)

    def stop(self) -> EnergyReport:
        self._t1 = time.perf_counter()
        s = sample_power()
        if s is not None:
            self._samples.append(s)
        wall = max(0.0, self._t1 - self._t0)
        return estimate_energy(self._samples, wall)
