"""Convert metered GPU Wh into local $ and optional CO2.

Loads config/energy.yaml (optional — missing file just omits $ / CO2 lines).
Pairs with energy.py (nvidia-smi samples) and the ledger scoreboard.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from .errors import ConfigError

ENV_ENERGY = "OLLAMAQ_ENERGY"


class EnergyRates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    usd_per_kwh: float = 0.16
    g_co2_per_kwh: float | None = None
    label: str = "grid"

    @field_validator("usd_per_kwh")
    @classmethod
    def _nonneg_usd(cls, v: float) -> float:
        if v < 0:
            raise ValueError("usd_per_kwh must be >= 0")
        return v

    @field_validator("g_co2_per_kwh")
    @classmethod
    def _nonneg_co2(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("g_co2_per_kwh must be >= 0")
        return v

    def cost_usd(self, energy_wh: float) -> float:
        """USD for this many watt-hours."""
        return (energy_wh / 1000.0) * self.usd_per_kwh

    def co2_g(self, energy_wh: float) -> float | None:
        """Grams CO2e, or None if intensity not configured."""
        if self.g_co2_per_kwh is None:
            return None
        return (energy_wh / 1000.0) * self.g_co2_per_kwh


@dataclass
class EnergyCostLine:
    energy_wh: float
    energy_kwh: float
    usd: float
    co2_g: float | None
    label: str
    usd_per_kwh: float

    def as_dict(self) -> dict:
        d = {
            "energy_wh": round(self.energy_wh, 4),
            "energy_kwh": round(self.energy_kwh, 6),
            "energy_usd": round(self.usd, 6),
            "usd_per_kwh": self.usd_per_kwh,
            "label": self.label,
        }
        if self.co2_g is not None:
            d["co2_g"] = round(self.co2_g, 3)
        return d


def cost_for_wh(rates: EnergyRates, energy_wh: float) -> EnergyCostLine:
    kwh = energy_wh / 1000.0
    return EnergyCostLine(
        energy_wh=energy_wh,
        energy_kwh=kwh,
        usd=rates.cost_usd(energy_wh),
        co2_g=rates.co2_g(energy_wh),
        label=rates.label,
        usd_per_kwh=rates.usd_per_kwh,
    )


def energy_search_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get(ENV_ENERGY)
    if env:
        paths.append(Path(env))
    home = os.environ.get("OLLAMAQ_HOME")
    if home:
        paths.append(Path(home) / "config" / "energy.yaml")
    paths.append(Path.cwd() / "config" / "energy.yaml")
    return paths


def load_energy_rates(path: str | Path | None = None) -> EnergyRates | None:
    """Load energy rates. None if no file (optional). Raises ConfigError if invalid."""
    candidates = [Path(path)] if path else energy_search_paths()
    for cand in candidates:
        if cand.is_file():
            try:
                raw = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"invalid YAML in {cand}: {exc}") from exc
            try:
                return EnergyRates.model_validate(raw)
            except ValidationError as exc:
                raise ConfigError(f"invalid energy config {cand}: {exc}") from exc
    return None
