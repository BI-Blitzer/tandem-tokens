"""Energy rates + scoreboard $ / CO2 lines."""

from __future__ import annotations

from pathlib import Path

from ollamaq.energy_cost import EnergyRates, cost_for_wh, load_energy_rates
from ollamaq.ledger import summarize


def test_cost_for_wh() -> None:
    rates = EnergyRates(usd_per_kwh=0.20, g_co2_per_kwh=400.0, label="test")
    # 500 Wh = 0.5 kWh → $0.10, 200 g CO2
    line = cost_for_wh(rates, 500.0)
    assert abs(line.usd - 0.10) < 1e-9
    assert abs(line.co2_g - 200.0) < 1e-9
    assert line.energy_kwh == 0.5


def test_cost_without_co2() -> None:
    rates = EnergyRates(usd_per_kwh=0.16, g_co2_per_kwh=None)
    line = cost_for_wh(rates, 1000.0)
    assert abs(line.usd - 0.16) < 1e-9
    assert line.co2_g is None


def test_load_energy_rates(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "energy.yaml"
    p.write_text("usd_per_kwh: 0.25\ng_co2_per_kwh: 100\nlabel: lab\n", encoding="utf-8")
    monkeypatch.setenv("OLLAMAQ_ENERGY", str(p))
    rates = load_energy_rates()
    assert rates is not None
    assert rates.usd_per_kwh == 0.25
    assert rates.label == "lab"


def test_scoreboard_format_with_rates() -> None:
    entries = [{
        "job_id": "j", "items": 1, "prompt_tokens": 100, "completion_tokens": 50,
        "wall_seconds": 10.0, "energy_wh": 1000.0, "avg_power_w": 200.0,
        "estimated_cloud_cost_usd": {"claude-opus-4-8": 1.0},
    }]
    sb = summarize(entries)
    rates = EnergyRates(usd_per_kwh=0.16, g_co2_per_kwh=380.0, label="grid")
    out = sb.format(headline_model="claude-opus-4-8", energy_rates=rates)
    assert "local energy cost" in out
    assert "0.160000" in out or "$0.16" in out
    assert "CO2e" in out
    assert "net vs claude-opus-4-8" in out
    assert "0.8400" in out or "0.84" in out  # 1.0 - 0.16
