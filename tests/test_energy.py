"""Energy metering: parse nvidia-smi power CSV and estimate Wh."""

from __future__ import annotations

from ollamaq.energy import (
    EnergyMeter,
    PowerSample,
    estimate_energy,
    parse_power_csv,
)


def test_parse_power_csv() -> None:
    s = parse_power_csv("134.34, 360.00, 57\n")
    assert s is not None
    assert s.watts == 134.34
    assert s.limit_w == 360.0
    assert s.temp_c == 57.0


def test_parse_power_na() -> None:
    assert parse_power_csv("[N/A], [N/A], 40\n") is None
    assert parse_power_csv("") is None


def test_estimate_energy() -> None:
    samples = [PowerSample(watts=100), PowerSample(watts=200)]
    # 150 W avg * (36 s / 3600) = 1.5 Wh
    r = estimate_energy(samples, wall_seconds=36.0)
    assert r.available
    assert r.avg_power_w == 150.0
    assert r.peak_power_w == 200.0
    assert abs(r.energy_wh - 1.5) < 1e-9


def test_estimate_empty() -> None:
    r = estimate_energy([], 10.0)
    assert r.available is False


def test_meter_without_smi(monkeypatch) -> None:
    import ollamaq.energy as em

    monkeypatch.setattr(em, "sample_power", lambda: None)
    m = EnergyMeter().start()
    report = m.stop()
    assert report.available is False
