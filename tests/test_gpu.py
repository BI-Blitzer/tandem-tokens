"""Contract tests for GPU detection parsing and model-fit recommendation."""

from __future__ import annotations

import ollamaq.gpu as gpu_mod
from ollamaq.config import Config
from ollamaq.gpu import (GpuInfo, ModelFit, _find_nvidia_smi, parse_nvidia_smi,
                         parse_rocm_smi, recommend)


def _cfg() -> Config:
    return Config.model_validate({
        "defaults": {"model": "qwen2.5-14b-16k"},
        "models": {
            "qwen2.5-14b-16k": {"num_ctx": 16384},
            "gemma2:9b": {"num_ctx": 8192},
            "gemma2:27b": {"num_ctx": 8192, "enabled": False},
        },
        "routing": {"clean": "qwen2.5-14b-16k", "classify": "gemma2:9b"},
    })


def _gpu(total: float, free: float = 10000) -> GpuInfo:
    return GpuInfo(name="RTX 5080", total_mb=total, free_mb=free, driver="1.0")


# installed sizes in MB (as /api/tags would report, minus the :latest matching)
SIZES = {"qwen2.5-14b-16k:latest": 9000, "gemma2:9b": 5400, "gemma2:27b": 15000}


def test_fit_flags_by_budget() -> None:
    rec = recommend(_cfg(), SIZES, _gpu(16303))  # budget ~13857 MB
    by_name = {m.name: m for m in rec.fits}
    assert by_name["qwen2.5-14b-16k"].fits is True   # 9000 < 13857
    assert by_name["gemma2:9b"].fits is True          # 5400 < 13857
    assert by_name["gemma2:27b"].fits is False        # 15000 > 13857 -> spill


def test_smaller_gpu_spills_more() -> None:
    rec = recommend(_cfg(), SIZES, _gpu(8000))  # budget 6800 MB
    by_name = {m.name: m for m in rec.fits}
    assert by_name["gemma2:9b"].fits is True          # 5400 < 6800 -> fits
    assert by_name["qwen2.5-14b-16k"].fits is False   # 9000 > 6800 -> spills


def test_routing_advice_warns_on_spill() -> None:
    rec = recommend(_cfg(), SIZES, _gpu(8000))  # budget 6800; 14b spills
    advice = "\n".join(rec.routing_advice)
    assert "clean" in advice and "spills to CPU" in advice   # routed to 14b, spills
    assert "classify" in advice and "fits" in advice          # routed to 9b, fits


def test_no_gpu_infinite_budget() -> None:
    rec = recommend(_cfg(), SIZES, gpu=None)  # unknown GPU -> everything "fits"
    assert all(m.fits for m in rec.fits if m.size_mb > 0)
    assert "no NVIDIA GPU detected" in rec.format()


def test_bench_tps_merged() -> None:
    bench = {"results": [{"model": "gemma2:9b", "tokens_per_second": 88.0}]}
    rec = recommend(_cfg(), SIZES, _gpu(16303), bench=bench)
    m = next(m for m in rec.fits if m.name == "gemma2:9b")
    assert m.tok_s == 88.0
    assert "88 tok/s" in rec.format()


def test_format_marks_fit_and_spill() -> None:
    out = recommend(_cfg(), SIZES, _gpu(16303)).format()
    assert "FITS" in out and "SPILL" in out
    assert "[disabled]" in out  # gemma2:27b


def test_parse_nvidia_smi() -> None:
    g = parse_nvidia_smi("NVIDIA GeForce RTX 5080, 610.62, 16303, 10097\n")
    assert g.name == "NVIDIA GeForce RTX 5080" and g.total_mb == 16303 and g.free_mb == 10097


def test_parse_nvidia_smi_garbage() -> None:
    assert parse_nvidia_smi("") is None
    assert parse_nvidia_smi("no commas here") is None


def test_parse_rocm_smi() -> None:
    # ~16 GiB card
    out = '{"card0": {"VRAM Total Memory (B)": "17163091968", "VRAM Total Used Memory (B)": "1000000"}}'
    g = parse_rocm_smi(out)
    assert g is not None and g.name == "AMD card0"
    assert round(g.total_mb) == 16368        # 17163091968 / 1024^2


def test_parse_rocm_smi_garbage() -> None:
    assert parse_rocm_smi("not json") is None
    assert parse_rocm_smi('{"card0": {"other": "1"}}') is None  # no VRAM key


def test_find_nvidia_smi_falls_back_when_not_on_path(tmp_path, monkeypatch) -> None:
    # simulate a stripped PATH (which() finds nothing) but a known install path present
    fake = tmp_path / "System32" / "nvidia-smi.exe"
    fake.parent.mkdir(parents=True)
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _: None)
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    assert _find_nvidia_smi() == str(fake)


def test_find_nvidia_smi_none_when_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _: None)
    monkeypatch.setenv("SystemRoot", str(tmp_path))  # no nvidia-smi anywhere
    # (the hardcoded /usr and Program Files paths won't exist in the test sandbox)
    result = _find_nvidia_smi()
    assert result is None or result.endswith("nvidia-smi") or result.endswith("nvidia-smi.exe")
