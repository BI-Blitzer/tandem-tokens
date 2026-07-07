"""GPU detection and model-fit recommendation.

Answers "which of my installed models actually fit on this GPU (vs. spill to CPU
and crawl)?" using: nvidia-smi for VRAM, Ollama /api/tags for on-disk model
sizes, and config/bench.json (if present) for measured throughput.

Helper note: pynvml/GPUtil aren't required — nvidia-smi is parsed directly (it's
always present with the driver). Non-NVIDIA GPUs return None and recommendation
degrades to config-size-only advice.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional

# Fraction of total VRAM a single model may occupy and still run fully on GPU,
# leaving room for the desktop, KV cache, and context. Empirical for a 16 GB card.
USABLE_VRAM_FRACTION = 0.85


@dataclass
class GpuInfo:
    name: str
    total_mb: float
    free_mb: float
    driver: str = ""

    @property
    def budget_mb(self) -> float:
        return self.total_mb * USABLE_VRAM_FRACTION


def detect_nvidia_gpu() -> Optional[GpuInfo]:
    """First NVIDIA GPU via nvidia-smi, or None if unavailable/non-NVIDIA."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    first = out.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 4:
        return None
    try:
        return GpuInfo(name=parts[0], driver=parts[1],
                       total_mb=float(parts[2]), free_mb=float(parts[3]))
    except ValueError:
        return None


@dataclass
class ModelFit:
    name: str
    size_mb: float
    enabled: bool
    fits: bool          # runs fully on GPU when it's the only model loaded
    tok_s: Optional[float] = None  # from bench.json if available


@dataclass
class Recommendation:
    gpu: Optional[GpuInfo]
    fits: list[ModelFit] = field(default_factory=list)
    routing_advice: list[str] = field(default_factory=list)

    def format(self) -> str:
        lines: list[str] = []
        if self.gpu:
            lines.append(f"GPU: {self.gpu.name} - {self.gpu.total_mb:.0f} MB total, "
                         f"{self.gpu.free_mb:.0f} MB free "
                         f"(budget ~{self.gpu.budget_mb:.0f} MB for one model)")
        else:
            lines.append("GPU: no NVIDIA GPU detected (recommending by model size only)")
        lines.append("")
        lines.append("installed models (fit = runs fully on GPU, no CPU spill):")
        for m in self.fits:
            mark = "FITS" if m.fits else "SPILL"
            flag = "" if m.enabled else "  [disabled]"
            tps = f"  {m.tok_s:.0f} tok/s" if m.tok_s else ""
            lines.append(f"  [{mark}] {m.name:24} {m.size_mb:7.0f} MB{tps}{flag}")
        if self.routing_advice:
            lines.append("")
            lines.append("routing advice (task_tag -> model):")
            lines.extend(f"  {a}" for a in self.routing_advice)
        return "\n".join(lines)


def recommend(config, installed_sizes: dict[str, float],
              gpu: Optional[GpuInfo], bench: Optional[dict] = None) -> Recommendation:
    """Build a recommendation. `installed_sizes` maps model name -> MB (from
    Ollama /api/tags). `bench` is the parsed bench.json (optional)."""
    from .config import normalize_model_name

    budget = gpu.budget_mb if gpu else float("inf")
    tps_by_model = {}
    if bench:
        for r in bench.get("results", []):
            tps_by_model[r.get("model")] = r.get("tokens_per_second")

    # match config models to installed sizes by normalized (:latest) name
    size_by_norm = {normalize_model_name(n): mb for n, mb in installed_sizes.items()}

    fits: list[ModelFit] = []
    for name, entry in config.models.items():
        size_mb = size_by_norm.get(normalize_model_name(name), 0.0)
        fits.append(ModelFit(
            name=name, size_mb=size_mb, enabled=entry.enabled,
            fits=(size_mb > 0 and size_mb <= budget),
            tok_s=tps_by_model.get(name),
        ))
    fits.sort(key=lambda m: m.size_mb)

    fit_names = {m.name for m in fits if m.fits}
    advice: list[str] = []
    for tag, model in config.routing.items():
        if model in fit_names:
            advice.append(f"{tag:10} -> {model}  (fits)")
        else:
            # suggest the largest fitting model as a fallback hint
            alt = next((m.name for m in reversed(fits)
                        if m.fits and m.enabled), None)
            hint = f"; consider {alt}" if alt and alt != model else ""
            advice.append(f"{tag:10} -> {model}  (WARNING: spills to CPU{hint})")
    return Recommendation(gpu=gpu, fits=fits, routing_advice=advice)
