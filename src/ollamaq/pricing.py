"""Estimate the cloud spend a local map run avoided.

Loads config/pricing.yaml (USD per 1M tokens per cloud model) and, given the
local run's actual prompt/completion token totals, computes what the same tokens
would have cost on each cloud model. This is the number the digest surfaces so
savings are visible per-run instead of theoretical.

Caveat baked into the wording everywhere: cloud tokenizers differ from local
ones, so this is an estimate of the MAP-PHASE cost avoided, not a guarantee and
not a whole-workflow figure (the Claude reduce pass still costs real tokens).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from .errors import ConfigError

ENV_PRICING = "OLLAMAQ_PRICING"


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_mtok: float
    output_per_mtok: float
    label: str = ""
    verify: bool = False


class Pricing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_compare: str
    models: dict[str, ModelPrice]

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """USD the given cloud model would have charged for these token counts."""
        p = self.models[model]
        return (
            prompt_tokens / 1e6 * p.input_per_mtok
            + completion_tokens / 1e6 * p.output_per_mtok
        )


class CostRow(BaseModel):
    model: str
    label: str
    usd: float
    verify: bool


def estimate_all(
    pricing: Pricing, prompt_tokens: int, completion_tokens: int
) -> list[CostRow]:
    """One CostRow per priced model, most expensive first (default_compare pinned
    to the front so the digest can headline it)."""
    rows = [
        CostRow(
            model=name,
            label=p.label or name,
            usd=round(pricing.estimate(name, prompt_tokens, completion_tokens), 4),
            verify=p.verify,
        )
        for name, p in pricing.models.items()
    ]
    rows.sort(key=lambda r: (r.model != pricing.default_compare, -r.usd))
    return rows


def pricing_search_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get(ENV_PRICING)
    if env:
        paths.append(Path(env))
    home = os.environ.get("OLLAMAQ_HOME")
    if home:
        paths.append(Path(home) / "config" / "pricing.yaml")
    paths.append(Path.cwd() / "config" / "pricing.yaml")
    return paths


def load_pricing(path: str | Path | None = None) -> Pricing | None:
    """Load pricing config. Returns None if no file is found (pricing is optional
    — a missing file just omits the savings block from the digest). Raises
    ConfigError only when a file exists but is invalid."""
    candidates = [Path(path)] if path else pricing_search_paths()
    for cand in candidates:
        if cand.is_file():
            try:
                raw = yaml.safe_load(cand.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise ConfigError(f"invalid YAML in {cand}: {exc}") from exc
            try:
                p = Pricing.model_validate(raw)
            except ValidationError as exc:
                raise ConfigError(f"invalid pricing {cand}: {exc}") from exc
            if p.default_compare not in p.models:
                raise ConfigError(
                    f"default_compare {p.default_compare!r} is not defined under models:"
                )
            return p
    return None
