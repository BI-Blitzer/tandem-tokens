"""Contract tests for pricing load + savings estimation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ollamaq.errors import ConfigError
from ollamaq.pricing import Pricing, estimate_all, load_pricing

YAML = textwrap.dedent(
    """
    default_compare: claude-opus-4-8
    models:
      claude-opus-4-8:
        input_per_mtok: 5.0
        output_per_mtok: 25.0
        label: Claude Opus 4.8
      claude-haiku-4-5:
        input_per_mtok: 1.0
        output_per_mtok: 5.0
        label: Claude Haiku 4.5
      gpt-5:
        input_per_mtok: 1.25
        output_per_mtok: 10.0
        label: OpenAI GPT-5
        verify: true
    """
).strip()


def _pricing() -> Pricing:
    import yaml

    return Pricing.model_validate(yaml.safe_load(YAML))


def test_estimate_math() -> None:
    p = _pricing()
    # 1M in + 1M out on Opus = $5 + $25
    assert p.estimate("claude-opus-4-8", 1_000_000, 1_000_000) == 30.0
    # 100k out only on Haiku = 0.1 * $5 = $0.50
    assert p.estimate("claude-haiku-4-5", 0, 100_000) == 0.5


def test_estimate_all_pins_default_first_then_by_cost() -> None:
    rows = estimate_all(_pricing(), 1_000_000, 1_000_000)
    assert rows[0].model == "claude-opus-4-8"  # default_compare pinned first
    # remaining ordered most-expensive first
    assert [r.model for r in rows[1:]] == ["gpt-5", "claude-haiku-4-5"]
    assert rows[1].verify is True  # gpt-5 flagged


def test_load_missing_returns_none(tmp_path: Path) -> None:
    assert load_pricing(tmp_path / "nope.yaml") is None


def test_load_valid(tmp_path: Path) -> None:
    p = tmp_path / "pricing.yaml"
    p.write_text(YAML, encoding="utf-8")
    loaded = load_pricing(p)
    assert loaded is not None
    assert loaded.default_compare == "claude-opus-4-8"


def test_load_bad_default_compare(tmp_path: Path) -> None:
    p = tmp_path / "pricing.yaml"
    p.write_text("default_compare: ghost\nmodels: {a: {input_per_mtok: 1, output_per_mtok: 2}}\n",
                 encoding="utf-8")
    with pytest.raises(ConfigError, match="default_compare"):
        load_pricing(p)


def test_load_bad_yaml(tmp_path: Path) -> None:
    p = tmp_path / "pricing.yaml"
    p.write_text("default_compare: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_pricing(p)


def test_shipped_pricing_config_is_valid() -> None:
    """The pricing.yaml we ship must load and headline Opus 4.8."""
    loaded = load_pricing(Path("config") / "pricing.yaml")
    assert loaded is not None
    assert "claude-opus-4-8" in loaded.models
    assert loaded.default_compare in loaded.models
