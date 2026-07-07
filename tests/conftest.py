"""Shared fixtures. A valid in-memory config and a temp config file on disk."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ollamaq.config import Config


VALID_YAML = textwrap.dedent(
    """
    ollama:
      base_url: http://localhost:11434
      timeout_s: 300
      keep_alive: 30m
    defaults:
      model: qwen2.5-14b-16k
      temperature: 0.1
    models:
      qwen2.5-14b-16k:
        max_parallel: 2
        num_ctx: 16384
      "gemma2:9b":
        max_parallel: 4
        num_ctx: 8192
      "gemma2:27b":
        max_parallel: 1
        num_ctx: 8192
        enabled: false
    routing:
      clean: qwen2.5-14b-16k
      classify: "gemma2:9b"
    """
).strip()


@pytest.fixture
def valid_config() -> Config:
    import yaml

    return Config.model_validate(yaml.safe_load(VALID_YAML))


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    p = tmp_path / "config" / "models.yaml"
    p.parent.mkdir(parents=True)
    p.write_text(VALID_YAML, encoding="utf-8")
    return p
