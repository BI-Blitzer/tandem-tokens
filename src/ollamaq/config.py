"""Configuration loading and model routing.

The routing table in config/models.yaml is the single source of truth for
which local model handles which task. No other module may hardcode model names.

Search order for the config file (first hit wins):
  1. explicit ``path`` argument
  2. $OLLAMAQ_CONFIG (full path to a yaml file)
  3. $OLLAMAQ_HOME/config/models.yaml
  4. ./config/models.yaml (relative to the current working directory)
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from .errors import (
    ConfigError,
    ModelDisabledError,
    UnknownModelError,
    UnknownTaskTagError,
)

ENV_CONFIG = "OLLAMAQ_CONFIG"
ENV_HOME = "OLLAMAQ_HOME"


def normalize_model_name(name: str) -> str:
    """Ollama treats a bare name as its ``:latest`` tag. ``/api/tags`` always
    reports the explicit tag, so compare names through this normalizer.

    >>> normalize_model_name("gemma2:9b")
    'gemma2:9b'
    >>> normalize_model_name("qwen2.5-14b-16k")
    'qwen2.5-14b-16k:latest'
    """
    return name if ":" in name else f"{name}:latest"


class OllamaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = "http://localhost:11434"
    timeout_s: float = 300.0
    keep_alive: str = "30m"


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_parallel: int = 1
    num_ctx: int = 8192
    enabled: bool = True
    notes: str = ""


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float = 0.1


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ollama: OllamaSettings = OllamaSettings()
    defaults: Defaults
    models: dict[str, ModelEntry]
    routing: dict[str, str]

    @model_validator(mode="after")
    def _check_references(self) -> "Config":
        if self.defaults.model not in self.models:
            raise ValueError(
                f"defaults.model {self.defaults.model!r} is not defined under models:"
            )
        for tag, target in self.routing.items():
            if target not in self.models:
                raise ValueError(f"routing.{tag} points to undefined model {target!r}")
        return self

    def resolve_model(self, model: str | None, task_tag: str | None) -> str:
        """Resolve which model a job runs on.

        Rules (exact, in order):
          1. An explicit model name (anything except None/"auto") must exist in
             ``models`` and be enabled -> returned as-is.
             Raises UnknownModelError / ModelDisabledError otherwise.
          2. None/"auto" requires a task_tag; the tag is looked up in ``routing``.
             Raises ConfigError if no tag, UnknownTaskTagError if unrouted,
             ModelDisabledError if the routed target is disabled.
        """
        if model and model != "auto":
            entry = self.models.get(model)
            if entry is None:
                raise UnknownModelError(f"model {model!r} is not defined in models.yaml")
            if not entry.enabled:
                raise ModelDisabledError(
                    f"model {model!r} is disabled in models.yaml (opt-in required)"
                )
            return model
        if not task_tag:
            raise ConfigError("model='auto' requires a task_tag to route on")
        target = self.routing.get(task_tag)
        if target is None:
            raise UnknownTaskTagError(
                f"task_tag {task_tag!r} has no routing entry in models.yaml"
            )
        if not self.models[target].enabled:
            raise ModelDisabledError(
                f"routing for {task_tag!r} targets disabled model {target!r}"
            )
        return target


def config_search_paths() -> list[Path]:
    """The candidate config locations, in priority order (see module docstring)."""
    paths: list[Path] = []
    env = os.environ.get(ENV_CONFIG)
    if env:
        paths.append(Path(env))
    home = os.environ.get(ENV_HOME)
    if home:
        paths.append(Path(home) / "config" / "models.yaml")
    paths.append(Path.cwd() / "config" / "models.yaml")
    return paths


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the config. Raises ConfigError on any failure."""
    candidates = [Path(path)] if path else config_search_paths()
    for cand in candidates:
        if cand.is_file():
            try:
                raw = yaml.safe_load(cand.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise ConfigError(f"invalid YAML in {cand}: {exc}") from exc
            try:
                return Config.model_validate(raw)
            except ValidationError as exc:
                raise ConfigError(f"invalid config {cand}: {exc}") from exc
    searched = ", ".join(str(c) for c in candidates)
    raise ConfigError(f"no models.yaml found (searched: {searched})")
