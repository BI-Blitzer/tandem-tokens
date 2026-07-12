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
ENV_BASE_URL = "OLLAMAQ_BASE_URL"
ENV_API_KEY = "OLLAMAQ_API_KEY"


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
    # Sent as "Authorization: Bearer <key>" when set — for a remote Ollama behind
    # an authenticating reverse proxy. Prefer $OLLAMAQ_API_KEY over putting a
    # secret in a yaml that may be committed.
    api_key: str | None = None


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


class GrokCloudSettings(BaseModel):
    """Optional xAI Grok fallback (OpenAI-compatible). Disabled by default."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    base_url: str = "https://api.x.ai/v1"
    # Prefer $XAI_API_KEY / $GROK_API_KEY over committing a key here.
    api_key: str | None = None
    default_model: str = "grok-3-mini"
    timeout_s: float = 120.0


class CloudSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grok: GrokCloudSettings = GrokCloudSettings()


class FallbackSettings(BaseModel):
    """Escalation when local schema repair fails.

    ``escalate_to`` may be a local model name, or a cloud target:
    ``cloud:grok`` or ``cloud:grok/grok-3-mini``. Job-spec ``retry.escalate_to``
    overrides this; use ``"auto"`` in the job to take the config default.
    """

    model_config = ConfigDict(extra="forbid")

    escalate_to: str | None = None
    # Optional per-task overrides (same string forms as escalate_to).
    by_task_tag: dict[str, str] = {}


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ollama: OllamaSettings = OllamaSettings()
    defaults: Defaults
    models: dict[str, ModelEntry]
    routing: dict[str, str]
    cloud: CloudSettings = CloudSettings()
    fallbacks: FallbackSettings = FallbackSettings()

    @model_validator(mode="after")
    def _check_references(self) -> "Config":
        if self.defaults.model not in self.models:
            raise ValueError(
                f"defaults.model {self.defaults.model!r} is not defined under models:"
            )
        for tag, target in self.routing.items():
            if target not in self.models:
                raise ValueError(f"routing.{tag} points to undefined model {target!r}")
        # Local fallback targets must exist; cloud:… is validated at resolve time.
        for label, target in [
            ("fallbacks.escalate_to", self.fallbacks.escalate_to),
            *[
                (f"fallbacks.by_task_tag.{t}", m)
                for t, m in self.fallbacks.by_task_tag.items()
            ],
        ]:
            if target and not target.startswith("cloud:") and target not in self.models:
                raise ValueError(f"{label} points to undefined model {target!r}")
        return self

    def resolve_model(self, model: str | None, task_tag: str | None) -> str:
        """Resolve which *local* model a job runs on.

        Rules (exact, in order):
          1. An explicit model name (anything except None/"auto") must exist in
             ``models`` and be enabled -> returned as-is.
             Raises UnknownModelError / ModelDisabledError otherwise.
             Cloud targets (``cloud:…``) are NOT resolved here — use
             ``resolve_escalate_target``.
          2. None/"auto" requires a task_tag; the tag is looked up in ``routing``.
             Raises ConfigError if no tag, UnknownTaskTagError if unrouted,
             ModelDisabledError if the routed target is disabled.
        """
        if model and model != "auto":
            if model.startswith("cloud:"):
                raise ConfigError(
                    f"cloud target {model!r} cannot be the primary model; "
                    "use it only as retry.escalate_to / fallbacks.escalate_to"
                )
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

    def resolve_escalate_target(
        self, escalate_to: str | None, task_tag: str | None = None
    ) -> str | None:
        """Resolve the escalation target for a job.

        - None / "" → config fallbacks (by_task_tag, then escalate_to)
        - "auto" → same as config fallbacks
        - local model name → must exist and be enabled
        - cloud:grok[/model] → returned as-is (cloud path validates at call time)
        """
        raw = escalate_to
        if not raw or raw == "auto":
            if task_tag and task_tag in self.fallbacks.by_task_tag:
                raw = self.fallbacks.by_task_tag[task_tag]
            else:
                raw = self.fallbacks.escalate_to
        if not raw:
            return None
        if raw.startswith("cloud:"):
            return raw
        # Local model — same rules as resolve_model explicit name.
        return self.resolve_model(raw, None)

    def grok_settings_or_none(self):
        """Return GrokSettings if cloud.grok is enabled and a key is available."""
        from .cloud import GrokSettings, resolve_grok_api_key

        g = self.cloud.grok
        if not g.enabled:
            return None
        key = resolve_grok_api_key(g.api_key)
        if not key:
            return None
        return GrokSettings(
            base_url=g.base_url,
            api_key=key,
            default_model=g.default_model,
            timeout_s=g.timeout_s,
            enabled=True,
        )


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
                cfg = Config.model_validate(raw)
            except ValidationError as exc:
                raise ConfigError(f"invalid config {cand}: {exc}") from exc
            # 12-factor overrides: point at a different Ollama (e.g.
            # host.docker.internal from Docker, or a remote VPS) and authenticate
            # to a reverse proxy in front of it, without editing models.yaml.
            override = os.environ.get(ENV_BASE_URL)
            if override:
                cfg.ollama.base_url = override
            key = os.environ.get(ENV_API_KEY)
            if key:
                cfg.ollama.api_key = key
            return cfg
    searched = ", ".join(str(c) for c in candidates)
    raise ConfigError(f"no models.yaml found (searched: {searched})")
