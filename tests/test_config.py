"""Contract tests for config loading and model routing.

These pin resolve_model() exactly - the routing rules a lower model must not break.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from ollamaq.config import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_CONFIG,
    Config,
    load_config,
    normalize_model_name,
)
from ollamaq.errors import (
    ConfigError,
    ModelDisabledError,
    UnknownModelError,
    UnknownTaskTagError,
)


def test_normalize_model_name() -> None:
    assert normalize_model_name("gemma2:9b") == "gemma2:9b"
    assert normalize_model_name("qwen2.5-14b-16k") == "qwen2.5-14b-16k:latest"


def test_load_from_explicit_path(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert cfg.defaults.model == "qwen2.5-14b-16k"
    assert "gemma2:9b" in cfg.models


def test_load_from_env(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_CONFIG, str(config_file))
    cfg = load_config()
    assert cfg.routing["clean"] == "qwen2.5-14b-16k"


def test_base_url_env_override(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # containers point at the host's Ollama without editing models.yaml
    monkeypatch.setenv(ENV_BASE_URL, "http://host.docker.internal:11434")
    cfg = load_config(config_file)
    assert cfg.ollama.base_url == "http://host.docker.internal:11434"


def test_base_url_unset_keeps_config_value(config_file: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_BASE_URL, raising=False)
    cfg = load_config(config_file)
    assert cfg.ollama.base_url == "http://localhost:11434"


def test_api_key_env_override(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # authenticate to a reverse proxy in front of a remote Ollama
    monkeypatch.setenv(ENV_API_KEY, "tok-123")
    cfg = load_config(config_file)
    assert cfg.ollama.api_key == "tok-123"


def test_api_key_default_is_none(config_file: Path,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    cfg = load_config(config_file)
    assert cfg.ollama.api_key is None


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no models.yaml"):
        load_config(tmp_path / "nope.yaml")


def test_bad_yaml_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("ollama: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


# These call model_validate directly, so they see pydantic's native
# ValidationError. load_config() is the layer that wraps it into ConfigError
# (covered by test_bad_yaml_raises and the load_config tests above).
def test_load_config_wraps_validation_error(tmp_path: Path) -> None:
    p = tmp_path / "bad_schema.yaml"
    p.write_text(
        "defaults: {model: ghost}\nmodels: {a: {}}\nrouting: {}\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="invalid config"):
        load_config(p)


def test_routing_to_undefined_model_rejected() -> None:
    with pytest.raises(ValidationError, match="undefined model"):
        Config.model_validate(
            {
                "defaults": {"model": "a"},
                "models": {"a": {}},
                "routing": {"clean": "ghost"},
            }
        )


def test_default_model_must_exist() -> None:
    with pytest.raises(ValidationError, match="defaults.model"):
        Config.model_validate(
            {"defaults": {"model": "ghost"}, "models": {"a": {}}, "routing": {}}
        )


def test_unknown_extra_key_rejected() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate(
            {
                "defaults": {"model": "a"},
                "models": {"a": {}},
                "routing": {},
                "surprise": 1,
            }
        )


# ---- resolve_model: the routing contract -------------------------------------


def test_resolve_explicit_enabled(valid_config: Config) -> None:
    assert valid_config.resolve_model("gemma2:9b", None) == "gemma2:9b"


def test_resolve_explicit_unknown(valid_config: Config) -> None:
    with pytest.raises(UnknownModelError):
        valid_config.resolve_model("does-not-exist", None)


def test_resolve_explicit_disabled(valid_config: Config) -> None:
    with pytest.raises(ModelDisabledError):
        valid_config.resolve_model("gemma2:27b", None)


def test_resolve_auto_uses_routing(valid_config: Config) -> None:
    assert valid_config.resolve_model("auto", "classify") == "gemma2:9b"


def test_resolve_none_uses_routing(valid_config: Config) -> None:
    assert valid_config.resolve_model(None, "clean") == "qwen2.5-14b-16k"


def test_resolve_auto_without_tag(valid_config: Config) -> None:
    with pytest.raises(ConfigError, match="requires a task_tag"):
        valid_config.resolve_model("auto", None)


def test_resolve_auto_unknown_tag(valid_config: Config) -> None:
    with pytest.raises(UnknownTaskTagError):
        valid_config.resolve_model("auto", "no-such-tag")
