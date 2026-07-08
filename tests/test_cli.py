"""Contract tests for the CLI surface, driven by typer's CliRunner."""

from __future__ import annotations

from pathlib import Path

import httpx
import respx
from typer.testing import CliRunner

from ollamaq import __version__
from ollamaq.cli import app

runner = CliRunner()
BASE = "http://localhost:11434"


def _all_output(res) -> str:
    """Combined stdout+stderr. click>=8.2 captures stderr separately; errors
    written with err=True land there, so tests must look at both streams."""
    err = ""
    try:
        err = res.stderr
    except (ValueError, AttributeError):
        err = ""
    return (res.stdout or "") + (err or "")


def test_version() -> None:
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert __version__ in res.stdout


def test_models_lists_routing(config_file: Path) -> None:
    res = runner.invoke(app, ["models", "--config", str(config_file)])
    assert res.exit_code == 0
    assert "clean" in res.stdout
    assert "qwen2.5-14b-16k" in res.stdout
    assert "[disabled]" in res.stdout  # gemma2:27b


def test_doctor_missing_config_exits_2(tmp_path: Path) -> None:
    res = runner.invoke(app, ["doctor", "--config", str(tmp_path / "nope.yaml")])
    assert res.exit_code == 2


@respx.mock
def test_doctor_unreachable_exits_1(config_file: Path) -> None:
    respx.get(f"{BASE}/api/version").mock(side_effect=httpx.ConnectError("down"))
    res = runner.invoke(app, ["doctor", "--config", str(config_file)])
    assert res.exit_code == 1
    assert "UNREACHABLE" in _all_output(res)


@respx.mock
def test_doctor_healthy_when_models_present(config_file: Path) -> None:
    # Ollama reports explicit :latest tags; config names them bare. Coverage
    # must match through normalization.
    respx.get(f"{BASE}/api/version").mock(return_value=httpx.Response(200, json={"version": "1"}))
    respx.get(f"{BASE}/api/tags").mock(
        return_value=httpx.Response(
            200,
            json={"models": [{"name": "qwen2.5-14b-16k:latest"}, {"name": "gemma2:9b"}]},
        )
    )
    res = runner.invoke(app, ["doctor", "--config", str(config_file)])
    assert res.exit_code == 0
    assert "all enabled models are installed" in _all_output(res)


@respx.mock
def test_doctor_warns_missing_model_exits_1(config_file: Path) -> None:
    respx.get(f"{BASE}/api/version").mock(return_value=httpx.Response(200, json={"version": "1"}))
    respx.get(f"{BASE}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "gemma2:9b"}]})
    )
    res = runner.invoke(app, ["doctor", "--config", str(config_file)])
    assert res.exit_code == 1
    assert "not installed" in _all_output(res)
