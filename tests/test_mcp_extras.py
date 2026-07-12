"""MCP core extras: scoreboard, energy_now, hybrid_policy, path resolution."""

from __future__ import annotations

import json
from pathlib import Path

from ollamaq import mcp_core
from ollamaq.ledger import append_entry


def test_hybrid_policy_text() -> None:
    text = mcp_core.hybrid_policy_text()
    assert "OFFLOAD" in text
    assert "digest" in text.lower()


def test_scoreboard(tmp_path: Path) -> None:
    append_entry(tmp_path, {
        "job_id": "j1", "name": "n", "mode": "map", "model": "m",
        "items": 2, "ok": 2, "failed": 0, "cached": 0, "escalated": 0,
        "wall_seconds": 5.0, "prompt_tokens": 100, "completion_tokens": 50,
        "energy_wh": 0.5, "avg_power_w": 120.0,
        "estimated_cloud_cost_usd": {"claude-opus-4-8": 0.01},
    })
    out = mcp_core.scoreboard(tmp_path)
    assert "scoreboard" in out
    assert "0.500" in out or "0.5" in out
    assert "local tokens" in out


def test_energy_now_shape() -> None:
    raw = mcp_core.energy_now()
    data = json.loads(raw)
    assert "available" in data
    if data["available"]:
        assert "watts" in data


def test_resolve_root_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMAQ_QUEUE", str(tmp_path / "q"))
    p = mcp_core.resolve_root("OLLAMAQ_QUEUE", "queue")
    assert p == tmp_path / "q"
