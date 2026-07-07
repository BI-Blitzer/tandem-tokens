"""Contract tests for the verifier subprocess protocol, using real subprocesses."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from ollamaq.errors import VerifierError
from ollamaq.verify import run_verifier


def _script(tmp_path: Path, body: str) -> str:
    p = tmp_path / "verifier.py"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return f'"{sys.executable}" "{p}"'


def test_scores_from_stdin(tmp_path: Path) -> None:
    cmd = _script(tmp_path, """
        import sys, json
        d = json.load(sys.stdin)
        print(json.dumps({"pass": d["n"] >= 5, "score": d["n"] / 10, "notes": "ok"}))
    """)
    v = run_verifier(cmd, {"n": 8})
    assert v.ok and v.passed and v.score == 0.8 and v.notes == "ok"


def test_pass_defaults_false(tmp_path: Path) -> None:
    cmd = _script(tmp_path, """
        import sys, json
        print(json.dumps({"score": 0.3}))
    """)
    v = run_verifier(cmd, {})
    assert v.ok and v.passed is False and v.score == 0.3


def test_missing_score_is_not_ok(tmp_path: Path) -> None:
    cmd = _script(tmp_path, """
        import json
        print(json.dumps({"pass": True}))
    """)
    v = run_verifier(cmd, {})
    assert not v.ok and v.score == 0.0 and "score" in v.notes


def test_bad_json_is_not_ok(tmp_path: Path) -> None:
    cmd = _script(tmp_path, "print('not json at all')")
    v = run_verifier(cmd, {})
    assert not v.ok and v.score == 0.0 and "JSON" in v.notes


def test_no_output_is_not_ok(tmp_path: Path) -> None:
    cmd = _script(tmp_path, "pass")  # prints nothing
    v = run_verifier(cmd, {})
    assert not v.ok and "no stdout" in v.notes


def test_timeout(tmp_path: Path) -> None:
    cmd = _script(tmp_path, """
        import time
        time.sleep(5)
    """)
    v = run_verifier(cmd, {}, timeout_s=0.5)
    assert not v.ok and "timed out" in v.notes


def test_command_not_found_raises() -> None:
    with pytest.raises(VerifierError, match="not found"):
        run_verifier("definitely-not-a-real-command-xyz", {})


def test_empty_command_raises() -> None:
    with pytest.raises(VerifierError, match="empty"):
        run_verifier("   ", {})
