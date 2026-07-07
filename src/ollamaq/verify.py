"""Run a user verifier subprocess to score a gen_test candidate.

Protocol (documented in PLAN.md section 3):
  - The verifier command receives the candidate's JSON value on STDIN.
  - It must print a JSON verdict object on STDOUT:
        {"pass": bool, "score": float, "notes": str}
    `pass` and `notes` are optional (default False / ""); `score` is required.
  - Exit code is ignored for scoring — the verdict comes from stdout — but a
    non-zero exit with unparseable stdout yields an ok=False verdict scored 0.

Failure policy (robust for long unattended loops):
  - If the command cannot be launched at all (not found, not executable), that's
    a setup error affecting every candidate -> VerifierError (abort the run).
  - If it launches but times out, crashes, or emits bad JSON, that single
    candidate gets Verdict(ok=False, score=0.0) with the reason in `notes`, and
    the loop continues.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any

from .errors import VerifierError


@dataclass
class Verdict:
    ok: bool          # False when the verifier misbehaved for this candidate
    passed: bool
    score: float
    notes: str = ""


def _parse_verdict(stdout: str) -> Verdict:
    try:
        d = json.loads(stdout.strip())
    except json.JSONDecodeError as exc:
        return Verdict(ok=False, passed=False, score=0.0,
                       notes=f"verifier stdout not JSON: {exc}")
    if not isinstance(d, dict) or "score" not in d:
        return Verdict(ok=False, passed=False, score=0.0,
                       notes="verifier JSON missing required 'score'")
    try:
        score = float(d["score"])
    except (TypeError, ValueError):
        return Verdict(ok=False, passed=False, score=0.0,
                       notes=f"verifier 'score' not a number: {d.get('score')!r}")
    return Verdict(ok=True, passed=bool(d.get("pass", False)), score=score,
                   notes=str(d.get("notes", "")))


def run_verifier(cmd: str, value: Any, timeout_s: float = 30.0) -> Verdict:
    """Score one candidate. See module docstring for the protocol and policy."""
    # posix=False keeps Windows backslash paths intact; strip the surrounding
    # quotes it leaves on quoted tokens (e.g. a python.exe path with spaces).
    argv = [tok.strip('"') for tok in shlex.split(cmd, posix=False)]
    if not argv:
        raise VerifierError("empty verifier command")
    payload = json.dumps(value, ensure_ascii=False)
    try:
        proc = subprocess.run(
            argv,
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        raise VerifierError(f"verifier command not found: {cmd!r} ({exc})") from exc
    except subprocess.TimeoutExpired:
        return Verdict(ok=False, passed=False, score=0.0,
                       notes=f"verifier timed out after {timeout_s}s")
    except OSError as exc:
        raise VerifierError(f"verifier failed to launch: {cmd!r} ({exc})") from exc

    if not proc.stdout.strip():
        return Verdict(ok=False, passed=False, score=0.0,
                       notes=f"verifier produced no stdout (exit {proc.returncode}); "
                             f"stderr: {proc.stderr.strip()[:200]}")
    return _parse_verdict(proc.stdout)
