"""Contract tests for the auto-offload detector + hook script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ollamaq.detect import detect_offloadable


# ---- detector: should fire ---------------------------------------------------

def test_fires_on_bulk_clean() -> None:
    s = detect_offloadable("Please clean all 500 rows in this CSV and normalize the phone numbers.")
    assert s and "map mode" in s and "clean" in s


def test_fires_on_classify_batch() -> None:
    s = detect_offloadable("Classify each of these support tickets into a category.")
    assert s and "classify" in s


def test_fires_on_embed() -> None:
    s = detect_offloadable("Embed all these documents so I can do semantic search.")
    assert s and "embed mode" in s


def test_fires_on_extract_files() -> None:
    s = detect_offloadable("Extract the invoice totals from these files.")
    assert s and "extract" in s


def test_fires_on_gentest() -> None:
    s = detect_offloadable("Generate 50 regex candidates and test each against the fixtures.")
    assert s and "gen_test mode" in s


# ---- detector: should stay silent --------------------------------------------

def test_silent_on_single_question() -> None:
    assert detect_offloadable("What does this function do?") is None


def test_silent_on_normal_edit() -> None:
    assert detect_offloadable("Fix the bug in cli.py where the daemon hangs.") is None


def test_silent_on_verb_without_bulk() -> None:
    # a task verb but no bulk/data signal
    assert detect_offloadable("Summarize your last message.") is None


def test_silent_on_empty_and_huge() -> None:
    assert detect_offloadable("") is None
    assert detect_offloadable("clean rows " * 1000) is None  # > 4000 chars -> skip


# ---- hook script (simulates Claude Code calling it) --------------------------

def _run_hook(payload: dict) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(Path("hooks") / "auto_offload.py")],
        input=json.dumps(payload), capture_output=True, text=True, timeout=30)
    return proc.returncode, proc.stdout.strip()


def test_hook_emits_context_on_bulk() -> None:
    rc, out = _run_hook({"prompt": "Clean all 300 rows in this csv."})
    assert rc == 0
    data = json.loads(out)
    assert data["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "tandem-tokens" in data["hookSpecificOutput"]["additionalContext"]


def test_hook_silent_on_normal_prompt() -> None:
    rc, out = _run_hook({"prompt": "What time is it?"})
    assert rc == 0 and out == ""  # no context injected


def test_hook_survives_bad_input() -> None:
    proc = subprocess.run(
        [sys.executable, str(Path("hooks") / "auto_offload.py")],
        input="not json", capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0 and proc.stdout.strip() == ""
