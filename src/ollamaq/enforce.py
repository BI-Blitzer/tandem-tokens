"""Validate a model's raw text output against the job's JSON schema, with a
bounded repair-retry loop.

Contract (honored by worker.py):
  - No schema on the job  -> output is accepted as-is (parsed if it happens to be
    JSON, else kept as {"text": raw}).
  - Schema present        -> the text must parse as JSON AND validate. On failure,
    the policy in RetrySpec.on_schema_fail decides whether to repair/retry.

process_attempt() is pure (no I/O): it takes raw text + schema and returns an
AttemptOutcome. The retry LOOP lives in worker.py because only it can call the
model again; this module just judges one attempt and builds the repair prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import jsonschema


@dataclass
class AttemptOutcome:
    ok: bool
    value: Optional[Any] = None      # parsed+validated payload when ok
    error: str = ""                  # why it failed, when not ok


def try_parse_json(text: str) -> tuple[bool, Any, str]:
    """Best-effort: parse text as JSON, tolerating ```json fences and surrounding
    prose by extracting the outermost {...} or [...] span."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # drop a leading ```json / ``` fence and trailing ```
        stripped = stripped.split("\n", 1)[-1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    try:
        return True, json.loads(stripped), ""
    except json.JSONDecodeError:
        pass
    # fall back to the first balanced-looking JSON span
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = stripped.find(opener), stripped.rfind(closer)
        if 0 <= i < j:
            try:
                return True, json.loads(stripped[i : j + 1]), ""
            except json.JSONDecodeError:
                continue
    return False, None, "output is not valid JSON"


def validate_against_schema(value: Any, schema: dict[str, Any]) -> tuple[bool, str]:
    try:
        jsonschema.validate(value, schema)
        return True, ""
    except jsonschema.ValidationError as exc:
        return False, f"schema violation: {exc.message}"


def process_attempt(raw: str, schema: Optional[dict[str, Any]]) -> AttemptOutcome:
    """Judge a single model response. Pure - no model calls, no I/O."""
    if schema is None:
        ok, parsed, _ = try_parse_json(raw)
        return AttemptOutcome(ok=True, value=parsed if ok else {"text": raw})
    ok, parsed, err = try_parse_json(raw)
    if not ok:
        return AttemptOutcome(ok=False, error=err)
    valid, verr = validate_against_schema(parsed, schema)
    if not valid:
        return AttemptOutcome(ok=False, error=verr)
    return AttemptOutcome(ok=True, value=parsed)


def build_repair_prompt(original_template: str, chunk_text: str, bad_output: str,
                        error: str, schema: dict[str, Any]) -> str:
    """Prompt that asks the model to fix its previous, non-conforming output."""
    return (
        "Your previous response did NOT satisfy the required JSON schema.\n"
        f"Error: {error}\n\n"
        "Required JSON schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Your previous (invalid) response was:\n"
        f"{bad_output}\n\n"
        "Return ONLY corrected JSON that satisfies the schema. No commentary, no code fences."
    )
