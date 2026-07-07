"""Contract tests for schema parsing/validation and the repair prompt."""

from __future__ import annotations

from ollamaq.enforce import (
    build_repair_prompt,
    process_attempt,
    try_parse_json,
    validate_against_schema,
)

SCHEMA = {
    "type": "object",
    "properties": {"id": {"type": "integer"}},
    "required": ["id"],
}


def test_try_parse_plain() -> None:
    ok, val, _ = try_parse_json('{"id": 1}')
    assert ok and val == {"id": 1}


def test_try_parse_fenced() -> None:
    ok, val, _ = try_parse_json('```json\n{"id": 2}\n```')
    assert ok and val == {"id": 2}


def test_try_parse_with_prose() -> None:
    ok, val, _ = try_parse_json('Sure! Here it is: {"id": 3} hope that helps')
    assert ok and val == {"id": 3}


def test_try_parse_fail() -> None:
    ok, _, err = try_parse_json("no json here")
    assert not ok and "not valid JSON" in err


def test_validate_pass() -> None:
    ok, _ = validate_against_schema({"id": 1}, SCHEMA)
    assert ok


def test_validate_fail() -> None:
    ok, err = validate_against_schema({"id": "x"}, SCHEMA)
    assert not ok and "schema violation" in err


def test_process_no_schema_keeps_text() -> None:
    out = process_attempt("just some prose", None)
    assert out.ok and out.value == {"text": "just some prose"}


def test_process_no_schema_parses_json() -> None:
    out = process_attempt('{"a": 1}', None)
    assert out.ok and out.value == {"a": 1}


def test_process_schema_ok() -> None:
    out = process_attempt('{"id": 7}', SCHEMA)
    assert out.ok and out.value == {"id": 7}


def test_process_schema_parse_fail() -> None:
    out = process_attempt("garbage", SCHEMA)
    assert not out.ok and "not valid JSON" in out.error


def test_process_schema_validation_fail() -> None:
    out = process_attempt('{"id": "x"}', SCHEMA)
    assert not out.ok and "schema violation" in out.error


def test_repair_prompt_mentions_error_and_schema() -> None:
    p = build_repair_prompt("tmpl {{chunk}}", "data", '{"id":"x"}', "schema violation: bad", SCHEMA)
    assert "schema violation: bad" in p
    assert "required" in p.lower() or "id" in p
    assert "ONLY" in p
