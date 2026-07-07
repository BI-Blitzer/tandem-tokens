"""Contract tests for the job spec loader and validators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ollamaq.errors import JobSpecError
from ollamaq.jobspec import JobSpec, load_jobspec

MINIMAL = {
    "name": "j",
    "task_tag": "clean",
    "input": {"kind": "csv", "path": "d.csv", "chunk": {"strategy": "rows", "size": 10}},
    "prompt": {"template": "clean {{chunk}}"},
}


def test_minimal_ok() -> None:
    spec = JobSpec.model_validate(MINIMAL)
    assert spec.mode.value == "map"
    assert spec.model == "auto"
    assert spec.retry.max_attempts == 3


def test_template_requires_placeholder() -> None:
    bad = {**MINIMAL, "prompt": {"template": "no placeholder"}}
    with pytest.raises(Exception, match="chunk"):
        JobSpec.model_validate(bad)


def test_auto_model_requires_task_tag() -> None:
    bad = {k: v for k, v in MINIMAL.items() if k != "task_tag"}
    with pytest.raises(Exception, match="task_tag"):
        JobSpec.model_validate(bad)


def test_gen_test_requires_verifier() -> None:
    bad = {**MINIMAL, "mode": "gen_test", "model": "qwen3:8b"}
    with pytest.raises(Exception, match="verifier"):
        JobSpec.model_validate(bad)


def test_images_kind_accepted() -> None:
    spec = JobSpec.model_validate({
        "name": "v", "model": "llama3.2-vision:11b",
        "input": {"kind": "images", "path": "shots/*.png",
                  "chunk": {"strategy": "files", "size": 1}},
        "prompt": {"template": "extract from {{chunk}}"},
    })
    assert spec.input.kind.value == "images"


def test_inline_requires_values() -> None:
    bad = {
        "name": "j",
        "model": "gemma2:9b",
        "input": {"kind": "inline", "chunk": {"strategy": "rows", "size": 2}},
        "prompt": {"template": "{{chunk}}"},
    }
    with pytest.raises(Exception, match="values"):
        JobSpec.model_validate(bad)


def test_overlap_below_size() -> None:
    bad = {**MINIMAL, "input": {"kind": "csv", "path": "d.csv",
                                "chunk": {"strategy": "rows", "size": 5, "overlap": 5}}}
    with pytest.raises(Exception, match="overlap"):
        JobSpec.model_validate(bad)


def test_schema_alias_and_params_options() -> None:
    spec = JobSpec.model_validate(
        {
            **MINIMAL,
            "prompt": {"template": "{{chunk}}", "schema": {"type": "object"}},
            "params": {"temperature": 0.2, "num_ctx": 4096, "extra": {"top_k": 5}},
        }
    )
    assert spec.prompt.schema_ == {"type": "object"}
    assert spec.params.to_options() == {"top_k": 5, "temperature": 0.2, "num_ctx": 4096}


def test_pipeline_requires_stages() -> None:
    bad = {"name": "p", "mode": "pipeline",
           "input": {"kind": "inline", "values": [1], "chunk": {"strategy": "rows", "size": 1}}}
    with pytest.raises(Exception, match="stages"):
        JobSpec.model_validate(bad)


def test_pipeline_ok_without_top_level_prompt() -> None:
    spec = JobSpec.model_validate({
        "name": "p", "mode": "pipeline",
        "input": {"kind": "inline", "values": [1], "chunk": {"strategy": "rows", "size": 1}},
        "stages": [{"name": "s1", "model": "gemma2:9b",
                    "prompt": {"template": "do {{chunk}}"}}],
    })
    assert spec.prompt is None and len(spec.stages) == 1


def test_map_still_requires_prompt() -> None:
    bad = {"name": "m", "model": "gemma2:9b",
           "input": {"kind": "inline", "values": [1], "chunk": {"strategy": "rows", "size": 1}}}
    with pytest.raises(Exception, match="prompt"):
        JobSpec.model_validate(bad)


def test_pipeline_stage_auto_needs_tag() -> None:
    bad = {"name": "p", "mode": "pipeline",
           "input": {"kind": "inline", "values": [1], "chunk": {"strategy": "rows", "size": 1}},
           "stages": [{"name": "s1", "prompt": {"template": "do {{chunk}}"}}]}  # model defaults auto
    with pytest.raises(Exception, match="task_tag"):
        JobSpec.model_validate(bad)


def test_load_jobspec_missing(tmp_path: Path) -> None:
    with pytest.raises(JobSpecError, match="not found"):
        load_jobspec(tmp_path / "nope.json")


def test_load_jobspec_bad_json(tmp_path: Path) -> None:
    p = tmp_path / "j.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(JobSpecError, match="not valid JSON"):
        load_jobspec(p)


def test_load_jobspec_invalid_schema(tmp_path: Path) -> None:
    p = tmp_path / "j.json"
    p.write_text(json.dumps({"name": "j"}), encoding="utf-8")
    with pytest.raises(JobSpecError, match="invalid job spec"):
        load_jobspec(p)


def test_load_jobspec_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "j.json"
    p.write_text(json.dumps(MINIMAL), encoding="utf-8")
    spec = load_jobspec(p)
    assert spec.name == "j"
