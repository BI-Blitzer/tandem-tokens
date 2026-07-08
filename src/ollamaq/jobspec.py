"""The job spec - the single contract every subsystem shares.

This models the FULL spec documented in PLAN.md section 3, including fields that
only later phases act on (verifier, loop, escalate_to). Validating them now means
the contract a lower model sees never shifts underneath it. Fields not yet honored
are marked "[Phase N]" in their docstrings; loading a spec that uses them is allowed,
but the runner rejects unsupported *modes* explicitly (see load_jobspec).

Load specs via load_jobspec() - it turns any validation problem into JobSpecError
with a path-qualified message, never a raw pydantic traceback.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .errors import JobSpecError


class Mode(str, Enum):
    map = "map"
    gen_test = "gen_test"
    pipeline = "pipeline"
    embed = "embed"      # embed each item's text via a local embedding model


class InputKind(str, Enum):
    glob = "glob"      # path is a glob; each matched file is one item
    jsonl = "jsonl"    # path is a .jsonl file; each line is chunked by rows
    csv = "csv"        # path is a .csv file; rows are chunked
    text = "text"      # path is a text file; chunked by lines or bytes
    inline = "inline"  # items provided directly in `values`
    images = "images"  # path is an image glob; each image is one item (vision)


class ChunkStrategy(str, Enum):
    rows = "rows"      # for csv/jsonl: N records per chunk
    lines = "lines"    # for text: N lines per chunk
    bytes = "bytes"    # for text: ~N bytes per chunk (split on line boundaries)
    files = "files"    # for glob: one file per chunk (size ignored)


class ChunkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: ChunkStrategy
    size: int = Field(default=50, gt=0)
    overlap: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _overlap_below_size(self) -> "ChunkSpec":
        if self.overlap >= self.size:
            raise ValueError(f"chunk.overlap ({self.overlap}) must be < size ({self.size})")
        return self


class InputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: InputKind
    path: Optional[str] = None
    values: Optional[list[Any]] = None
    chunk: ChunkSpec

    @model_validator(mode="after")
    def _path_or_values(self) -> "InputSpec":
        if self.kind == InputKind.inline:
            if self.values is None:
                raise ValueError("input.kind='inline' requires `values`")
        else:
            if not self.path:
                raise ValueError(f"input.kind={self.kind.value!r} requires `path`")
        return self


class PromptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    system: Optional[str] = None
    template: str = Field(min_length=1)
    schema_: Optional[dict[str, Any]] = Field(default=None, alias="schema")

    @model_validator(mode="after")
    def _template_has_placeholder(self) -> "PromptSpec":
        if "{{chunk}}" not in self.template:
            raise ValueError("prompt.template must contain the {{chunk}} placeholder")
        return self


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: Optional[float] = None
    num_ctx: Optional[int] = Field(default=None, gt=0)
    # Any other Ollama option may be passed through here.
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_options(self) -> dict[str, Any]:
        """Flatten into the dict Ollama's `options` field expects."""
        opts: dict[str, Any] = dict(self.extra)
        if self.temperature is not None:
            opts["temperature"] = self.temperature
        if self.num_ctx is not None:
            opts["num_ctx"] = self.num_ctx
        return opts


class VerifierSpec(BaseModel):
    """[Phase 3] Scoring subprocess. Parsed now; honored in Phase 3."""

    model_config = ConfigDict(extra="forbid")

    cmd: str
    keep_top: int = Field(default=20, gt=0)
    timeout_s: float = Field(default=30.0, gt=0)


class OnSchemaFail(str, Enum):
    repair_then_retry = "repair_then_retry"
    retry = "retry"
    fail = "fail"


class RetrySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=3, ge=1)
    on_schema_fail: OnSchemaFail = OnSchemaFail.repair_then_retry
    escalate_to: Optional[str] = None  # bigger model for persistent failures
    transient_retries: int = Field(default=2, ge=0)  # retries on connection/HTTP errors


class LoopSpec(BaseModel):
    """gen_test continuous-loop control (Phase 3)."""

    model_config = ConfigDict(extra="forbid")

    candidates: int = Field(default=8, ge=1)  # candidates generated per iteration
    max_iterations: int = Field(default=10, ge=1)
    stop_when: str = ""                         # e.g. "score>=0.95 or no_improvement(3)"
    feedback: Literal["none", "top_k"] = "top_k"


class FilterOp(str, Enum):
    eq = "=="
    ne = "!="
    gt = ">"
    ge = ">="
    lt = "<"
    le = "<="
    truthy = "truthy"   # field is present and truthy
    falsy = "falsy"     # field is absent or falsy


class StageFilter(BaseModel):
    """Applied AFTER a pipeline stage: items whose output MATCHES continue to the
    next stage; non-matching items are finalized with their current value. This
    is what makes 'clean only the rows flagged dirty' expressible."""

    model_config = ConfigDict(extra="forbid")

    field: str                              # dot-path into the stage's output value
    op: FilterOp
    value: Any = None                       # unused for truthy/falsy


class PipelineStage(BaseModel):
    """One stage of a pipeline. Runs as a map over the active item set; its output
    value becomes the next stage's {{chunk}} input (JSON-encoded)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(min_length=1)
    model: str = "auto"
    task_tag: Optional[str] = None
    prompt: PromptSpec
    params: Params = Field(default_factory=Params)
    retry: RetrySpec = Field(default_factory=RetrySpec)
    filter: Optional[StageFilter] = None

    @model_validator(mode="after")
    def _auto_needs_tag(self) -> "PipelineStage":
        if self.model == "auto" and not self.task_tag:
            raise ValueError(f"stage {self.name!r}: model='auto' requires task_tag")
        return self


class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    mode: Mode = Mode.map
    model: str = "auto"
    task_tag: Optional[str] = None
    input: InputSpec
    prompt: Optional[PromptSpec] = None      # required for map/gen_test; unused for pipeline
    params: Params = Field(default_factory=Params)
    verifier: Optional[VerifierSpec] = None
    retry: RetrySpec = Field(default_factory=RetrySpec)
    loop: Optional[LoopSpec] = None
    stages: Optional[list[PipelineStage]] = None  # required for pipeline
    cache: bool = False                           # dedupe: reuse prior validated outputs

    @model_validator(mode="after")
    def _cross_field(self) -> "JobSpec":
        if self.mode == Mode.pipeline:
            if not self.stages:
                raise ValueError("mode='pipeline' requires a non-empty 'stages' list")
            return self
        # map / gen_test / embed all need a routable model.
        if self.model == "auto" and not self.task_tag:
            raise ValueError("model='auto' requires task_tag so it can be routed")
        if self.mode == Mode.embed:
            return self  # embed uses the chunk text directly; no prompt/schema
        if self.prompt is None:
            raise ValueError(f"mode={self.mode.value!r} requires a 'prompt'")
        if self.mode == Mode.gen_test and self.verifier is None:
            raise ValueError("mode='gen_test' requires a verifier")
        return self


def load_jobspec(path: str | Path) -> JobSpec:
    """Load and validate a job spec JSON file.

    Raises JobSpecError (never a raw pydantic/JSON error) on any problem:
    missing file, invalid JSON, or schema violation.
    """
    p = Path(path)
    if not p.is_file():
        raise JobSpecError(f"job spec not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JobSpecError(f"job spec {p} is not valid JSON: {exc}") from exc
    try:
        return JobSpec.model_validate(raw)
    except ValidationError as exc:
        raise JobSpecError(f"invalid job spec {p}: {exc}") from exc
