"""Contract tests for streaming: iter_chunks parity + the streaming runner."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx

from ollamaq.chunker import chunk_input, iter_chunks
from ollamaq.config import Config
from ollamaq.jobspec import InputSpec, JobSpec
from ollamaq.stream import run_stream

BASE = "http://localhost:11434"


# ---- iter_chunks parity with eager chunk_input -------------------------------

def _write_csv(tmp_path: Path, rows: int) -> Path:
    p = tmp_path / "d.csv"
    p.write_text("id,v\n" + "\n".join(f"{i},x{i}" for i in range(rows)) + "\n",
                 encoding="utf-8")
    return p


def _spec_csv(path: Path, size: int, overlap: int = 0) -> InputSpec:
    return InputSpec.model_validate({"kind": "csv", "path": str(path),
                                     "chunk": {"strategy": "rows", "size": size, "overlap": overlap}})


def test_iter_chunks_csv_matches_eager(tmp_path: Path) -> None:
    spec = _spec_csv(_write_csv(tmp_path, 23), size=5)
    eager = chunk_input(spec)
    streamed = list(iter_chunks(spec))
    assert [(w.index, w.text, w.source) for w in streamed] == \
           [(w.index, w.text, w.source) for w in eager]


def test_iter_chunks_jsonl_matches_eager(tmp_path: Path) -> None:
    p = tmp_path / "d.jsonl"
    p.write_text("".join(f'{{"a":{i}}}\n' for i in range(17)), encoding="utf-8")
    spec = InputSpec.model_validate({"kind": "jsonl", "path": str(p),
                                     "chunk": {"strategy": "rows", "size": 4}})
    assert [w.text for w in iter_chunks(spec)] == [w.text for w in chunk_input(spec)]


def test_iter_chunks_text_lines_matches_eager(tmp_path: Path) -> None:
    p = tmp_path / "d.txt"
    p.write_text("\n".join(f"line{i}" for i in range(20)) + "\n", encoding="utf-8")
    spec = InputSpec.model_validate({"kind": "text", "path": str(p),
                                     "chunk": {"strategy": "lines", "size": 3}})
    assert [w.text for w in iter_chunks(spec)] == [w.text for w in chunk_input(spec)]


def test_iter_chunks_overlap_falls_back_to_eager(tmp_path: Path) -> None:
    spec = _spec_csv(_write_csv(tmp_path, 10), size=4, overlap=1)
    assert [w.text for w in iter_chunks(spec)] == [w.text for w in chunk_input(spec)]


def test_iter_chunks_is_lazy(tmp_path: Path) -> None:
    # a generator, not a materialized list — pulling one item shouldn't read all
    spec = _spec_csv(_write_csv(tmp_path, 1000), size=10)
    gen = iter_chunks(spec)
    first = next(gen)
    assert first.index == 0  # got a chunk without exhausting the file


# ---- streaming runner (mocked) -----------------------------------------------

def _cfg() -> Config:
    return Config.model_validate({
        "ollama": {"base_url": BASE},
        "defaults": {"model": "gemma2:9b"},
        "models": {"gemma2:9b": {"max_parallel": 4, "num_ctx": 8192}},
        "routing": {"clean": "gemma2:9b"},
    })


@respx.mock
async def test_run_stream_writes_incrementally(tmp_path: Path) -> None:
    respx.post(f"{BASE}/api/generate").mock(return_value=httpx.Response(
        200, json={"response": '{"ok": true}', "eval_count": 3, "eval_duration": 1_000_000_000}))
    csv = _write_csv(tmp_path, 25)  # 25 rows / size 5 = 5 chunks
    spec = JobSpec.model_validate({
        "name": "s", "model": "gemma2:9b",
        "input": {"kind": "csv", "path": str(csv), "chunk": {"strategy": "rows", "size": 5}},
        "prompt": {"template": "clean {{chunk}}", "schema": {"type": "object", "required": ["ok"]}},
    })
    rd = tmp_path / "results" / "s1"
    res = await run_stream(_cfg(), spec, rd, pricing=None, batch_size=2)
    assert res == {"ok": 5, "fail": 0, "items": 5, "model": "gemma2:9b"}
    merged = (rd / "merged.jsonl").read_text().splitlines()
    assert len(merged) == 5
    assert json.loads(merged[0])["value"] == {"ok": True}
    assert "items: 5" in (rd / "digest.md").read_text()
    assert json.loads((rd / "run.json").read_text())["streamed"] is True
    # ledger got an entry
    assert (tmp_path / "results" / "ledger.jsonl").is_file()
