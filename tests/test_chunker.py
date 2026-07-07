"""Contract tests for the chunker: exact item counts, overlap, and re-emission."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ollamaq.chunker import chunk_input
from ollamaq.errors import JobSpecError
from ollamaq.jobspec import InputSpec


def _spec(**kw) -> InputSpec:
    return InputSpec.model_validate(kw)


def test_csv_rows(tmp_path: Path) -> None:
    p = tmp_path / "d.csv"
    p.write_text("id,name\n1,a\n2,b\n3,c\n4,d\n5,e\n", encoding="utf-8")
    items = chunk_input(_spec(kind="csv", path=str(p),
                              chunk={"strategy": "rows", "size": 2}))
    assert len(items) == 3  # 5 rows / 2 -> 3 chunks
    # every chunk repeats the header
    assert all(i.text.startswith("id,name") for i in items)
    assert items[0].text == "id,name\n1,a\n2,b"
    assert items[2].text == "id,name\n5,e"


def test_csv_rows_overlap(tmp_path: Path) -> None:
    p = tmp_path / "d.csv"
    p.write_text("h\n1\n2\n3\n4\n", encoding="utf-8")
    items = chunk_input(_spec(kind="csv", path=str(p),
                              chunk={"strategy": "rows", "size": 2, "overlap": 1}))
    # step = 1 -> windows start at 0,1,2,3
    assert [i.text.split("\n")[1:] for i in items] == [
        ["1", "2"], ["2", "3"], ["3", "4"], ["4"]
    ]


def test_jsonl_rows(tmp_path: Path) -> None:
    p = tmp_path / "d.jsonl"
    p.write_text('{"a":1}\n{"a":2}\n{"a":3}\n', encoding="utf-8")
    items = chunk_input(_spec(kind="jsonl", path=str(p),
                              chunk={"strategy": "rows", "size": 2}))
    assert len(items) == 2
    assert json.loads(items[0].text) == [{"a": 1}, {"a": 2}]


def test_jsonl_malformed(tmp_path: Path) -> None:
    p = tmp_path / "d.jsonl"
    p.write_text('{"a":1}\n{bad}\n', encoding="utf-8")
    with pytest.raises(JobSpecError, match="not valid JSON"):
        chunk_input(_spec(kind="jsonl", path=str(p),
                          chunk={"strategy": "rows", "size": 5}))


def test_text_lines(tmp_path: Path) -> None:
    p = tmp_path / "d.txt"
    p.write_text("l1\nl2\nl3\nl4\nl5\n", encoding="utf-8")
    items = chunk_input(_spec(kind="text", path=str(p),
                              chunk={"strategy": "lines", "size": 2}))
    assert len(items) == 3
    assert items[1].text == "l3\nl4"


def test_text_bytes(tmp_path: Path) -> None:
    p = tmp_path / "d.txt"
    p.write_text("aaaa\nbbbb\ncccc\n", encoding="utf-8")  # 5 bytes each incl newline
    items = chunk_input(_spec(kind="text", path=str(p),
                              chunk={"strategy": "bytes", "size": 10}))
    # ~10 bytes -> 2 lines per chunk
    assert len(items) == 2
    assert items[0].text == "aaaa\nbbbb"


def test_glob_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("AAA", encoding="utf-8")
    (tmp_path / "b.txt").write_text("BBB", encoding="utf-8")
    items = chunk_input(_spec(kind="glob", path=str(tmp_path / "*.txt"),
                              chunk={"strategy": "files", "size": 1}))
    assert [i.text for i in items] == ["AAA", "BBB"]


def test_glob_no_match(tmp_path: Path) -> None:
    with pytest.raises(JobSpecError, match="matched no files"):
        chunk_input(_spec(kind="glob", path=str(tmp_path / "*.none"),
                          chunk={"strategy": "files", "size": 1}))


def test_images_glob(tmp_path: Path) -> None:
    import base64
    (tmp_path / "a.png").write_bytes(b"\x89PNGfake-a")
    (tmp_path / "b.png").write_bytes(b"\x89PNGfake-b")
    items = chunk_input(_spec(kind="images", path=str(tmp_path / "*.png"),
                              chunk={"strategy": "files", "size": 1}))
    assert [i.text for i in items] == ["a.png", "b.png"]           # filename as text
    assert all(len(i.images) == 1 for i in items)                   # one image each
    assert base64.standard_b64decode(items[0].images[0]) == b"\x89PNGfake-a"


def test_images_wrong_strategy(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    with pytest.raises(JobSpecError, match="images input requires"):
        chunk_input(_spec(kind="images", path=str(tmp_path / "*.png"),
                          chunk={"strategy": "rows", "size": 1}))


def test_images_no_match(tmp_path: Path) -> None:
    with pytest.raises(JobSpecError, match="matched no files"):
        chunk_input(_spec(kind="images", path=str(tmp_path / "*.png"),
                          chunk={"strategy": "files", "size": 1}))


def test_inline_rows() -> None:
    items = chunk_input(_spec(kind="inline", values=[1, 2, 3, 4, 5],
                              chunk={"strategy": "rows", "size": 2}))
    assert len(items) == 3
    assert json.loads(items[0].text) == [1, 2]


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(JobSpecError, match="not found"):
        chunk_input(_spec(kind="csv", path=str(tmp_path / "no.csv"),
                          chunk={"strategy": "rows", "size": 2}))


def test_wrong_strategy_for_kind(tmp_path: Path) -> None:
    p = tmp_path / "d.csv"
    p.write_text("h\n1\n", encoding="utf-8")
    with pytest.raises(JobSpecError, match="requires chunk.strategy"):
        chunk_input(_spec(kind="csv", path=str(p),
                          chunk={"strategy": "lines", "size": 2}))


def test_indices_are_sequential(tmp_path: Path) -> None:
    p = tmp_path / "d.csv"
    p.write_text("h\n" + "\n".join(str(i) for i in range(10)) + "\n", encoding="utf-8")
    items = chunk_input(_spec(kind="csv", path=str(p),
                              chunk={"strategy": "rows", "size": 3}))
    assert [i.index for i in items] == [0, 1, 2, 3]
