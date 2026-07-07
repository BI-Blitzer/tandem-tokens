"""Split a job's input into WorkItems. Pure, deterministic, side-effect-free
except for reading the input files named in the spec.

Each WorkItem carries a stable ``index`` (0-based, in emission order) and a
``text`` payload that gets substituted into the prompt template's {{chunk}}.
The mapping strategy -> shape:

  csv   + rows   -> N data rows per item, re-emitted as CSV (header repeated)
  jsonl + rows   -> N json objects per item, re-emitted as a JSON array
  text  + lines  -> N lines per item
  text  + bytes  -> ~size bytes per item, split only on line boundaries
  glob  + files  -> one file's full contents per item (size/overlap ignored)
  inline + rows  -> N values per item, re-emitted as a JSON array

`overlap` repeats the trailing N units of one chunk at the head of the next
(rows/lines only; ignored for bytes/files).
"""

from __future__ import annotations

import csv
import base64
import glob
import io
import json
from dataclasses import dataclass
from pathlib import Path

from .errors import JobSpecError
from .jobspec import ChunkStrategy, InputKind, InputSpec


@dataclass(frozen=True)
class WorkItem:
    index: int
    text: str
    source: str  # human-readable origin, e.g. "rows 0-49" or a file path
    images: tuple[str, ...] = ()  # base64 image(s) for vision jobs; empty otherwise


def _windows(units: list, size: int, overlap: int):
    """Yield successive (start_index, slice) windows with overlap."""
    step = size - overlap
    i = 0
    n = len(units)
    while i < n:
        yield i, units[i : i + size]
        i += step


def _chunk_rows_csv(path: Path, size: int, overlap: int) -> list[WorkItem]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return []
    header, data = rows[0], rows[1:]
    items: list[WorkItem] = []
    for idx, (start, window) in enumerate(_windows(data, size, overlap)):
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(header)
        w.writerows(window)
        items.append(
            WorkItem(index=idx, text=buf.getvalue().rstrip("\n"),
                     source=f"rows {start}-{start + len(window) - 1}")
        )
    return items


def _chunk_rows_jsonl(path: Path, size: int, overlap: int) -> list[WorkItem]:
    objs = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            objs.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise JobSpecError(f"{path}:{lineno + 1} is not valid JSON: {exc}") from exc
    items: list[WorkItem] = []
    for idx, (start, window) in enumerate(_windows(objs, size, overlap)):
        items.append(
            WorkItem(index=idx, text=json.dumps(window, ensure_ascii=False),
                     source=f"records {start}-{start + len(window) - 1}")
        )
    return items


def _chunk_inline(values: list, size: int, overlap: int) -> list[WorkItem]:
    items: list[WorkItem] = []
    for idx, (start, window) in enumerate(_windows(values, size, overlap)):
        items.append(
            WorkItem(index=idx, text=json.dumps(window, ensure_ascii=False),
                     source=f"values {start}-{start + len(window) - 1}")
        )
    return items


def _chunk_lines(path: Path, size: int, overlap: int) -> list[WorkItem]:
    lines = path.read_text(encoding="utf-8").splitlines()
    items: list[WorkItem] = []
    for idx, (start, window) in enumerate(_windows(lines, size, overlap)):
        items.append(
            WorkItem(index=idx, text="\n".join(window),
                     source=f"lines {start}-{start + len(window) - 1}")
        )
    return items


def _chunk_bytes(path: Path, size: int) -> list[WorkItem]:
    """~size bytes per item, never splitting a line."""
    items: list[WorkItem] = []
    buf: list[str] = []
    cur = 0
    idx = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        add = len(line.encode("utf-8")) + 1
        if buf and cur + add > size:
            items.append(WorkItem(index=idx, text="\n".join(buf), source=f"chunk {idx}"))
            idx += 1
            buf, cur = [], 0
        buf.append(line)
        cur += add
    if buf:
        items.append(WorkItem(index=idx, text="\n".join(buf), source=f"chunk {idx}"))
    return items


def _chunk_files(pattern: str) -> list[WorkItem]:
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise JobSpecError(f"glob {pattern!r} matched no files")
    return [
        WorkItem(index=i, text=Path(m).read_text(encoding="utf-8"), source=m)
        for i, m in enumerate(matches)
    ]


def _chunk_images(pattern: str) -> list[WorkItem]:
    """One image file per item. `text` is the filename (so {{chunk}} names it);
    the base64-encoded bytes ride in `images` for the vision model."""
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise JobSpecError(f"image glob {pattern!r} matched no files")
    items = []
    for i, m in enumerate(matches):
        b64 = base64.standard_b64encode(Path(m).read_bytes()).decode("ascii")
        items.append(WorkItem(index=i, text=Path(m).name, source=m, images=(b64,)))
    return items


def chunk_input(spec: InputSpec) -> list[WorkItem]:
    """Turn an InputSpec into an ordered list of WorkItems.

    Raises JobSpecError for missing files, empty globs, malformed jsonl, or an
    unsupported (kind, strategy) pairing.
    """
    kind, strat = spec.kind, spec.chunk.strategy
    size, overlap = spec.chunk.size, spec.chunk.overlap

    if kind == InputKind.inline:
        assert spec.values is not None  # guaranteed by InputSpec validator
        if strat != ChunkStrategy.rows:
            raise JobSpecError("inline input requires chunk.strategy='rows'")
        return _chunk_inline(spec.values, size, overlap)

    if kind == InputKind.glob:
        if strat != ChunkStrategy.files:
            raise JobSpecError("glob input requires chunk.strategy='files'")
        return _chunk_files(spec.path)  # type: ignore[arg-type]

    if kind == InputKind.images:
        if strat != ChunkStrategy.files:
            raise JobSpecError("images input requires chunk.strategy='files'")
        return _chunk_images(spec.path)  # type: ignore[arg-type]

    path = Path(spec.path)  # type: ignore[arg-type]
    if not path.is_file():
        raise JobSpecError(f"input file not found: {path}")

    if kind == InputKind.csv:
        if strat != ChunkStrategy.rows:
            raise JobSpecError("csv input requires chunk.strategy='rows'")
        return _chunk_rows_csv(path, size, overlap)

    if kind == InputKind.jsonl:
        if strat != ChunkStrategy.rows:
            raise JobSpecError("jsonl input requires chunk.strategy='rows'")
        return _chunk_rows_jsonl(path, size, overlap)

    if kind == InputKind.text:
        if strat == ChunkStrategy.lines:
            return _chunk_lines(path, size, overlap)
        if strat == ChunkStrategy.bytes:
            return _chunk_bytes(path, size)
        raise JobSpecError("text input requires chunk.strategy='lines' or 'bytes'")

    raise JobSpecError(f"unsupported input.kind {kind!r}")  # pragma: no cover
