"""Disk-backed dedupe cache for validated item outputs.

Opt-in per job (`cache: true`). Keyed on the exact request that produced a
result — model, system prompt, the rendered per-item prompt, the JSON schema,
and the Ollama options — so ANY change to those is a natural cache miss. Only
validated successes are stored; schema failures are never cached (they should
retry fresh next run).

On a hit, the worker skips the HTTP call AND re-validation entirely (the schema
is part of the key, so a schema change can't produce a stale hit). One JSON file
per key under `cache/`; writes are atomic (temp + os.replace) so concurrent
workers never observe a half-written entry.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional


def make_key(model: str, system: Optional[str], prompt: str,
             schema: Optional[dict], options: Optional[dict]) -> str:
    """Stable sha256 hex key for a request. Canonical JSON (sorted keys) so the
    key depends only on content, not dict ordering."""
    canonical = json.dumps(
        {"model": model, "system": system, "prompt": prompt,
         "schema": schema, "options": options},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Cache:
    """One JSON file per key under `root`. Never raises on read — a missing or
    corrupt entry is simply a miss."""

    def __init__(self, root: str | Path = "cache") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value, or None on miss / unreadable / corrupt file."""
        p = self._path(key)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data.get("value") if isinstance(data, dict) else None

    def put(self, key: str, value: Any) -> None:
        """Store a validated value atomically (temp file + os.replace)."""
        p = self._path(key)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"value": value}, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, p)
