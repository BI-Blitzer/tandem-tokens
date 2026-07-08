"""Contract tests for the dedupe cache (Cache + make_key)."""

from __future__ import annotations

from pathlib import Path

from ollamaq.cache import Cache, make_key


def test_key_stable_and_order_independent() -> None:
    k1 = make_key("m", "sys", "prompt", {"type": "object", "a": 1, "b": 2}, {"x": 1})
    k2 = make_key("m", "sys", "prompt", {"b": 2, "type": "object", "a": 1}, {"x": 1})
    assert k1 == k2  # dict key order must not matter


def test_key_sensitive_to_each_field() -> None:
    base = make_key("m", "sys", "p", {"t": 1}, {"x": 1})
    assert make_key("m2", "sys", "p", {"t": 1}, {"x": 1}) != base   # model
    assert make_key("m", "sys2", "p", {"t": 1}, {"x": 1}) != base   # system
    assert make_key("m", "sys", "p2", {"t": 1}, {"x": 1}) != base   # prompt
    assert make_key("m", "sys", "p", {"t": 2}, {"x": 1}) != base    # schema
    assert make_key("m", "sys", "p", {"t": 1}, {"x": 2}) != base    # options


def test_miss_then_hit(tmp_path: Path) -> None:
    c = Cache(tmp_path / "cache")
    assert c.get("k") is None
    c.put("k", {"cleaned": True})
    assert c.get("k") == {"cleaned": True}


def test_get_corrupt_is_miss(tmp_path: Path) -> None:
    c = Cache(tmp_path / "cache")
    (tmp_path / "cache" / "bad.json").write_text("{not json", encoding="utf-8")
    assert c.get("bad") is None


def test_put_is_atomic_no_tmp_left(tmp_path: Path) -> None:
    c = Cache(tmp_path / "cache")
    c.put("k", [1, 2, 3])
    files = {p.name for p in (tmp_path / "cache").iterdir()}
    assert files == {"k.json"}  # no leftover .json.tmp


def test_roundtrip_preserves_value(tmp_path: Path) -> None:
    c = Cache(tmp_path / "cache")
    val = {"rows": [{"id": 1, "name": "Ann"}], "nested": {"x": [1, 2]}}
    c.put("k", val)
    assert Cache(tmp_path / "cache").get("k") == val  # survives a fresh instance


def test_cap_evicts_oldest(tmp_path: Path) -> None:
    import os, time
    c = Cache(tmp_path / "cache", max_entries=10)
    for i in range(12):
        c.put(f"k{i:02d}", i)
        os.utime(c._path(f"k{i:02d}"), (i, i))  # deterministic mtimes (oldest = k00)
    c.put("k99", 99)  # 13 entries -> over cap 10 -> evict down to 9, +1 = 10
    remaining = sorted(p.stem for p in (tmp_path / "cache").glob("*.json"))
    assert len(remaining) <= 10
    assert "k99" in remaining          # newest kept
    assert "k00" not in remaining      # oldest evicted


def test_cap_zero_disables_eviction(tmp_path: Path) -> None:
    c = Cache(tmp_path / "cache", max_entries=0)
    for i in range(50):
        c.put(f"k{i}", i)
    assert len(list((tmp_path / "cache").glob("*.json"))) == 50
