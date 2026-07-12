"""Mechanical residue fix: force phones to XXX-XXX-XXXX in a contacts merged.jsonl.

Reads results/<job>/merged.jsonl (map clean output shape), rewrites phones from
digits only, writes results/<job>/merged.normalized.jsonl and prints quality stats.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def normalize_phone(raw: str) -> str:
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) >= 10:
        digits = digits[-10:]  # keep last 10
    if len(digits) != 10:
        return raw  # leave unfixable as-is for human review
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"


def main(merged_path: Path) -> int:
    out_path = merged_path.with_name("merged.normalized.jsonl")
    total = ok = fixed = unfixable = 0
    with merged_path.open(encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            contacts = (row.get("value") or {}).get("contacts") or []
            for c in contacts:
                total += 1
                before = c.get("phone") or ""
                after = normalize_phone(before)
                if re.fullmatch(r"\d{3}-\d{3}-\d{4}", after):
                    ok += 1
                    if after != before:
                        fixed += 1
                    c["phone"] = after
                else:
                    unfixable += 1
            # also normalize email/name lightly
            for c in contacts:
                if isinstance(c.get("email"), str):
                    c["email"] = c["email"].strip().lower()
                if isinstance(c.get("name"), str):
                    c["name"] = " ".join(c["name"].split()).title()
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"contacts={total} phone_ok={ok} fixed={fixed} unfixable={unfixable}")
    print(f"wrote {out_path}")
    rate = ok / total if total else 0
    return 0 if rate >= 0.98 else 1


if __name__ == "__main__":
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/uc1-map-clean-contacts/merged.jsonl")
    raise SystemExit(main(path))
