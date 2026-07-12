"""Quality residue fix: rebuild clean contacts from source CSV + model name/email hints.

For each contact id in source CSV:
  - name/email: prefer model output if present, else title-case / lower from source
  - phone: ALWAYS re-derive from source digits (model often drops a digit)

Writes results/<job>/merged.normalized.jsonl in the same chunked shape as map output.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


def digits_phone(raw: str) -> str:
    d = re.sub(r"\D", "", raw or "")
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    if len(d) > 10:
        d = d[-10:]
    if len(d) != 10:
        return ""  # unfixable
    return f"{d[:3]}-{d[3:6]}-{d[6:]}"


def title_name(s: str) -> str:
    return " ".join((s or "").split()).title()


def main(source_csv: Path, merged_jsonl: Path) -> int:
    src: dict[int, dict] = {}
    with source_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            src[int(row["id"])] = row

    # gather model outputs by id
    model: dict[int, dict] = {}
    chunks: list[dict] = []
    for line in merged_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        chunks.append(obj)
        for c in (obj.get("value") or {}).get("contacts") or []:
            model[int(c["id"])] = c

    out_contacts: list[dict] = []
    ok = 0
    for i, row in sorted(src.items()):
        m = model.get(i, {})
        phone = digits_phone(row.get("phone", ""))
        if phone:
            ok += 1
        else:
            phone = digits_phone(m.get("phone", "")) or (m.get("phone") or row.get("phone") or "")
        out_contacts.append({
            "id": i,
            "name": title_name(m.get("name") or row.get("name", "")),
            "email": (m.get("email") or row.get("email", "")).strip().lower(),
            "phone": phone,
        })

    # rewrite as single merged line for easy harvest (also keep chunked file)
    out = merged_jsonl.with_name("merged.normalized.jsonl")
    with out.open("w", encoding="utf-8") as f:
        # one chunk of all contacts for simple consumers
        f.write(json.dumps({
            "index": 0,
            "source": f"normalized:{source_csv.name}",
            "value": {"contacts": out_contacts},
        }, ensure_ascii=False) + "\n")

    rate = ok / len(out_contacts) if out_contacts else 0
    print(f"contacts={len(out_contacts)} phone_ok={ok} rate={rate:.3f}")
    print(f"wrote {out}")
    return 0 if rate >= 0.99 else 1


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/hybrid_campaign/contacts_messy.csv")
    merged = Path(sys.argv[2] if len(sys.argv) > 2 else "results/uc1-map-clean-contacts/merged.jsonl")
    raise SystemExit(main(src, merged))
