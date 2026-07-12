"""Scale hybrid campaign inputs for a larger daemon run."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "hybrid_campaign"
random.seed(99)


def scale_contacts(n: int = 1000) -> Path:
    src = list(csv.DictReader((OUT / "contacts_messy.csv").open(encoding="utf-8")))
    rows = []
    for i in range(1, n + 1):
        base = dict(src[(i - 1) % len(src)])
        base["id"] = str(i)
        # re-mess phone slightly
        d = "".join(str(random.randint(0, 9)) for _ in range(10))
        styles = [
            f"{d[:3]}-{d[3:6]}-{d[6:]}",
            f"({d[:3]}) {d[3:6]} {d[6:]}",
            f"{d[:3]}.{d[3:6]}.{d[6:]}",
            d,
        ]
        base["phone"] = random.choice(styles)
        rows.append(base)
    p = OUT / "contacts_scale_1k.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "email", "phone"])
        w.writeheader()
        w.writerows(rows)
    return p


def scale_tickets(n: int = 120) -> Path:
    src = [
        json.loads(line)
        for line in (OUT / "support_tickets.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    p = OUT / "support_tickets_scale.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i in range(1, n + 1):
            t = dict(src[(i - 1) % len(src)])
            t["id"] = i
            t["text"] = t["text"] + f" [scale#{i}]"
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return p


def write_scale_jobs(contacts: Path, tickets: Path) -> None:
    jobs = ROOT / "jobs"
    # UC1 scale
    j1 = json.loads((jobs / "uc1_map_clean_contacts.job.json").read_text(encoding="utf-8"))
    j1["name"] = "uc1-map-clean-scale-1k"
    j1["input"]["path"] = str(contacts.relative_to(ROOT)).replace("\\", "/")
    j1["input"]["chunk"]["size"] = 25
    (jobs / "uc1_map_clean_scale_1k.job.json").write_text(
        json.dumps(j1, indent=2), encoding="utf-8"
    )
    # UC2 scale
    j2 = json.loads((jobs / "uc2_pipeline_support.job.json").read_text(encoding="utf-8"))
    j2["name"] = "uc2-pipeline-support-scale"
    j2["input"]["path"] = str(tickets.relative_to(ROOT)).replace("\\", "/")
    (jobs / "uc2_pipeline_support_scale.job.json").write_text(
        json.dumps(j2, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    c = scale_contacts(1000)
    t = scale_tickets(120)
    write_scale_jobs(c, t)
    print("contacts", c)
    print("tickets", t)
    print("jobs: uc1_map_clean_scale_1k.job.json, uc2_pipeline_support_scale.job.json")
