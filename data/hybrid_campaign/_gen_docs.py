"""One-shot generator for hybrid campaign docs.jsonl."""
import json
from pathlib import Path

topics = [
    ("energy", "GPU inference energy and local vs cloud cost tradeoffs"),
    ("ops", "Daemon queue draining, model residency, and VRAM thrash avoidance"),
    ("security", "API keys, MCP verifier sandboxing, and untrusted networks"),
    ("data", "Schema-constrained decoding and repair-retry for bulk cleaning"),
    ("product", "Hybrid orchestrators: Claude, Grok, Codex sharing one harness"),
]
docs = []
for i in range(36):
    domain, title = topics[i % len(topics)]
    paras = []
    for j in range(4):
        paras.append(
            f'Section {j + 1} of "{title}" (doc {i}). '
            f"This discusses practical guidance for {domain} in local AI systems. "
            f"Operators should measure tokens, wall time, and watt-hours together. "
            f"Item reference CAMPAIGN-{i:03d}-{j}. Key claim: offload mechanical map work "
            f"and keep premium models for reduce and hard reasoning only."
        )
    body = " ".join(paras)
    docs.append({"id": f"DOC-{i:03d}", "domain": domain, "title": title, "body": body})

p = Path(__file__).with_name("docs.jsonl")
with p.open("w", encoding="utf-8") as f:
    for d in docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print("docs", len(docs), "avg_chars", sum(len(d["body"]) for d in docs) // len(docs), p)
