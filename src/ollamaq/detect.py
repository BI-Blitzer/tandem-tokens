"""Heuristic detector for offloadable bulk-data requests, used by the
UserPromptSubmit hook (hooks/auto_offload.py) to nudge Claude toward /offload.

Pure and testable: `detect_offloadable(prompt)` returns a short suggestion string
(naming the likely mode + task_tag) or None. Deliberately conservative — it
should fire on clear bulk-data tasks, not on ordinary questions.
"""

from __future__ import annotations

import re
from typing import Optional

# mechanical task verbs -> the task_tag / mode they imply
_TASK_TAGS = {
    "clean": "clean", "normalize": "clean", "normalise": "clean", "dedupe": "clean",
    "deduplicate": "clean", "transform": "clean", "reformat": "clean",
    "extract": "extract", "parse": "extract", "pull out": "extract",
    "classify": "classify", "categorize": "classify", "categorise": "classify",
    "label": "classify", "tag": "classify", "triage": "classify",
    "summarize": "summarize", "summarise": "summarize",
    "embed": "embed", "vectorize": "embed", "vectorise": "embed",
    "translate": "code",
}
# bulk-quantity signals
_BULK = re.compile(
    r"\b(all|each|every|these|those|bulk|batch(?:es)?|"
    r"\d{2,}\s*(?:rows|records|items|files|documents|docs|lines|entries|tickets|"
    r"reviews|logs|rows?|emails|messages|comments))\b", re.I)
# data nouns that suggest a dataset even without a quantity
_DATA = re.compile(
    r"\b(csv|jsonl?|tsv|spreadsheet|dataset|corpus|rows|records|"
    r"a (?:bunch|list|batch|set) of|these files|each file|the files)\b", re.I)
_EMBED_HINT = re.compile(r"\b(embed|embedding|vector|semantic search|rag|similarity)\b", re.I)
_GENTEST_HINT = re.compile(r"\b(generate .* and (?:test|score|verify)|candidates?|"
                           r"try .* variations)\b", re.I)


def detect_offloadable(prompt: str) -> Optional[str]:
    """Return a one-line offload suggestion, or None if the prompt doesn't look
    like a bulk mechanical data task."""
    if not prompt or len(prompt) > 4000:
        return None
    low = prompt.lower()

    verb = next((v for v in _TASK_TAGS if v in low), None)
    has_bulk = bool(_BULK.search(prompt))
    has_data = bool(_DATA.search(prompt))
    embed = bool(_EMBED_HINT.search(prompt))
    gentest = bool(_GENTEST_HINT.search(prompt))

    # Fire when there's a bulk/data signal AND a task intent (a mechanical verb,
    # an embed ask, or a generate-and-test ask).
    if not ((has_bulk or has_data) and (verb or embed or gentest)):
        return None

    if embed:
        mode, tag = "embed", "embed"
    elif gentest:
        mode, tag = "gen_test", "generate"
    else:
        mode, tag = "map", (_TASK_TAGS.get(verb, "clean") if verb else "clean")

    return (f"This looks like bulk {tag} work that tandem-tokens can offload to a "
            f"local model ({mode} mode) for ~0 Claude tokens. Consider /offload "
            f"(check `ollamaq status`/`ollamaq recommend` first). Skip if it needs "
            f"repo-wide context or has no scriptable check.")
