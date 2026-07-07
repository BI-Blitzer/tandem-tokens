"""Example gen_test verifier: score a candidate sentence on hitting a target
word count (7). Reads the candidate JSON on stdin, writes a verdict JSON on
stdout — the protocol ollamaq.verify expects.

  in:  {"sentence": "The deep blue ocean hides ancient secrets"}
  out: {"pass": true, "score": 1.0, "notes": "7 words"}
"""

import json
import sys

TARGET = 7


def main() -> None:
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
        sentence = str(data.get("sentence", "")) if isinstance(data, dict) else str(data)
    except json.JSONDecodeError:
        sentence = raw

    words = sentence.split()
    n = len(words)
    # graded score: 1.0 at exactly TARGET, decaying with distance, floored at 0.
    score = max(0.0, 1.0 - abs(n - TARGET) / TARGET)
    print(json.dumps({
        "pass": n == TARGET,
        "score": round(score, 4),
        "notes": f"{n} words",
    }))


if __name__ == "__main__":
    main()
