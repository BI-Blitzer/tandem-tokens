"""Evaluate a gen_test `stop_when` mini-expression.

Grammar (deliberately tiny — no eval, no arbitrary code):

    expr   := term ( ' or ' term )*        # OR of terms (any true -> stop)
    term   := atom ( ' and ' atom )*       # AND of atoms (all true -> term true)
    atom   := 'score' ('>='|'>'|'<='|'<'|'==') NUMBER
            | 'no_improvement' '(' INT ')'

`score` refers to the best score achieved so far. `no_improvement(K)` is true
when the best-so-far has not increased across the last K iterations.

An empty expression never stops (loop runs to max_iterations). A malformed
expression raises ValueError at parse time so a bad job fails fast, not silently.
"""

from __future__ import annotations

import re

_SCORE_RE = re.compile(r"^score\s*(>=|<=|==|>|<)\s*(-?\d+(?:\.\d+)?)$")
_NOIMP_RE = re.compile(r"^no_improvement\(\s*(\d+)\s*\)$")


def _eval_atom(atom: str, best_score: float, best_history: list[float]) -> bool:
    atom = atom.strip()
    m = _SCORE_RE.match(atom)
    if m:
        op, num = m.group(1), float(m.group(2))
        return {
            ">=": best_score >= num,
            "<=": best_score <= num,
            "==": best_score == num,
            ">": best_score > num,
            "<": best_score < num,
        }[op]
    m = _NOIMP_RE.match(atom)
    if m:
        k = int(m.group(1))
        # Need at least k+1 recorded bests to judge k iterations of flatness.
        if len(best_history) <= k:
            return False
        return best_history[-1] <= best_history[-1 - k]
    raise ValueError(f"unparseable stop_when atom: {atom!r}")


def should_stop(expr: str, best_score: float, best_history: list[float]) -> bool:
    """True if the loop should stop now. `best_history` is the best-so-far score
    recorded after each completed iteration (monotonic non-decreasing)."""
    expr = expr.strip()
    if not expr:
        return False
    for or_term in expr.split(" or "):
        if all(_eval_atom(atom, best_score, best_history)
               for atom in or_term.split(" and ")):
            return True
    return False


def validate_stop_when(expr: str) -> None:
    """Parse-check an expression without needing runtime state. Raises ValueError
    on any unparseable atom. A no-op for the empty string."""
    if not expr.strip():
        return
    for or_term in expr.split(" or "):
        for atom in or_term.split(" and "):
            _eval_atom(atom, 0.0, [])
