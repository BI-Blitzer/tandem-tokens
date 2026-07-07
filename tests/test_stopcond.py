"""Contract tests for the stop_when mini-expression evaluator."""

from __future__ import annotations

import pytest

from ollamaq.stopcond import should_stop, validate_stop_when


def test_empty_never_stops() -> None:
    assert should_stop("", 1.0, [1.0]) is False


def test_score_ge() -> None:
    assert should_stop("score>=0.95", 0.95, [0.95]) is True
    assert should_stop("score>=0.95", 0.94, [0.94]) is False


def test_score_gt_and_lt() -> None:
    assert should_stop("score>0.5", 0.6, [0.6]) is True
    assert should_stop("score<0.5", 0.4, [0.4]) is True
    assert should_stop("score>0.5", 0.5, [0.5]) is False


def test_no_improvement() -> None:
    # best hasn't moved for 3 iterations -> stop
    assert should_stop("no_improvement(3)", 0.8, [0.8, 0.8, 0.8, 0.8]) is True
    # still improving -> don't stop
    assert should_stop("no_improvement(3)", 0.9, [0.6, 0.7, 0.8, 0.9]) is False
    # not enough history yet
    assert should_stop("no_improvement(3)", 0.8, [0.8, 0.8]) is False


def test_or_combines() -> None:
    # neither score threshold nor no_improvement met
    assert should_stop("score>=0.99 or no_improvement(3)", 0.5, [0.4, 0.5]) is False
    # score threshold met
    assert should_stop("score>=0.99 or no_improvement(3)", 0.99, [0.99]) is True
    # no_improvement met
    assert should_stop("score>=0.99 or no_improvement(2)", 0.5, [0.5, 0.5, 0.5]) is True


def test_and_combines() -> None:
    assert should_stop("score>=0.5 and score<0.9", 0.7, [0.7]) is True
    assert should_stop("score>=0.5 and score<0.9", 0.95, [0.95]) is False


def test_validate_good() -> None:
    validate_stop_when("score>=0.95 or no_improvement(3)")  # no raise
    validate_stop_when("")


def test_validate_bad() -> None:
    with pytest.raises(ValueError, match="unparseable"):
        validate_stop_when("scoore >= 0.5")
    with pytest.raises(ValueError):
        validate_stop_when("no_improvement(abc)")
