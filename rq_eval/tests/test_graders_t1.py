"""B5 — T1 grader toolbox (pure, deterministic)."""

from __future__ import annotations

import pytest

from rq_eval.graders.t1 import T1Tools


@pytest.fixture
def t1() -> T1Tools:
    return T1Tools()


def test_extract_number_handles_units(t1: T1Tools) -> None:
    assert t1.extract_number("$1.2B") == pytest.approx(1.2e9)
    assert t1.extract_number("12 percent") == pytest.approx(12.0)
    assert t1.extract_number("1,200 units") == pytest.approx(1200.0)
    assert t1.extract_number("no digits here") is None


def test_numeric_match_exact_and_tolerant(t1: T1Tools) -> None:
    assert t1.numeric_match("$1.2B", "1.2 billion", tolerance=0.0) is True
    assert t1.numeric_match("$1.2B", "$1.3B", tolerance=0.0) is False  # must fail
    assert t1.numeric_match("$1.2B", "$1.3B", tolerance=0.1) is True  # within 10%
    assert t1.numeric_match("no num", "5", tolerance=0.5) is False


def test_citation_membership(t1: T1Tools) -> None:
    allowed = {"chunk-1", "chunk-2"}
    assert t1.citation_member("chunk-1", allowed) is True
    assert t1.citation_member("chunk-9", allowed) is False
    assert t1.citation_member(None, allowed) is False


def test_atomicity_and_split(t1: T1Tools) -> None:
    assert t1.is_atomic("Revenue rose 12 percent.") is True
    assert t1.is_atomic("Revenue rose and costs fell.") is False
    assert t1.conjunction_split("Revenue rose and costs fell") == ["Revenue rose", "costs fell"]


def test_word_count(t1: T1Tools) -> None:
    assert t1.word_count("one two three") == 3
