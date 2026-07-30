"""B5 — scoring library: property tests (pure math)."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from rq_eval.scoring.aggregation import MinNAbstention, OffAskCap
from rq_eval.scoring.bands import BandMapper
from rq_eval.scoring.wilson import WilsonInterval


@given(n=st.integers(min_value=1, max_value=10_000), k=st.integers(min_value=0))
def test_wilson_interval_contains_phat_and_in_unit(n: int, k: int) -> None:
    k = min(k, n)
    low, high = WilsonInterval().interval(k, n)
    phat = k / n
    eps = 1e-9
    assert 0.0 <= low <= high <= 1.0
    assert low <= phat + eps
    assert phat <= high + eps


def test_wilson_zero_n_is_uninformative() -> None:
    assert WilsonInterval().interval(0, 0) == (0.0, 1.0)


@given(score=st.floats(min_value=0.0, max_value=1.0))
def test_band_thresholds(score: float) -> None:
    mapper = BandMapper(g=0.90, a=0.75)
    band = mapper.band(score)
    if score >= 0.90:
        assert band == "G"
    elif score >= 0.75:
        assert band == "A"
    else:
        assert band == "R"


@given(
    score=st.floats(min_value=0.0, max_value=1.0),
    cap=st.floats(min_value=0.0, max_value=1.0),
)
def test_off_ask_cap_never_increases(score: float, cap: float) -> None:
    cap_tool = OffAskCap()
    assert cap_tool.apply(score, on_ask=True, cap=cap) == score
    capped = cap_tool.apply(score, on_ask=False, cap=cap)
    assert capped <= score
    assert capped <= cap


@given(n=st.integers(min_value=0, max_value=100), min_n=st.integers(min_value=0, max_value=100))
def test_min_n_abstention(n: int, min_n: int) -> None:
    assert MinNAbstention().should_abstain(n, min_n) == (n < min_n)
