"""E8 — conformal factuality (§5), deterministic + per-stratum."""

from __future__ import annotations

import pytest

from rq_eval.audit.calibration import CalibrationStore
from rq_eval.config import load_config
from rq_eval.scoring.conformal import ConformalCalibrator, ConformalStratifier


def test_threshold_and_band_deterministic() -> None:
    cal = ConformalCalibrator(alpha=0.05, min_n=5)
    confidences = [0.9, 0.8, 0.85, 0.7, 0.95, 0.6, 0.75, 0.88, 0.92, 0.65]
    r1 = cal.calibrate(confidences)
    r2 = cal.calibrate(list(confidences))
    assert r1 == r2  # deterministic / replays exactly
    assert not r1.abstained
    assert 0.0 <= r1.threshold <= 1.0
    assert r1.band_low == pytest.approx(0.95)
    assert r1.band_high == pytest.approx(min(1.0, 0.95 + 1.0 / (len(confidences) + 1)))


def test_min_n_abstains_and_retains_all() -> None:
    cal = ConformalCalibrator(alpha=0.05, min_n=10)
    r = cal.calibrate([0.9, 0.8, 0.7])  # n < min_n
    assert r.abstained is True
    assert cal.retain(0.0, r) is True  # abstained -> retain everything


def test_retain_respects_threshold() -> None:
    cal = ConformalCalibrator(alpha=0.1, min_n=5)
    r = cal.calibrate([0.9, 0.8, 0.85, 0.7, 0.95, 0.6])
    assert cal.retain(r.threshold + 0.05, r) is True
    assert cal.retain(r.threshold - 0.05, r) is False


def test_per_stratum_thresholds_from_calibration_fixture() -> None:
    """Confidences from the real calibration set (via mock grounding) differ by stratum."""
    from rq_eval.providers.factory import ProviderFactory

    cfg = load_config()
    grounding = ProviderFactory(cfg).build().grounding
    store = CalibrationStore(cfg)
    points = [
        (grounding.entails(ex.context, ex.claim).raw_score, ex.label, ex.stratum)
        for ex in store.examples()
    ]
    strat = ConformalStratifier(ConformalCalibrator(alpha=0.2, min_n=2))
    results = strat.calibrate(points, per_stratum=True)
    assert "__global__" in results
    assert "finance" in results and "sports" in results
    # deterministic replay
    assert strat.calibrate(points, per_stratum=True)["finance"] == results["finance"]
    # marginal-only mode yields just the global result
    assert set(strat.calibrate(points, per_stratum=False)) == {"__global__"}
