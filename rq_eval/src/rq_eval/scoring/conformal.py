"""§5 conformal factuality — split-conformal guarantee (build order E8).

Distribution-free, finite-sample: on a human-labeled calibration set of ``n``
factual claims with verifier confidences, set the threshold to the conformal
quantile and retain claims above it. The retained set is factual with a
guarantee band ``[1−α, 1−α+1/(n+1)]`` (Mohri & Hashimoto 2024; Angelopoulos &
Bates 2023). Pure math — no model here (a confidence is passed in), so this stays
in the ``scoring/`` island.

Formula (split conformal, Vovk et al. 2005):
    νᵢ = 1 − confidenceᵢ            (nonconformity of a factual example)
    k  = ⌈(1−α)(n+1)⌉  (clamped to n)
    τ̂  = the k-th smallest νᵢ        (= Quantile_{⌈(1−α)(n+1)⌉/n})
    retain a claim iff confidence ≥ 1 − τ̂
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_GLOBAL = "__global__"


@dataclass(frozen=True, slots=True)
class ConformalResult:
    """A calibrated conformal threshold + its guarantee band."""

    threshold: float  # retain iff confidence >= threshold
    tau_hat: float
    band_low: float
    band_high: float
    n: int
    abstained: bool


class ConformalCalibrator:
    """Split-conformal calibration over factual-example confidences (pure)."""

    def __init__(self, alpha: float, min_n: int) -> None:
        """Store the error budget ``alpha`` and the min calibration size."""
        self._alpha = alpha
        self._min_n = min_n

    def calibrate(self, factual_confidences: list[float]) -> ConformalResult:
        """Return the conformal threshold + band for these factual confidences."""
        n = len(factual_confidences)
        band_low = 1.0 - self._alpha
        if n < self._min_n:  # too few points -> abstain from the guarantee (retain all)
            return ConformalResult(0.0, 1.0, band_low, 1.0, n, abstained=True)
        nu = sorted(1.0 - c for c in factual_confidences)
        k = min(math.ceil((1.0 - self._alpha) * (n + 1)), n)
        tau_hat = nu[k - 1]
        band_high = min(1.0, band_low + 1.0 / (n + 1))
        return ConformalResult(1.0 - tau_hat, tau_hat, band_low, band_high, n, abstained=False)

    def retain(self, confidence: float, result: ConformalResult) -> bool:
        """Retain a claim iff confidence ≥ threshold (always, if abstained)."""
        return True if result.abstained else confidence >= result.threshold


class ConformalStratifier:
    """Marginal (global) or per-stratum calibration from labeled points."""

    def __init__(self, calibrator: ConformalCalibrator) -> None:
        """Inject the calibrator."""
        self._calibrator = calibrator

    def calibrate(
        self, points: list[tuple[float, bool, str]], per_stratum: bool
    ) -> dict[str, ConformalResult]:
        """Calibrate from ``(confidence, label, stratum)`` points.

        Always includes a ``__global__`` result; per-stratum results are added
        when ``per_stratum`` is set (with the global as fallback).
        """
        results = {_GLOBAL: self._calibrator.calibrate([c for c, ok, _ in points if ok])}
        if per_stratum:
            for stratum in sorted({s for _, _, s in points}):
                results[stratum] = self._calibrator.calibrate(
                    [c for c, ok, s in points if ok and s == stratum]
                )
        return results

    @staticmethod
    def result_for(results: dict[str, ConformalResult], stratum: str) -> ConformalResult:
        """Return the stratum's result, falling back to the global one."""
        return results.get(stratum, results[_GLOBAL])
