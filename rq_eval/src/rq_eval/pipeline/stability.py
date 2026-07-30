"""§0 step 5 — pin & measure decomposition stability.

Factual-precision pipelines are provably sensitive to the decomposition method
(Wanner et al., 2024), so we re-run extraction N times and report claim-set
agreement — the biggest reproducibility risk in the category.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rq_eval.pipeline.pipeline import ClaimPipeline


class StabilityHarness:
    """Re-runs extraction and reports claim-set agreement ∈ [0, 1]."""

    def __init__(self, pipeline: ClaimPipeline) -> None:
        """Inject the pipeline to re-run (logger-less passes)."""
        self._pipeline = pipeline

    def measure(self, answer: str, context: str, runs: int) -> float:
        """Agreement = |∩ claim-id sets| / |∪ claim-id sets| over ``runs`` passes.

        1.0 when every run yields the same claim set (always so under the
        deterministic mock); < 1.0 exposes live decomposition instability. An
        empty union (no claims) is defined as 1.0.
        """
        sets = [set(self._pipeline.claim_ids(answer, context)) for _ in range(max(1, runs))]
        union = set().union(*sets) if sets else set()
        if not union:
            return 1.0
        intersection = set(sets[0]).intersection(*sets[1:]) if len(sets) > 1 else sets[0]
        return len(intersection) / len(union)
