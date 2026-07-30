"""Mock grounding — keyword-overlap entailment score (build order B2).

raw_score = coverage of the claim's content tokens by the source ∈ [0, 1]:
a claim whose tokens all appear in the source scores ~1 (grounded); a claim
with no support scores ~0. Thresholding to a boolean happens in our code.
"""

from __future__ import annotations

from rq_eval.providers.base import GroundingProvider, GroundingResult
from rq_eval.providers.mock.deterministic_text import DeterministicText


class MockGroundingProvider(GroundingProvider):
    """Deterministic keyword-overlap grounding for offline runs."""

    def __init__(self, seed: int) -> None:
        """Seed the deterministic text model."""
        self._dt = DeterministicText(seed)

    def check(self, source: str, claim: str) -> GroundingResult:
        """raw_score = |tokens(claim) ∩ tokens(source)| / |tokens(claim)|."""
        return GroundingResult(raw_score=self._dt.overlap(claim, source))
