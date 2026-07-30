"""Mock relevance — query/response token-overlap score (build order B2).

score = symmetric token Jaccard of query and response ∈ [0, 1]. Thresholding to
a boolean happens in our code (relevance_tau).
"""

from __future__ import annotations

from rq_eval.providers.base import RelevanceProvider
from rq_eval.providers.mock.deterministic_text import DeterministicText


class MockRelevanceProvider(RelevanceProvider):
    """Deterministic token-overlap relevance for offline runs."""

    def __init__(self, seed: int) -> None:
        """Seed the deterministic text model."""
        self._dt = DeterministicText(seed)

    def score(self, query: str, response: str) -> float:
        """Score = jaccard(tokens(query), tokens(response)) ∈ [0, 1]."""
        return self._dt.jaccard(query, response)
