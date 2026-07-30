"""Importance weights for claims (§1 step 7).

A false *vital* claim costs more (importance-sensitive factuality). Weights come
from completeness's vital/okay unit labels when present; absent that, uniform.
Toggled by ``accuracy.importance_weighting`` (off => all weights 1.0).
"""

from __future__ import annotations


class ImportanceWeights:
    """Maps a claim id to its importance weight (default 1.0)."""

    def __init__(self, enabled: bool, weights: dict[str, float] | None = None) -> None:
        """Enable/disable weighting; optionally supply per-claim weights."""
        self._enabled = enabled
        self._weights = weights or {}

    def weight(self, claim_id: str) -> float:
        """Return the claim's weight (1.0 if disabled or unmapped)."""
        if not self._enabled:
            return 1.0
        return self._weights.get(claim_id, 1.0)
