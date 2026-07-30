"""Aggregation helpers — off-ask cap + min-n abstention (build order B5).

Pure functions of their inputs; no model, no config object (callers pass the
config-sourced constants).
"""

from __future__ import annotations


class OffAskCap:
    """Caps a relevance score when the specific ask is not addressed."""

    def apply(self, score: float, on_ask: bool, cap: float) -> float:
        """Return ``score`` if ``on_ask`` else ``min(score, cap)``.

        Missing the specific ask caps the score regardless of on-topic volume.
        """
        return score if on_ask else min(score, cap)


class MinNAbstention:
    """Decides whether a proportion has too few observations to report."""

    def should_abstain(self, n: int, min_n: int) -> bool:
        """True iff ``n < min_n`` (e.g. < ~10 vital units → abstain)."""
        return n < min_n
