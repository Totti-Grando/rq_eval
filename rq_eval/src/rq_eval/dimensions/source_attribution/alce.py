"""§4 step 2 — ALCE citation recall + precision [code].

* citation recall = fraction of cited statements whose citation *set* supports them;
* citation precision = fraction of individual citations that are relevant (support).

Computed purely from the per-citation Attributable booleans. With one citation
per claim the two coincide; the code generalizes to multi-citation statements.
"""

from __future__ import annotations


class AlceScorer:
    """Computes ALCE citation recall + precision from per-citation verdicts."""

    def recall(self, statement_supported: list[bool]) -> float:
        """|statements whose citation set supports them| / |cited statements|."""
        if not statement_supported:
            return 0.0
        return sum(1 for s in statement_supported if s) / len(statement_supported)

    def precision(self, citation_relevant: list[bool]) -> float:
        """|relevant individual citations| / |total individual citations|."""
        if not citation_relevant:
            return 0.0
        return sum(1 for c in citation_relevant if c) / len(citation_relevant)
