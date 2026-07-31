"""§2 step 4 — merge/dedupe near-duplicate units [T2 embeddings + code].

Greedy clustering: a unit is dropped if its embedding cosine with an
already-kept unit is ≥ ``completeness.dedupe_tau``. Order-stable (keeps first).
"""

from __future__ import annotations

import math

from rq_eval.dimensions.completeness.unit import Unit
from rq_eval.providers.base import EmbeddingProvider, Vector


class UnitDeduper:
    """[T2] Clusters near-duplicate units by embedding cosine."""

    def __init__(self, embedding: EmbeddingProvider, tau: float) -> None:
        """Inject the embedding provider and the dedupe cosine threshold."""
        self._embedding = embedding
        self._tau = tau

    def dedupe(self, units: list[Unit]) -> list[Unit]:
        """Return units with near-duplicates removed (first of each cluster)."""
        if not units:
            return []
        vectors = self._embedding.embed([u.text for u in units])
        kept: list[Unit] = []
        kept_vectors: list[Vector] = []
        for unit, vec in zip(units, vectors, strict=True):
            if any(self._cosine(vec, kv) >= self._tau for kv in kept_vectors):
                continue
            kept.append(unit)
            kept_vectors.append(vec)
        return kept

    @staticmethod
    def _cosine(a: Vector, b: Vector) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return dot / norm if norm else 0.0
