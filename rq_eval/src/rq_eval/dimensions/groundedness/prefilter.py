"""§1 step 1 — similarity pre-filter [T1].

For each triplet, pick the nearest context span (Titan embeddings; mock: hashed
vectors) to hand the verifier a focused premise. Cheap, deterministic, and
**not the score** — it only chooses which span to entail against.
"""

from __future__ import annotations

import math

from rq_eval.providers.base import EmbeddingProvider, Vector


class SimilarityPreFilter:
    """[T1] Selects the context span most similar to a hypothesis."""

    def __init__(self, embedding: EmbeddingProvider) -> None:
        """Inject the embedding provider."""
        self._embedding = embedding

    def select(self, hypothesis: str, spans: list[str]) -> str:
        """Return the span with the highest cosine to ``hypothesis`` ("" if none)."""
        if not spans:
            return ""
        vectors = self._embedding.embed([hypothesis, *spans])
        origin = vectors[0]
        best_idx = max(range(len(spans)), key=lambda i: self._cosine(origin, vectors[i + 1]))
        return spans[best_idx]

    @staticmethod
    def _cosine(a: Vector, b: Vector) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return dot / norm if norm else 0.0
