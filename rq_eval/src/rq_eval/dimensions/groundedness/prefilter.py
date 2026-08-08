"""§1 step 1 — similarity pre-filter [T1].

For each triplet, rank retrieved chunks by cosine (Titan embeddings; mock: hashed
vectors) and keep the top-``groundedness_k`` to hand the verifier a focused set of
premises. Cheap, deterministic, and **not the score** — it only *focuses* which
chunks are entailed against (handing the verifier the whole corpus degrades it).
"""

from __future__ import annotations

import math

from rq_eval.contracts import ContextChunk
from rq_eval.providers.base import EmbeddingProvider, Vector


class SimilarityPreFilter:
    """[T1] Ranks context chunks by similarity to a hypothesis."""

    def __init__(self, embedding: EmbeddingProvider) -> None:
        """Inject the embedding provider."""
        self._embedding = embedding

    def select_k(self, hypothesis: str, chunks: list[ContextChunk], k: int) -> list[ContextChunk]:
        """Return the top-``k`` chunks by cosine to ``hypothesis`` (order preserved by rank)."""
        if not chunks:
            return []
        vectors = self._embedding.embed([hypothesis, *(c.text for c in chunks)])
        origin = vectors[0]
        ranked = sorted(
            range(len(chunks)),
            key=lambda i: self._cosine(origin, vectors[i + 1]),
            reverse=True,
        )
        return [chunks[i] for i in ranked[: max(1, k)]]

    @staticmethod
    def _cosine(a: Vector, b: Vector) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return dot / norm if norm else 0.0
