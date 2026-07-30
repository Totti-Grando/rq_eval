"""Mock embeddings — deterministic hashed vectors (build order B2)."""

from __future__ import annotations

from rq_eval.providers.base import EmbeddingProvider, Vector
from rq_eval.providers.mock.deterministic_text import DeterministicText


class MockEmbeddingProvider(EmbeddingProvider):
    """Fixed-dimension bag-of-hashed-tokens embeddings; cosine ~ token overlap."""

    def __init__(self, seed: int, dim: int = 64) -> None:
        """Seed the deterministic text model and fix the vector dimension."""
        self._dt = DeterministicText(seed, dim=dim)

    def embed(self, texts: list[str]) -> list[Vector]:
        """Return one L2-normalized vector per input text."""
        return [self._dt.embed(t) for t in texts]
