"""Deterministic text model shared by the mock providers (build order B2).

Pure, seedable, offline text heuristics — token overlap and hashed embeddings —
that let every mock provider return semantically-plausible **and** perfectly
reproducible outputs. No calculations here feed a real score; these only make
the mock path deterministic and exercise every downstream code path.
"""

from __future__ import annotations

import hashlib
import math
import re

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the of to in on for and or is are was were be been it its this that "
    "with as at by from into about over under how why what which who".split()
)


class DeterministicText:
    """Seedable token-overlap + hashed-embedding utilities for mock providers."""

    def __init__(self, seed: int, dim: int = 64) -> None:
        """Store the seed and embedding dimension."""
        self._seed = seed
        self._dim = dim

    @staticmethod
    def tokens(text: str) -> list[str]:
        """Lowercase content tokens (stopwords removed)."""
        return [t for t in _WORD.findall(text.lower()) if t not in _STOP]

    def overlap(self, a: str, b: str) -> float:
        """Directional coverage of ``a``'s tokens by ``b`` ∈ [0, 1].

        coverage = |tokens(a) ∩ tokens(b)| / |tokens(a)|  (0 if a has none).
        """
        ta = set(self.tokens(a))
        if not ta:
            return 0.0
        tb = set(self.tokens(b))
        return len(ta & tb) / len(ta)

    def jaccard(self, a: str, b: str) -> float:
        """Symmetric token Jaccard of ``a`` and ``b`` ∈ [0, 1]."""
        ta, tb = set(self.tokens(a)), set(self.tokens(b))
        if not ta and not tb:
            return 1.0
        union = ta | tb
        return len(ta & tb) / len(union) if union else 0.0

    def bit(self, *parts: str) -> bool:
        """Deterministic pseudo-random bit from the seed + parts."""
        h = hashlib.sha256(("|".join((str(self._seed), *parts))).encode()).digest()
        return bool(h[0] & 1)

    def embed(self, text: str) -> list[float]:
        """Fixed-dim L2-normalized bag-of-hashed-tokens embedding.

        Cosine similarity of two such vectors rises with shared tokens, so
        Method-A cosine behaves sensibly offline.
        """
        vec = [0.0] * self._dim
        for tok in self.tokens(text):
            idx = int.from_bytes(hashlib.md5(f"{self._seed}:{tok}".encode()).digest()[:4], "big")
            vec[idx % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # deterministic non-zero unit vector for empty/stopword-only text
            vec[self._seed % self._dim] = 1.0
            return vec
        return [v / norm for v in vec]
