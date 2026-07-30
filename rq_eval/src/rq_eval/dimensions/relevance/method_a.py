"""§3 step 1 — Method A: RAGAS answer-relevancy (diagnostic).

Generate N questions from the answer, embed them + the original question, and
score AR = (1/N) Σ cos(E_gi, E_o). The number is deterministic cosine — the model
never computes it. Diagnostic (pinned model + seed), not a gate.
"""

from __future__ import annotations

import math

from rq_eval.providers.base import EmbeddingProvider, GeneratorProvider, Vector

# Pinned reverse-question prompt (mock: [[repeat]] returns N copies of the
# answer; live: strip tag + unwrap {{ }} -> a real "generate a question" ask).
_PROMPT = "[[repeat]] Generate a question that this answer would answer. {{ {answer} }}"


class MethodAReverseQuestions:
    """[T3-gen question + T2 cosine] answer-relevancy score in [0, 1]."""

    def __init__(
        self, generator: GeneratorProvider, embedding: EmbeddingProvider, n: int, seed: int
    ) -> None:
        """Inject generator + embedding; pin the question count and seed."""
        self._generator = generator
        self._embedding = embedding
        self._n = n
        self._seed = seed

    def score(self, question: str, answer: str) -> float:
        """AR = mean cosine between the original question and reverse-questions."""
        gen = self._generator.generate(
            _PROMPT.replace("{answer}", answer), seed=self._seed, n=self._n
        )
        items = gen.items or [answer]
        vectors = self._embedding.embed([question, *items])
        origin = vectors[0]
        sims = [self._cosine(origin, v) for v in vectors[1:]]
        return sum(sims) / len(sims) if sims else 0.0

    @staticmethod
    def _cosine(a: Vector, b: Vector) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return dot / norm if norm else 0.0
