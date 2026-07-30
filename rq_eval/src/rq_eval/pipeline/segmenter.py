"""§0 step 1 — segment the answer into sentences [T1]."""

from __future__ import annotations

from rq_eval.providers.base import NlpProvider


class Segmenter:
    """[T1] Deterministic sentence segmentation via the NLP provider."""

    def __init__(self, nlp: NlpProvider) -> None:
        """Inject the NLP provider (spaCy live, regex mock)."""
        self._nlp = nlp

    def segment(self, text: str) -> list[str]:
        """Split ``text`` into sentences. Inputs→ ordered list of sentences."""
        return self._nlp.segment(text)
