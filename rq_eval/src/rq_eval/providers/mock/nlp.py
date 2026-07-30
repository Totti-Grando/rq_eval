"""Mock NLP — regex segmentation + identity coref (build order B2).

Deterministic stand-in for spaCy/coreferee so §0's pipeline runs offline. The
mock resolves a small set of leading pronouns to the last capitalized subject
seen in the carried context — enough to exercise the decontextualization path.
"""

from __future__ import annotations

import re

from rq_eval.providers.base import CorefResult, NlpProvider

_SENT = re.compile(r"(?<=[.!?])\s+")
_LEADING_PRONOUN = re.compile(r"^(he|she|it|they|this|that|these|those)\b", re.IGNORECASE)
_SUBJECT = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")


class MockNlpProvider(NlpProvider):
    """Deterministic regex-based segmentation and coreference resolution."""

    def __init__(self, seed: int) -> None:
        """Seed kept for signature parity with the live provider."""
        self._seed = seed

    def segment(self, text: str) -> list[str]:
        """[T1] Split into sentences on terminal punctuation + whitespace."""
        return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]

    def resolve_coref(self, text: str, context: str = "") -> CorefResult:
        """[T2] Replace a leading pronoun with the last subject in context."""
        subject = self._last_subject(context)
        if subject and _LEADING_PRONOUN.match(text.strip()):
            resolved = _LEADING_PRONOUN.sub(subject, text.strip(), count=1)
            return CorefResult(resolved_text=resolved)
        return CorefResult(resolved_text=text.strip())

    @staticmethod
    def _last_subject(context: str) -> str | None:
        matches = _SUBJECT.findall(context)
        return matches[-1] if matches else None
