"""Mock judge — deterministic boolean verdicts (build order B2).

Verdicts are reproducible and, where it matters, content-driven. An optional
leading ``[[tag]]`` in the question selects the deterministic rule so later
phases can make the mock behave sensibly without any model:

* ``[[affirm]]``          -> always True
* ``[[deny]]``            -> always False
* ``[[overlap]]``         -> True iff >=50% of the question's content tokens
                             appear in the context
* ``[[overlap:<tau>]]``   -> same with an explicit coverage threshold
* (no tag)                -> deterministic seeded-hash bit

There is no numeric output — this satisfies the booleans-only judge interface.
"""

from __future__ import annotations

import re

from rq_eval.providers.base import JudgeProvider, JudgeVerdict
from rq_eval.providers.mock.deterministic_text import DeterministicText

_TAG = re.compile(r"^\s*\[\[([^\]]+)\]\]\s*(.*)$", re.DOTALL)


class MockJudgeProvider(JudgeProvider):
    """Deterministic, tag-dispatched boolean judge for offline runs."""

    def __init__(self, seed: int) -> None:
        """Seed the deterministic text model."""
        self._dt = DeterministicText(seed)

    def binary(self, question: str, context: str) -> JudgeVerdict:
        """Return a deterministic verdict per the optional ``[[tag]]`` rule."""
        tag, body = self._parse(question)
        if tag == "affirm":
            return JudgeVerdict(True, "mock:affirm")
        if tag == "deny":
            return JudgeVerdict(False, "mock:deny")
        if tag and tag.startswith("overlap"):
            tau = self._tau(tag, default=0.5)
            cov = self._dt.overlap(body, context)
            return JudgeVerdict(cov >= tau, f"mock:overlap cov={cov:.3f} tau={tau:.3f}")
        bit = self._dt.bit(question, context)
        return JudgeVerdict(bit, "mock:seeded")

    @staticmethod
    def _parse(question: str) -> tuple[str | None, str]:
        m = _TAG.match(question)
        return (m.group(1).strip(), m.group(2)) if m else (None, question)

    @staticmethod
    def _tau(tag: str, default: float) -> float:
        if ":" in tag:
            try:
                return float(tag.split(":", 1)[1])
            except ValueError:
                return default
        return default
