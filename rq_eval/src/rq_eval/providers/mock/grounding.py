"""Mock grounding — deterministic three-way entailment (design §1).

label from token coverage + a crude negation heuristic:
* coverage = |tokens(hypothesis) ∩ tokens(premise)| / |tokens(hypothesis)|
* negation mismatch (one side negated, the other not) with real overlap -> C
* else coverage ≥ entail_tau -> E ; else -> N
raw_score = coverage. Thresholds come from config.
"""

from __future__ import annotations

from rq_eval.providers.base import EntailmentResult, GroundingProvider
from rq_eval.providers.mock.deterministic_text import DeterministicText

_NEGATION = frozenset("not no never none cannot n't without fails fewer lower".split())


class MockGroundingProvider(GroundingProvider):
    """Deterministic keyword-overlap + negation three-way entailment."""

    def __init__(self, seed: int, entail_tau: float, contra_tau: float) -> None:
        """Seed the text model and store the E/C coverage thresholds."""
        self._dt = DeterministicText(seed)
        self._entail_tau = entail_tau
        self._contra_tau = contra_tau

    def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        """Classify hypothesis vs premise as E / N / C with a raw coverage score."""
        cov = self._dt.overlap(hypothesis, premise)
        if cov >= self._contra_tau and self._negation_mismatch(premise, hypothesis):
            return EntailmentResult(label="C", raw_score=cov)
        if cov >= self._entail_tau:
            return EntailmentResult(label="E", raw_score=cov)
        return EntailmentResult(label="N", raw_score=cov)

    def _negation_mismatch(self, premise: str, hypothesis: str) -> bool:
        p = any(t in _NEGATION for t in self._dt.tokens(premise))
        h = any(t in _NEGATION for t in self._dt.tokens(hypothesis))
        return p != h
