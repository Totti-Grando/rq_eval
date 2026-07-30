"""§3 step 2 — Method B: Bedrock contextual-grounding relevance (default).

A fixed query↔response relevance score, returned raw here and thresholded in
our code. Deterministic-first: no generation step.
"""

from __future__ import annotations

from rq_eval.graders.relevance_grader import RelevanceGrader


class MethodBGuardrail:
    """[T2] Raw query↔response relevance score via the relevance provider."""

    def __init__(self, relevance_grader: RelevanceGrader) -> None:
        """Inject the relevance grader (wraps the provider + config tau)."""
        self._grader = relevance_grader

    def score(self, question: str, answer: str) -> float:
        """Return the raw relevance score (no threshold applied)."""
        return self._grader.raw(question, answer)
