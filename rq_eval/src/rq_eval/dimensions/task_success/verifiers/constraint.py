"""[T1] constraint-satisfaction verifier — reuse constraint_compliance (§4).

Deterministic checks: required tokens present, forbidden tokens absent, and word
count within [min_words, max_words]. (A local stand-in for the out-of-scope
``constraint_compliance`` category.)
"""

from __future__ import annotations

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord
from rq_eval.dimensions.task_success.task_templates import Outcome
from rq_eval.dimensions.task_success.verifiers.base import Verifier, VerifyContext


class ConstraintVerifier(Verifier):
    """[T1] Achieved iff includes/excludes/length constraints all hold."""

    def __init__(self, logger: AtomLogger) -> None:
        """Inject the atom logger."""
        self._logger = logger

    def verify(self, outcome: Outcome, ctx: VerifyContext) -> AtomRecord:
        """Check include/exclude tokens and word-count bounds."""
        low = ctx.answer.lower()
        includes = [str(x).lower() for x in outcome.params.get("includes", [])]
        excludes = [str(x).lower() for x in outcome.params.get("excludes", [])]
        words = len(ctx.answer.split())
        max_words = outcome.params.get("max_words")
        min_words = outcome.params.get("min_words")

        ok = all(x in low for x in includes) and not any(x in low for x in excludes)
        if max_words is not None and words > int(max_words):
            ok = False
        if min_words is not None and words < int(min_words):
            ok = False
        return self._logger.record(
            subject=f"outcome:{outcome.id}", role="outcome",
            question=f"constraints met? {outcome.text}", tier="T1",
            verdict=ok, weight=outcome.weight,
            evidence=f"verifier=constraint words={words}",
            grader_id="task_success.constraint", model="code", model_version="rq_eval",
        )
