"""[T2] coverage verifier — reuse completeness-style nugget recall (§4).

Does the answer cover a task requirement? Uses the grounding grader (answer =
premise, the requirement = hypothesis), thresholded in code.
"""

from __future__ import annotations

from rq_eval.contracts import AtomRecord
from rq_eval.dimensions.task_success.task_templates import Outcome
from rq_eval.dimensions.task_success.verifiers.base import Verifier, VerifyContext
from rq_eval.graders.grounding_grader import GroundingGrader


class CoverageVerifier(Verifier):
    """[T2] Achieved iff the answer covers the outcome's requirement."""

    def __init__(self, grounding: GroundingGrader) -> None:
        """Inject the grounding grader (bound to grounding_tau)."""
        self._grounding = grounding

    def verify(self, outcome: Outcome, ctx: VerifyContext) -> AtomRecord:
        """grounding(answer supports requirement) ≥ tau -> covered."""
        requirement = str(outcome.params.get("requirement", outcome.text))
        return self._grounding.check(
            subject=f"outcome:{outcome.id}", role="outcome",
            source=ctx.answer, claim=requirement, weight=outcome.weight,
        )
