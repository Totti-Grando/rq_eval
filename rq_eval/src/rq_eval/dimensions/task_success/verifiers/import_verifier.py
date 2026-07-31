"""[import] grounded/responsive verifier — reuse accuracy/relevance checks (§4).

Rather than couple to the other dimensions' atoms, this reuses the same T2
machinery: ``responsive`` = relevance(question, answer) ≥ tau; ``grounded`` =
grounding(context, answer) ≥ tau (satisfied when no context is available).
"""

from __future__ import annotations

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord
from rq_eval.dimensions.task_success.task_templates import Outcome
from rq_eval.dimensions.task_success.verifiers.base import Verifier, VerifyContext
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.graders.relevance_grader import RelevanceGrader


class ImportVerifier(Verifier):
    """[import] Reuses grounding (grounded) / relevance (responsive) checks."""

    def __init__(
        self, grounding: GroundingGrader, relevance: RelevanceGrader, logger: AtomLogger
    ) -> None:
        """Inject the grounding + relevance graders and a logger (no-context case)."""
        self._grounding = grounding
        self._relevance = relevance
        self._logger = logger

    def verify(self, outcome: Outcome, ctx: VerifyContext) -> AtomRecord:
        """Route on ``params.signal`` to the imported grounded/responsive check."""
        signal = str(outcome.params.get("signal", "responsive"))
        if signal == "responsive":
            return self._relevance.check(
                subject=f"outcome:{outcome.id}", role="outcome",
                query=ctx.question, response=ctx.answer, weight=outcome.weight,
            )
        if ctx.context_text:
            return self._grounding.check(
                subject=f"outcome:{outcome.id}", role="outcome",
                source=ctx.context_text, claim=ctx.answer, weight=outcome.weight,
            )
        # no context to contradict -> grounded is satisfied (T1)
        return self._logger.record(
            subject=f"outcome:{outcome.id}", role="outcome",
            question=f"grounded? {outcome.text}", tier="T1", verdict=True,
            weight=outcome.weight, evidence="verifier=import signal=grounded no-context",
            grader_id="task_success.import_grounded", model="code", model_version="rq_eval",
        )
