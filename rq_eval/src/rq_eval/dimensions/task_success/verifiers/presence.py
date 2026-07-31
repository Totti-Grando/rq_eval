"""[T1] artifact-presence verifier — parse/structure check (§4)."""

from __future__ import annotations

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord
from rq_eval.dimensions.task_success.task_templates import Outcome
from rq_eval.dimensions.task_success.verifiers.base import Verifier, VerifyContext


class PresenceVerifier(Verifier):
    """[T1] Achieved iff the answer contains any of the outcome's patterns."""

    def __init__(self, logger: AtomLogger) -> None:
        """Inject the atom logger."""
        self._logger = logger

    def verify(self, outcome: Outcome, ctx: VerifyContext) -> AtomRecord:
        """Achieved = any pattern substring present in the answer (case-insensitive)."""
        patterns = [str(p) for p in outcome.params.get("patterns", [])]
        low = ctx.answer.lower()
        hit = next((p for p in patterns if p.lower() in low), None)
        return self._logger.record(
            subject=f"outcome:{outcome.id}", role="outcome",
            question=f"artifact present? {outcome.text}", tier="T1",
            verdict=hit is not None, weight=outcome.weight,
            evidence=f"verifier=artifact_presence match={hit!r}",
            grader_id="task_success.presence", model="code", model_version="rq_eval",
        )
