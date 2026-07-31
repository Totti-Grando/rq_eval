"""[T1] state/end-condition verifier — terminal-state match (§4, τ-bench style).

Compares the answer/artifact against an expected end-state supplied in the
outcome params (substring or exact match). When no ground-truth state is given,
the outcome is treated as satisfied (nothing to contradict).
"""

from __future__ import annotations

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord
from rq_eval.dimensions.task_success.task_templates import Outcome
from rq_eval.dimensions.task_success.verifiers.base import Verifier, VerifyContext


class StateVerifier(Verifier):
    """[T1] Achieved iff the answer matches the expected terminal state."""

    def __init__(self, logger: AtomLogger) -> None:
        """Inject the atom logger."""
        self._logger = logger

    def verify(self, outcome: Outcome, ctx: VerifyContext) -> AtomRecord:
        """Match ``params.expected`` against the answer (exact or substring)."""
        expected = outcome.params.get("expected")
        exact = bool(outcome.params.get("exact", False))
        if expected is None:
            achieved, evidence = True, "no expected state -> satisfied"
        elif exact:
            achieved = ctx.answer.strip() == str(expected).strip()
            evidence = "exact-match"
        else:
            achieved = str(expected).lower() in ctx.answer.lower()
            evidence = "substring-match"
        return self._logger.record(
            subject=f"outcome:{outcome.id}", role="outcome",
            question=f"state matches? {outcome.text}", tier="T1",
            verdict=achieved, weight=outcome.weight,
            evidence=f"verifier=state {evidence}",
            grader_id="task_success.state", model="code", model_version="rq_eval",
        )
