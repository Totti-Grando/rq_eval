"""T3 judge adapter — records a boolean verdict as an atom (build order B5)."""

from __future__ import annotations

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import Tier
from rq_eval.providers.base import JudgeProvider


class JudgeGrader:
    """[T3] Wraps a judge: ask a yes/no question, log the atom, return the bool."""

    def __init__(
        self,
        judge: JudgeProvider,
        logger: AtomLogger,
        stamp: tuple[str, str],
        grader_id: str,
        seed: int,
    ) -> None:
        """Inject judge, atom logger, model stamp, grader id, and seed."""
        self._judge = judge
        self._logger = logger
        self._model, self._version = stamp
        self._grader_id = grader_id
        self._seed = seed

    def judge(
        self,
        *,
        subject: str,
        role: str,
        question: str,
        context: str,
        weight: float = 1.0,
        tier: Tier = "T3",
    ) -> bool:
        """Call the judge, record an atom, and return the boolean verdict."""
        verdict = self._judge.binary(question, context)
        self._logger.record(
            subject=subject,
            role=role,
            question=question,
            tier=tier,
            verdict=verdict.verdict,
            weight=weight,
            evidence=verdict.reason,
            grader_id=self._grader_id,
            model=self._model,
            model_version=self._version,
            seed=self._seed,
        )
        return verdict.verdict
