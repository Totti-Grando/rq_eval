"""T3 judge adapter — records a boolean ScoringJudge verdict as an atom (B5/R1)."""

from __future__ import annotations

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, Tier
from rq_eval.providers.base import ScoringJudge


class JudgeGrader:
    """[T3] Wraps a ScoringJudge: ask a yes/no question, log the atom, return bool."""

    def __init__(
        self,
        judge: ScoringJudge,
        logger: AtomLogger,
        stamp: tuple[str, str],
        grader_id: str,
        seed: int,
    ) -> None:
        """Inject scoring judge, atom logger, model stamp, grader id, and seed."""
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
        reference: str | None = None,
        weight: float = 1.0,
        tier: Tier = "T3",
    ) -> AtomRecord:
        """Call the judge (reference-grounded when given), log + return the atom."""
        verdict = self._judge.binary(question, context, reference)
        return self._logger.record(
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
