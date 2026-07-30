"""T2 relevance adapter — threshold-in-code + atom (build order B5).

Method-B style: the provider returns a raw query↔response score; THIS grader
applies ``relevance_tau`` to produce the boolean and logs both.
"""

from __future__ import annotations

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.providers.base import RelevanceProvider


class RelevanceGrader:
    """[T2] Scores response relevance to a query: raw ≥ tau → relevant."""

    def __init__(
        self,
        relevance: RelevanceProvider,
        tau: float,
        logger: AtomLogger,
        stamp: tuple[str, str],
        grader_id: str,
        seed: int,
    ) -> None:
        """Inject provider, config tau, logger, model stamp, grader id, seed."""
        self._relevance = relevance
        self._tau = tau
        self._logger = logger
        self._model, self._version = stamp
        self._grader_id = grader_id
        self._seed = seed

    def raw(self, query: str, response: str) -> float:
        """Return the raw relevance score without logging (for diagnostics)."""
        return self._relevance.score(query, response)

    def check(
        self, *, subject: str, role: str, query: str, response: str, weight: float = 1.0
    ) -> bool:
        """Raw = provider.score(query, response); verdict = raw ≥ tau; log; return."""
        raw = self._relevance.score(query, response)
        verdict = raw >= self._tau
        self._logger.record(
            subject=subject,
            role=role,
            question=f"relevant? (tau={self._tau})",
            tier="T2",
            verdict=verdict,
            weight=weight,
            evidence=f"score={raw:.4f} tau={self._tau}",
            grader_id=self._grader_id,
            model=self._model,
            model_version=self._version,
            seed=self._seed,
        )
        return verdict
