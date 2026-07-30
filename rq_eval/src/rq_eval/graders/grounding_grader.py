"""T2 grounding adapter — threshold-in-code + atom (build order B5).

The provider returns a raw score; THIS grader applies ``grounding_tau`` from
config to produce the boolean, and logs both (raw score as evidence). This is
where float→boolean thresholding lives — never in the provider.
"""

from __future__ import annotations

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord
from rq_eval.providers.base import GroundingProvider


class GroundingGrader:
    """[T2] Grounds a claim against a source: raw_score ≥ tau → grounded."""

    def __init__(
        self,
        grounding: GroundingProvider,
        tau: float,
        logger: AtomLogger,
        stamp: tuple[str, str],
        grader_id: str,
        seed: int,
    ) -> None:
        """Inject provider, config tau, logger, model stamp, grader id, seed."""
        self._grounding = grounding
        self._tau = tau
        self._logger = logger
        self._model, self._version = stamp
        self._grader_id = grader_id
        self._seed = seed

    def raw(self, source: str, claim: str) -> float:
        """Return the raw grounding score without logging (for diagnostics)."""
        return self._grounding.check(source, claim).raw_score

    def check(
        self, *, subject: str, role: str, source: str, claim: str, weight: float = 1.0
    ) -> AtomRecord:
        """Raw = provider.check(source, claim); verdict = raw ≥ tau; log; return atom."""
        raw = self._grounding.check(source, claim).raw_score
        verdict = raw >= self._tau
        return self._logger.record(
            subject=subject,
            role=role,
            question=f"grounded? (tau={self._tau})",
            tier="T2",
            verdict=verdict,
            weight=weight,
            evidence=f"score={raw:.4f} tau={self._tau}",
            grader_id=self._grader_id,
            model=self._model,
            model_version=self._version,
            seed=self._seed,
        )
