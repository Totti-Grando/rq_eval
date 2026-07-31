"""T2 grounding adapter — three-way entailment → atom (design §1/§6).

Wraps ``GroundingProvider.entails``: the provider returns an E/N/C label + raw
score; THIS grader turns it into the boolean downstream code reads
(``verdict = supported = label == 'E'``) and logs the atom. ``assess`` returns
the atom *and* the raw :class:`EntailmentResult` (label + confidence) so
groundedness/attribution can feed the conformal layer without a second call.
"""

from __future__ import annotations

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord
from rq_eval.providers.base import EntailmentResult, GroundingProvider


class GroundingGrader:
    """[T2] Three-way entailment adapter; verdict = supported (label == E)."""

    def __init__(
        self,
        grounding: GroundingProvider,
        logger: AtomLogger,
        stamp: tuple[str, str],
        grader_id: str,
        seed: int,
    ) -> None:
        """Inject provider, logger, model stamp, grader id, and seed."""
        self._grounding = grounding
        self._logger = logger
        self._model, self._version = stamp
        self._grader_id = grader_id
        self._seed = seed

    def classify(self, premise: str, hypothesis: str) -> EntailmentResult:
        """Return the raw E/N/C result without logging (diagnostics/pre-filter)."""
        return self._grounding.entails(premise, hypothesis)

    def raw(self, source: str, claim: str) -> float:
        """Return the raw entailment score without logging."""
        return self._grounding.entails(source, claim).raw_score

    def assess(
        self, *, subject: str, role: str, premise: str, hypothesis: str, weight: float = 1.0
    ) -> tuple[AtomRecord, EntailmentResult]:
        """Classify, log the atom (verdict = supported), return (atom, result)."""
        res = self._grounding.entails(premise, hypothesis)
        atom = self._logger.record(
            subject=subject,
            role=role,
            question=f"entailed? ({role})",
            tier="T2",
            verdict=res.supported,
            weight=weight,
            evidence=f"label={res.label} score={res.raw_score:.4f}",
            grader_id=self._grader_id,
            model=self._model,
            model_version=self._version,
            seed=self._seed,
        )
        return atom, res

    def check(
        self, *, subject: str, role: str, source: str, claim: str, weight: float = 1.0
    ) -> AtomRecord:
        """Convenience: log + return only the atom (premise=source, hyp=claim)."""
        return self.assess(
            subject=subject, role=role, premise=source, hypothesis=claim, weight=weight
        )[0]
