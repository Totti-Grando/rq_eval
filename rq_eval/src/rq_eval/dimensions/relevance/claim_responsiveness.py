"""§3 step 4 — per-claim responsive atom (on-topic ∧ on-ask) [T2].

For each claim we compute two T2 signals and AND them into the single
``responsive`` atom that accuracy imports:

* on-topic — symmetric relevance of the claim to the question (relevance provider);
* on-ask  — the claim covers the question's specific terms (judge coverage).

The subtle "on-topic but answers a different sub-question" case (on-topic ∧
¬on-ask) is sent to a thin [T3] residual judge for audit only.
"""

from __future__ import annotations

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, Claim
from rq_eval.dimensions.responsiveness import ResponsivenessExport
from rq_eval.graders.judge_grader import JudgeGrader
from rq_eval.graders.relevance_grader import RelevanceGrader


class ClaimResponsiveness:
    """Computes + logs the per-claim responsive atom and publishes it."""

    def __init__(
        self,
        on_topic: RelevanceGrader,
        on_ask: JudgeGrader,
        residual: JudgeGrader,
        logger: AtomLogger,
        stamp: tuple[str, str],
        seed: int,
    ) -> None:
        """Inject the on-topic/on-ask/residual graders, logger, stamp, seed."""
        self._on_topic = on_topic
        self._on_ask = on_ask
        self._residual = residual
        self._logger = logger
        self._model, self._version = stamp
        self._seed = seed

    def compute(
        self, question: str, claims: list[Claim], export: ResponsivenessExport
    ) -> list[AtomRecord]:
        """Return one responsive atom per claim; publish each to ``export``."""
        atoms: list[AtomRecord] = []
        for claim in claims:
            on_topic = self._on_topic.check(
                subject=claim.id, role="on_topic", query=question, response=claim.text
            )
            on_ask = self._on_ask.judge(
                subject=claim.id,
                role="on_ask_claim",
                question=f"[[overlap:0.5]] {question}",
                context=claim.text,
                tier="T2",
            )
            responsive = on_topic.verdict and on_ask.verdict
            atom = self._logger.record(
                subject=claim.id,
                role="responsive",
                question="on_topic AND on_ask",
                tier="T2",
                verdict=responsive,
                evidence=f"on_topic={on_topic.verdict} on_ask={on_ask.verdict}",
                grader_id="relevance.responsive",
                model=self._model,
                model_version=self._version,
                seed=self._seed,
            )
            export.set(claim.id, responsive, atom.id)
            atoms.append(atom)
            if on_topic.verdict and not on_ask.verdict:  # thin residual (audit only)
                self._residual.judge(
                    subject=claim.id,
                    role="residual",
                    question="[[deny]] Does this on-topic claim answer a different sub-question?",
                    context=claim.text,
                )
        return atoms
