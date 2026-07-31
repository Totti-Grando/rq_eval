"""§1 accuracy — derived from the cached claims (build order B7).

Not an independent scorer: four booleans per claim, then compose in code
``accuracy = Σ correct·w / Σ w`` where ``correct = grounded ∧ source_adequate ∧
attributed ∧ responsive``. ``responsive`` is imported from relevance (§3).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, Claim, DimensionResult, EvalInput
from rq_eval.dimensions.accuracy.claim_accuracy import ClaimAccuracy, ClaimAccuracyDeps
from rq_eval.dimensions.accuracy.importance import ImportanceWeights
from rq_eval.dimensions.accuracy.stubs import InferenceValidityStub
from rq_eval.dimensions.base import Dimension
from rq_eval.dimensions.groundedness.export import GroundednessExport
from rq_eval.dimensions.responsiveness import ResponsivenessExport
from rq_eval.dimensions.source_quality.provider import SourceQualityProviderImpl
from rq_eval.dimensions.source_quality.reliability_list import ReliabilityList
from rq_eval.dimensions.source_quality.scorer import SourceQualityScorer
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.graders.judge_grader import JudgeGrader
from rq_eval.graders.t1 import T1Tools
from rq_eval.providers.model_stamp import ModelStamp
from rq_eval.scoring.bands import BandMapper
from rq_eval.scoring.formulas import default_registry
from rq_eval.scoring.wilson import WilsonInterval

if TYPE_CHECKING:
    from rq_eval.config import Config
    from rq_eval.providers.factory import Providers

_FORMULA = "conjunction_weighted_mean"


class AccuracyDimension(Dimension):
    """§1 — composes four per-claim booleans into a weighted accuracy score."""

    def __init__(
        self,
        providers: Providers,
        cfg: Config,
        logger: AtomLogger,
        claims: list[Claim],
        export: ResponsivenessExport,
        weights: ImportanceWeights | None = None,
        grounded_export: GroundednessExport | None = None,
    ) -> None:
        """Assemble graders/stubs from injected providers + config."""
        self._cfg = cfg
        self._claims = claims
        self._export = export
        stamp = ModelStamp(cfg)
        seed = cfg.seeds.judge
        grounding = GroundingGrader(
            providers.grounding, logger, stamp.grounding(), "accuracy.grounded", seed
        )
        attribution = GroundingGrader(
            providers.grounding, logger, stamp.grounding(), "accuracy.attributed", seed
        )
        residual = JudgeGrader(providers.judge, logger, stamp.judge(), "accuracy.residual", seed)
        weights = weights or ImportanceWeights(cfg.accuracy.importance_weighting)
        sq_supports = GroundingGrader(
            providers.grounding, logger, stamp.grounding(), "accuracy.sq_supports", seed
        )
        sq_judge = JudgeGrader(
            providers.judge, logger, stamp.judge(), "accuracy.sq_disinterest", seed
        )
        sq_scorer = SourceQualityScorer(
            cfg, logger, sq_supports, sq_judge, ReliabilityList(cfg), providers.resolver.resolve
        )
        self._claim_accuracy = ClaimAccuracy(
            ClaimAccuracyDeps(
                grounding=grounding, attribution=attribution, residual_truth=residual,
                t1=T1Tools(), source_quality=SourceQualityProviderImpl(cfg, sq_scorer),
                inference=InferenceValidityStub(), weights=weights, logger=logger,
                grounding_tau=cfg.thresholds.grounding_tau,
                numeric_tolerance=cfg.accuracy.numeric_tolerance,
                source_adequate_default=cfg.source_quality.source_adequate_default,
                grounded_export=grounded_export,
            )
        )
        self._registry = default_registry()
        self._bands = BandMapper(cfg.thresholds.bands.G, cfg.thresholds.bands.A)

    def evaluate(self, eval_input: EvalInput) -> DimensionResult:
        """Compute accuracy = Σ correct·w / Σ w over the claims; log atoms."""
        cited = {c.id: c.text for c in eval_input.context}
        atoms: list[AtomRecord] = []
        for claim in self._claims:
            atoms.extend(
                self._claim_accuracy.evaluate_claim(claim, eval_input.context, cited, self._export)
            )
        score = self._registry.compute(_FORMULA, atoms)
        n = len(self._claims)
        correct = self._correct_claims(atoms)
        low, high = WilsonInterval().interval(correct, n)
        return DimensionResult(
            dimension=self.name, score=score, band=self._bands.band(score),
            ci_low=low, ci_high=high, n=n,
            inputs_hash=self._hash(eval_input.answer, self._claims),
            atom_ids=[a.id for a in atoms], formula_id=_FORMULA, abstained=(n == 0),
            extra={"correct_claims": float(correct)},
        )

    name = "accuracy"

    @staticmethod
    def _correct_claims(atoms: list[AtomRecord]) -> int:
        by_subject: dict[str, bool] = {}
        for a in atoms:
            by_subject[a.subject] = by_subject.get(a.subject, True) and a.verdict
        return sum(1 for ok in by_subject.values() if ok)

    @staticmethod
    def _hash(answer: str, claims: list[Claim]) -> str:
        joined = answer + "||" + "|".join(c.id for c in claims)
        return hashlib.sha256(joined.encode()).hexdigest()[:16]
