"""§1 accuracy — DAG resolution over the cached claims (RQ §1).

Not an independent scorer: **`accuracy = successful nodes / total nodes`**, equal
weight, counted **per node**. Layer 1 (the protected floor, built here) scores
every claim as an independent **axiom** — `grounded ∧ source-adequate ∧
attributed` (truth-only; responsiveness is relevance's job, not accuracy's). Layer
2 (DAG derivation-rescue, G5) is additive and flag-gated. The axiom-to-derived
ratio is reported as a diagnostic.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, Claim, DimensionResult, EvalInput
from rq_eval.dimensions.accuracy.claim_accuracy import ClaimAccuracy, ClaimAccuracyDeps
from rq_eval.dimensions.base import Dimension
from rq_eval.dimensions.groundedness.export import GroundednessExport
from rq_eval.dimensions.source_attribution.provider import AttributionProviderImpl
from rq_eval.dimensions.source_quality.coi import CoiRule
from rq_eval.dimensions.source_quality.provider import SourceQualityProviderImpl
from rq_eval.dimensions.source_quality.reliability_list import ReliabilityList
from rq_eval.dimensions.source_quality.scorer import SourceQualityScorer
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.graders.judge_grader import JudgeGrader
from rq_eval.graders.t1 import T1Tools
from rq_eval.pipeline.claim_graph import ClaimGraph
from rq_eval.providers.model_stamp import ModelStamp
from rq_eval.scoring.bands import BandMapper
from rq_eval.scoring.formulas import default_registry
from rq_eval.scoring.wilson import WilsonInterval

if TYPE_CHECKING:
    from rq_eval.config import Config
    from rq_eval.providers.factory import Providers

_FORMULA = "dag_resolution"  # successful nodes / total nodes (Layer 2 off = axiom floor)


class AccuracyDimension(Dimension):
    """§1 — per-node axiom-truth resolution (Layer 1 floor; Layer 2 additive)."""

    name = "accuracy"

    def __init__(
        self,
        providers: Providers,
        cfg: Config,
        logger: AtomLogger,
        claims: list[Claim],
        grounded_export: GroundednessExport | None = None,
        attribution_conformal_threshold: float | None = None,
        graph: ClaimGraph | None = None,
    ) -> None:
        """Assemble the truth-axiom graders/providers from injected providers + config."""
        self._cfg = cfg
        self._claims = claims
        self._logger = logger
        # Layer 2 (DAG derivation-rescue) reads the shared graph only when enabled
        self._dag_rescue = cfg.accuracy.dag_rescue_enabled
        self._graph = graph
        stamp = ModelStamp(cfg)
        seed = cfg.seeds.judge
        grounding = GroundingGrader(
            providers.grounding, logger, stamp.grounding(), "accuracy.grounded", seed
        )
        attribution = AttributionProviderImpl(
            cfg,
            grounded_export or GroundednessExport(),
            conformal_threshold=attribution_conformal_threshold,
        )
        residual = JudgeGrader(providers.judge, logger, stamp.judge(), "accuracy.residual", seed)
        sq_judge = JudgeGrader(
            providers.judge, logger, stamp.judge(), "accuracy.sq_disinterest", seed
        )
        sq_scorer = SourceQualityScorer(
            cfg, logger, grounded_export or GroundednessExport(), sq_judge, ReliabilityList(cfg),
            CoiRule(cfg), providers.resolver.resolve,
        )
        self._claim_accuracy = ClaimAccuracy(
            ClaimAccuracyDeps(
                grounding=grounding, attribution=attribution, residual_truth=residual,
                t1=T1Tools(), source_quality=SourceQualityProviderImpl(cfg, sq_scorer),
                logger=logger, numeric_tolerance=cfg.accuracy.numeric_tolerance,
                source_adequate_default=cfg.source_quality.source_adequate_default,
                grounded_export=grounded_export,
            )
        )
        self._registry = default_registry()
        self._bands = BandMapper(cfg.thresholds.bands.G, cfg.thresholds.bands.A)

    def evaluate(self, eval_input: EvalInput) -> DimensionResult:
        """Score accuracy = successful / total per node (Layer 1 + optional Layer 2)."""
        cited = {c.id: c.text for c in eval_input.context}
        atoms: list[AtomRecord] = []
        for claim in self._claims:
            atoms.extend(self._claim_accuracy.evaluate_claim(claim, eval_input.context, cited))
        axiom_pass = {a.subject: a.verdict for a in atoms if a.role == "axiom"}
        rescued = self._rescue(axiom_pass)
        atoms.extend(rescued)
        score = self._registry.compute(_FORMULA, atoms)
        n = len(self._claims)
        axioms = sum(1 for ok in axiom_pass.values() if ok)
        successful = axioms + len(rescued)
        low, high = WilsonInterval().interval(successful, n)
        return DimensionResult(
            dimension=self.name, score=score, band=self._bands.band(score),
            ci_low=low, ci_high=high, n=n,
            inputs_hash=self._hash(eval_input.answer, self._claims),
            atom_ids=[a.id for a in atoms], formula_id=_FORMULA, abstained=(n == 0),
            extra={
                "successful_claims": float(successful),
                # evidential breadth: how many successes are direct axioms vs derived
                "axiom_derived_ratio": (axioms / successful) if successful else 0.0,
            },
        )

    def _rescue(self, axiom_pass: dict[str, bool]) -> list[AtomRecord]:
        """[Layer 2, flagged] Rescue bare-failed claims that resolve to true axioms.

        Roots-first (claims are in answer order; edges point earlier → later), a
        node is *true* if it is a passing axiom or a **dependent** whose confirmed
        parents all resolve true (local validity is given by the confirmed edge —
        a valid step on a false premise stays valid-but-false, localized to the
        false parent). Emits a ``derived`` atom for each rescued node.
        """
        if not self._dag_rescue or self._graph is None:
            return []
        resolved: dict[str, bool] = {}
        rescued: list[AtomRecord] = []
        for claim in self._claims:
            if axiom_pass.get(claim.id, False):
                resolved[claim.id] = True
                continue
            parents = self._graph.parents(claim.id)
            if parents and all(resolved.get(p, False) for p in parents):
                resolved[claim.id] = True
                rescued.append(
                    self._logger.record(
                        subject=claim.id, role="derived",
                        question="sub-DAG resolves to true axioms via valid steps?",
                        tier="code", verdict=True,
                        evidence=f"parents={sorted(parents)} all-true",
                        grader_id="accuracy.derived", model="code", model_version="rq_eval",
                    )
                )
            else:
                resolved[claim.id] = False
        return rescued

    @staticmethod
    def _hash(answer: str, claims: list[Claim]) -> str:
        joined = answer + "||" + "|".join(c.id for c in claims)
        return hashlib.sha256(joined.encode()).hexdigest()[:16]
