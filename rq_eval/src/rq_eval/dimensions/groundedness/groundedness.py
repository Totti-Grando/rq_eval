"""§1 groundedness — source faithfulness (build order E3).

Similarity pre-filter [T1] → three-way entailment per triplet [T2] →
``groundedness = |E-labeled| / |total triplets|`` [code]. Exports the per-claim
``grounded?`` (all its triplets E) for accuracy, and per-triplet confidences for
the conformal layer.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, DimensionResult, EvalInput, Triplet
from rq_eval.dimensions.base import Dimension
from rq_eval.dimensions.groundedness.export import GroundednessExport
from rq_eval.dimensions.groundedness.prefilter import SimilarityPreFilter
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.providers.model_stamp import ModelStamp
from rq_eval.scoring.bands import BandMapper
from rq_eval.scoring.formulas import default_registry
from rq_eval.scoring.wilson import WilsonInterval

if TYPE_CHECKING:
    from rq_eval.config import Config
    from rq_eval.providers.factory import Providers

_FORMULA = "mean"  # |E| / |total triplets|


class GroundednessDimension(Dimension):
    """§1 — per-triplet three-way entailment; exports per-claim grounded."""

    name = "groundedness"

    def __init__(
        self,
        providers: Providers,
        cfg: Config,
        logger: AtomLogger,
        triplets: list[Triplet],
        export: GroundednessExport,
    ) -> None:
        """Assemble pre-filter + grounding grader; store triplets + export."""
        self._logger = logger
        self._triplets = triplets
        self._export = export
        stamp = ModelStamp(cfg)
        self._prefilter = SimilarityPreFilter(providers.embedding)
        self._grounding = GroundingGrader(
            providers.grounding, logger, stamp.grounding(), "groundedness.triplet", cfg.seeds.judge
        )
        self._claim_stamp = stamp.grounding()
        self._seed = cfg.seeds.judge
        self._registry = default_registry()
        self._bands = BandMapper(cfg.thresholds.bands.G, cfg.thresholds.bands.A)

    def evaluate(self, eval_input: EvalInput) -> DimensionResult:
        """Entail each triplet against its nearest span; score |E|/|total|."""
        spans = [c.text for c in eval_input.context]
        triplet_atoms: list[AtomRecord] = []
        by_claim: dict[str, list[tuple[bool, float]]] = {}
        for triplet in self._triplets:
            premise = self._prefilter.select(triplet.text, spans)
            atom, res = self._grounding.assess(
                subject=triplet.id, role="triplet_grounded", premise=premise,
                hypothesis=triplet.text,
            )
            triplet_atoms.append(atom)
            self._export.add_triplet(atom.id, res.label)
            by_claim.setdefault(triplet.claim_id, []).append((res.supported, res.raw_score))

        self._export_per_claim(by_claim)
        score = self._registry.compute(_FORMULA, triplet_atoms)
        supported = sum(1 for a in triplet_atoms if a.verdict)
        n = len(triplet_atoms)
        low, high = WilsonInterval().interval(supported, n)
        return DimensionResult(
            dimension=self.name, score=score, band=self._bands.band(score),
            ci_low=low, ci_high=high, n=n,
            inputs_hash=hashlib.sha256(" ".join(spans).encode()).hexdigest()[:16],
            atom_ids=[a.id for a in triplet_atoms], formula_id=_FORMULA, abstained=(n == 0),
            extra={"supported": float(supported), "total": float(n)},
        )

    def _export_per_claim(self, by_claim: dict[str, list[tuple[bool, float]]]) -> None:
        """Log + publish the per-claim grounded atom (all its triplets E)."""
        for claim_id, items in by_claim.items():
            grounded = all(ok for ok, _ in items)
            atom = self._logger.record(
                subject=claim_id, role="grounded", question="all triplets entailed?", tier="T2",
                verdict=grounded, evidence=f"n_triplets={len(items)}",
                grader_id="groundedness.claim", model=self._claim_stamp[0],
                model_version=self._claim_stamp[1], seed=self._seed,
            )
            self._export.set(claim_id, atom, [conf for _, conf in items])
