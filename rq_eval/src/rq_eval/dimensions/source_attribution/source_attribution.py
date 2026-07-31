"""§4 source_attribution — ALCE citation recall/precision (build order E6).

Per claim carrying a citation, a three-way (4-way configurable) verdict of the
CITED chunk vs the claim; ALCE recall + precision computed in code. Score =
citation precision (recall reported alongside). Claims with no citation are
excluded here (they route to accuracy's unsourced residual — counted once).
Provides the ``AttributionProvider`` accuracy imports as ``attributed?``.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, Claim, DimensionResult, EvalInput
from rq_eval.dimensions.base import Dimension
from rq_eval.dimensions.source_attribution.alce import AlceScorer
from rq_eval.dimensions.source_attribution.export import AttributionExport
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.providers.model_stamp import ModelStamp
from rq_eval.scoring.bands import BandMapper
from rq_eval.scoring.formulas import default_registry
from rq_eval.scoring.wilson import WilsonInterval

if TYPE_CHECKING:
    from rq_eval.config import Config
    from rq_eval.providers.factory import Providers

_FORMULA = "mean"  # citation precision = |attributable| / |citations|


class SourceAttributionDimension(Dimension):
    """§4 — cited-chunk↔claim attribution; ALCE precision (recall reported)."""

    name = "source_attribution"

    def __init__(
        self,
        providers: Providers,
        cfg: Config,
        logger: AtomLogger,
        claims: list[Claim],
        export: AttributionExport,
    ) -> None:
        """Assemble the (cited-chunk) grounding grader; store claims + export."""
        self._logger = logger
        self._claims = claims
        self._export = export
        stamp = ModelStamp(cfg)
        self._grounding = GroundingGrader(
            providers.grounding, logger, stamp.grounding(), "source_attribution.cite",
            cfg.seeds.judge,
        )
        self._alce = AlceScorer()
        self._registry = default_registry()
        self._bands = BandMapper(cfg.thresholds.bands.G, cfg.thresholds.bands.A)

    def evaluate(self, eval_input: EvalInput) -> DimensionResult:
        """Verify each cited claim against its cited chunk; score ALCE precision."""
        cited_map = {c.id: c for c in eval_input.context}
        cited_claims = [c for c in self._claims if c.citation and c.citation in cited_map]
        atoms: list[AtomRecord] = []
        attributable: list[bool] = []
        for claim in cited_claims:
            atom, res = self._grounding.assess(
                subject=f"cite:{claim.id}", role="citation_attributable",
                premise=cited_map[claim.citation].text, hypothesis=claim.text,  # type: ignore[index]
            )
            atoms.append(atom)
            attributable.append(res.supported)
            self._export.set(claim.id, res.raw_score)

        precision = self._registry.compute(_FORMULA, atoms)  # |attributable| / |citations|
        recall = self._alce.recall(attributable)
        supported = sum(1 for a in attributable if a)
        low, high = WilsonInterval().interval(supported, len(cited_claims))
        return DimensionResult(
            dimension=self.name, score=precision, band=self._bands.band(precision),
            ci_low=low, ci_high=high, n=len(cited_claims),
            inputs_hash=hashlib.sha256(eval_input.answer.encode()).hexdigest()[:16],
            atom_ids=[a.id for a in atoms], formula_id=_FORMULA,
            abstained=(len(cited_claims) == 0),
            extra={
                "citation_precision": precision,
                "citation_recall": recall,
                "cited_claims": float(len(cited_claims)),
                "excluded_no_citation": float(len(self._claims) - len(cited_claims)),
            },
        )
