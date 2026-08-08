"""§4 source_attribution — ALCE citation precision via set-ops over ``S``.

Attribution is **not a second NLI pass** — it is a set operation over §1's
already-computed support set ``S``. Per claim, resolve the cited set ``C``
(explicit regex + implicit scope, confirmed in ``S``); ``attributed ⟺ C∩S≠∅``.
Score = citation precision (recall + the ``explicit``/``implicit-confirmed`` split
reported); diagnostics ``C−S`` (mis-citation) and ``S−C`` (uncited-supported)
fall out of the same sets. Only source-referencing claims (the **axiom subset**)
are scored; non-source-referencing claims are N/A and route out.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, Claim, DimensionResult, EvalInput
from rq_eval.dimensions.base import Dimension
from rq_eval.dimensions.groundedness.export import GroundednessExport
from rq_eval.dimensions.source_attribution.alce import AlceScorer
from rq_eval.dimensions.source_attribution.citations import ScopePropagator
from rq_eval.dimensions.source_attribution.export import AttributionExport
from rq_eval.scoring.bands import BandMapper
from rq_eval.scoring.formulas import default_registry
from rq_eval.scoring.wilson import WilsonInterval

if TYPE_CHECKING:
    from rq_eval.config import Config
    from rq_eval.providers.factory import Providers

_FORMULA = "mean"  # citation precision = |attributed| / |cited claims|
_CODE = ("code", "rq_eval")


class SourceAttributionDimension(Dimension):
    """§4 — cited-set ∩ support-set attribution; ALCE precision (recall reported)."""

    name = "source_attribution"

    def __init__(
        self,
        providers: Providers,
        cfg: Config,
        logger: AtomLogger,
        claims: list[Claim],
        export: AttributionExport,
        grounded_export: GroundednessExport | None = None,
    ) -> None:
        """Store claims + the attribution export + the §1 support set."""
        self._logger = logger
        self._claims = claims
        self._export = export
        self._grounded = grounded_export or GroundednessExport()
        self._alce = AlceScorer()
        self._registry = default_registry()
        self._bands = BandMapper(cfg.thresholds.bands.G, cfg.thresholds.bands.A)

    def evaluate(self, eval_input: EvalInput) -> DimensionResult:
        """Resolve C per claim, intersect with S; score ALCE precision."""
        retrieved = {c.id for c in eval_input.context}
        scope = ScopePropagator(retrieved)
        atoms: list[AtomRecord] = []
        attributable: list[bool] = []
        mis_cited = 0  # |C\S| claims
        uncited_supported = 0  # |S\C| claims
        for claim in self._claims:
            support = self._grounded.claim_support_chunks(claim.id)
            cited, tag = scope.cited_for(claim.source_sentence, support)
            if claim.citation and claim.citation in retrieved:  # pipeline-resolved explicit cite
                cited = cited | {claim.citation}
                tag = "explicit" if tag == "none" else tag
            if not cited:  # non-source-referencing -> N/A, route out (no double-count)
                if support:
                    uncited_supported += 1
                continue
            overlap = cited & support
            attributed = bool(overlap)
            if not attributed:
                mis_cited += 1
            atoms.append(self._log(claim.id, attributed, tag, cited, support))
            attributable.append(attributed)
            self._export.set(claim.id, self._confidence(claim.id))

        precision = self._registry.compute(_FORMULA, atoms)
        recall = self._alce.recall(attributable)
        supported = sum(1 for a in attributable if a)
        low, high = WilsonInterval().interval(supported, len(atoms))
        return DimensionResult(
            dimension=self.name, score=precision, band=self._bands.band(precision),
            ci_low=low, ci_high=high, n=len(atoms),
            inputs_hash=hashlib.sha256(eval_input.answer.encode()).hexdigest()[:16],
            atom_ids=[a.id for a in atoms], formula_id=_FORMULA, abstained=(len(atoms) == 0),
            extra={
                "citation_precision": precision,
                "citation_recall": recall,
                "cited_claims": float(len(atoms)),
                "mis_cited": float(mis_cited),  # |C\S|
                "uncited_supported": float(uncited_supported),  # |S\C|
                "excluded_no_citation": float(len(self._claims) - len(atoms)),
            },
        )

    def _confidence(self, claim_id: str) -> float:
        confs = self._grounded.confidences(claim_id) if self._grounded.has(claim_id) else []
        return max(confs) if confs else 0.0

    def _log(
        self, claim_id: str, attributed: bool, tag: str, cited: set[str], support: set[str]
    ) -> AtomRecord:
        return self._logger.record(
            subject=f"cite:{claim_id}", role="citation_attributable",
            question="cited set intersects support set (C∩S≠∅)?", tier="T1", verdict=attributed,
            evidence=f"tag={tag} C={sorted(cited)} S∩C={sorted(cited & support)}",
            grader_id="source_attribution.cite", model=_CODE[0], model_version=_CODE[1],
        )
