"""§4/§6 — AttributionProvider implementation (accuracy imports this)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rq_eval.dimensions.source_attribution.labels import AttributionLabeler
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.providers.base import AttributionProvider, AttributionResult

if TYPE_CHECKING:
    from rq_eval.config import Config


class AttributionProviderImpl(AttributionProvider):
    """attributed = Attributable ∧ confidence ≥ precision_threshold (§4.3/§4.5)."""

    def __init__(
        self, cfg: Config, grounding: GroundingGrader, conformal_threshold: float | None = None
    ) -> None:
        """Inject config + grounding grader; optionally a conformal confidence gate.

        When ``conformal_threshold`` is given it replaces the precision threshold,
        realizing ``attributed? = Attributable ∧ conformal-confident`` (§4.5/§5).
        """
        self._grounding = grounding
        self._threshold = (
            conformal_threshold
            if conformal_threshold is not None
            else cfg.source_attribution.precision_threshold
        )
        self._labeler = AttributionLabeler(cfg.source_attribution.labels)

    def attributed(self, claim: str, cited_chunk: str) -> AttributionResult:
        """Entail the CITED chunk against the claim; apply the confidence gate."""
        res = self._grounding.classify(premise=cited_chunk, hypothesis=claim)
        attributable = self._labeler.is_attributable(res)
        attributed = attributable and res.raw_score >= self._threshold
        return AttributionResult(
            attributed=attributed, confidence=res.raw_score, label=self._labeler.label(res)
        )
