"""§4/§6 — AttributionProvider implementation (accuracy imports this).

A **set operation over the §1 support set ``S``**, not a second NLI pass:
``attributed ⟺ (C ∩ S ≠ ∅) ∧ confidence ≥ threshold``, where ``C`` is the claim's
cited set and the confidence is the claim's groundedness support strength. Because
``C ⊆ retrieved`` and ``S`` is over retrieved chunks, ``attributed ⟹ S ≠ ∅``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rq_eval.dimensions.groundedness.export import GroundednessExport
from rq_eval.providers.base import AttributionProvider, AttributionResult

if TYPE_CHECKING:
    from rq_eval.config import Config


class AttributionProviderImpl(AttributionProvider):
    """attributed = (C ∩ S ≠ ∅) ∧ confidence ≥ threshold (set-op over §1's S)."""

    def __init__(
        self,
        cfg: Config,
        grounded_export: GroundednessExport,
        conformal_threshold: float | None = None,
    ) -> None:
        """Inject config + the §1 support set; optionally a conformal confidence gate.

        When ``conformal_threshold`` is given it replaces the precision threshold,
        realizing ``attributed? = (C∩S≠∅) ∧ conformal-confident`` (§4.5/§5).
        """
        self._grounded = grounded_export
        self._threshold = (
            conformal_threshold
            if conformal_threshold is not None
            else cfg.source_attribution.precision_threshold
        )

    def attributed(self, claim_id: str, cited: set[str]) -> AttributionResult:
        """Intersect the cited set ``C`` with the support set ``S``; apply the gate."""
        support = self._grounded.claim_support_chunks(claim_id)
        overlap = cited & support
        confs = self._grounded.confidences(claim_id) if self._grounded.has(claim_id) else []
        confidence = max(confs) if confs else 0.0
        attributable = bool(overlap)
        attributed = attributable and confidence >= self._threshold
        if not cited:
            label = "uncited"
        elif attributable:
            label = "attributed"
        else:
            label = "mis-cited"  # cited but no cited source is in S (C\S)
        return AttributionResult(attributed=attributed, confidence=confidence, label=label)
