"""Attribution confidences export (§4 → conformal §5).

Per cited claim, the attribution confidence + atom id, so the conformal layer
(§5/E8) can calibrate a distribution-free guarantee over retained attributions.
"""

from __future__ import annotations


class AttributionExport:
    """Per-claim attribution confidence, written by §4, read by the conformal layer."""

    def __init__(self) -> None:
        """Start empty; source_attribution populates one entry per cited claim."""
        self._conf: dict[str, float] = {}

    def set(self, claim_id: str, confidence: float) -> None:
        """Publish the attribution confidence for ``claim_id``."""
        self._conf[claim_id] = confidence

    def confidence(self, claim_id: str) -> float:
        """Return the attribution confidence for ``claim_id``."""
        return self._conf[claim_id]

    def all_confidences(self) -> dict[str, float]:
        """Return every cited claim's attribution confidence."""
        return dict(self._conf)
