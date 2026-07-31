"""§4 step 1 — map three-way entailment to attribution labels.

AttrScore 3-way (Attributable / Contradictory / Extrapolatory) by default; CAQA
4-way (Supported / Insufficient / Contradictory / Irrelevant) when configured.
``Attributable`` (== Supported) is the label that counts as attributed.
"""

from __future__ import annotations

from rq_eval.providers.base import EntailmentResult

_IRRELEVANT_TAU = 0.2  # 4-way: a Neutral with very low overlap is "Irrelevant"


class AttributionLabeler:
    """Maps an :class:`EntailmentResult` to the configured attribution label."""

    def __init__(self, mode: str) -> None:
        """Store the label scheme ('three' or 'four')."""
        self._mode = mode

    def label(self, res: EntailmentResult) -> str:
        """Return the attribution label for a cited-chunk↔claim entailment."""
        if self._mode == "four":
            if res.label == "E":
                return "Supported"
            if res.label == "C":
                return "Contradictory"
            return "Irrelevant" if res.raw_score < _IRRELEVANT_TAU else "Insufficient"
        # three-way (AttrScore)
        if res.label == "E":
            return "Attributable"
        return "Contradictory" if res.label == "C" else "Extrapolatory"

    @staticmethod
    def is_attributable(res: EntailmentResult) -> bool:
        """True iff the cited chunk entails the claim (Attributable / Supported)."""
        return res.label == "E"
