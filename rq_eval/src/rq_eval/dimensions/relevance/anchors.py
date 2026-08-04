"""§3 relevance — anchor selection (on-ask seed + centrality + conformal band).

Anchors are the load-bearing roots of the support tree: the claims that address
the question head-on. They are seeded by the DIVER-QA on-ask check, then
**confirmed/expanded by graph centrality** — a claim many others support is
central even if its own question-match was borderline. Because a missed anchor
orphans its whole subtree, anchor recall is high-recall and **conformal-bounded**
(reusing the §5 machinery with ``relevance.anchor_alpha``): "we miss a true
anchor" becomes a *bounded probability*, reported as a band, not an unknown.
"""

from __future__ import annotations

from dataclasses import dataclass

from rq_eval.dimensions.relevance.edges import Edge
from rq_eval.scoring.conformal import ConformalCalibrator


@dataclass(frozen=True, slots=True)
class AnchorResult:
    """The anchor set plus provenance and the conformal recall band."""

    anchor_ids: set[str]
    seed_ids: set[str]
    centrality: dict[str, int]
    band_low: float
    band_high: float
    n: int
    abstained: bool


class AnchorSelector:
    """Seed anchors by on-ask, expand by centrality, wrap recall in conformal."""

    def __init__(self, calibrator: ConformalCalibrator, centrality_min: int) -> None:
        """Inject the conformal calibrator and the centrality promotion threshold."""
        self._calibrator = calibrator
        self._centrality_min = centrality_min

    def select(
        self,
        claim_ids: list[str],
        seed_ids: set[str],
        edges: list[Edge],
        confidences: dict[str, float],
    ) -> AnchorResult:
        """Return the anchor set: on-ask seeds ∪ high-centrality claims.

        Centrality is in-degree over confirmed support edges (how many distinct
        premises support a claim); a claim supported by ``≥ centrality_min``
        others is promoted even if its on-ask seed was borderline.
        """
        centrality: dict[str, int] = {cid: 0 for cid in claim_ids}
        incoming: dict[str, set[str]] = {cid: set() for cid in claim_ids}
        for e in edges:
            if e.dst in incoming and e.src != e.dst:
                incoming[e.dst].add(e.src)
        for cid, srcs in incoming.items():
            centrality[cid] = len(srcs)

        anchors = set(seed_ids)
        for cid in claim_ids:
            if centrality[cid] >= self._centrality_min:
                anchors.add(cid)

        band = self._calibrator.calibrate([confidences.get(a, 0.0) for a in sorted(anchors)])
        return AnchorResult(
            anchor_ids=anchors, seed_ids=set(seed_ids), centrality=centrality,
            band_low=band.band_low, band_high=band.band_high, n=band.n,
            abstained=band.abstained,
        )
