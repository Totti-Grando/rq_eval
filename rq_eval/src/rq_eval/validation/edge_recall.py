"""§0.3 — the edge-detection recall/precision harness (the honest error bar).

Edge recall is the weakest link in the claim graph: an unmarked or convergent
multi-premise edge that is missed makes a valid derived claim look like a failed
orphan. So edge detection is **measured against a human-linked sample and
reported** — this is the number that gates whether the Layer-2 flags (accuracy
DAG-rescue, relevance tree) are trusted in production. Pure code over a labelled
fixture set; no scoring side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

from rq_eval.contracts import Claim
from rq_eval.pipeline.claim_graph import ClaimGraph, GraphNode
from rq_eval.pipeline.edge_detection import EdgeDetector


@dataclass(frozen=True, slots=True)
class EdgeRecallReport:
    """Detection recall/precision against the human-linked gold edges."""

    recall: float
    precision: float
    true_positives: int
    detected: int
    gold: int


@dataclass(frozen=True, slots=True)
class EdgeCase:
    """One labelled case: the claims and the gold ``(parent_id, child_id)`` edges."""

    claims: list[Claim]
    gold_edges: set[tuple[str, str]]


class EdgeRecallHarness:
    """Runs the detector over labelled cases and reports recall/precision."""

    def __init__(self, detector: EdgeDetector) -> None:
        """Inject the edge detector under test."""
        self._detector = detector

    def measure(self, cases: list[EdgeCase]) -> EdgeRecallReport:
        """Aggregate detection recall + precision across ``cases``."""
        tp = detected = gold = 0
        for case in cases:
            found = self._detect(case.claims)
            tp += len(found & case.gold_edges)
            detected += len(found)
            gold += len(case.gold_edges)
        recall = tp / gold if gold else 1.0
        precision = tp / detected if detected else 1.0
        return EdgeRecallReport(recall, precision, tp, detected, gold)

    def _detect(self, claims: list[Claim]) -> set[tuple[str, str]]:
        graph = ClaimGraph()
        for claim in claims:  # bare nodes are enough for edge detection
            graph.add_node(GraphNode(claim, "independent", claim.text, context_incomplete=False))
        self._detector.detect(claims, graph)
        return {(src, dst) for src, dst, _ in graph.edges()}
