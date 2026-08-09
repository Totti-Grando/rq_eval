"""G4 — edge detection (backward premise-BFS) + the recall harness."""

from __future__ import annotations

from rq_eval.config import load_config
from rq_eval.contracts import Claim
from rq_eval.graders.t1 import T1Tools
from rq_eval.pipeline.claim_graph import ClaimGraph, GraphNode
from rq_eval.pipeline.edge_detection import EdgeDetector
from rq_eval.providers.factory import ProviderFactory
from rq_eval.validation.edge_recall import EdgeCase, EdgeRecallHarness


def _claim(cid: str, text: str) -> Claim:
    return Claim(
        id=cid, text=text, source_sentence=text, verifiable=True, decontextualized=True,
        extractor_version="claim-extractor-v1",
    )


def _detector(edge_tau: float = 0.5, topical_min: float = 0.3) -> EdgeDetector:
    grounding = ProviderFactory(load_config()).build().grounding
    return EdgeDetector(grounding, T1Tools(), edge_tau, topical_min, numeric_tolerance=0.0)


def _graph(claims: list[Claim]) -> ClaimGraph:
    g = ClaimGraph()
    for c in claims:
        g.add_node(GraphNode(c, "independent", c.text, context_incomplete=False))
    return g


def _edges(claims: list[Claim], **kw: float) -> set[tuple[str, str]]:
    g = _graph(claims)
    _detector(**kw).detect(claims, g)
    return {(s, d) for s, d, _ in g.edges()}


def test_single_parent_edge_detected() -> None:
    claims = [
        _claim("c1", "Revenue rose ten percent in the quarter"),
        _claim("c2", "Revenue rose in the quarter"),  # entailed by c1 (subset of terms)
    ]
    assert ("c1", "c2") in _edges(claims)


def test_arithmetic_convergent_edge_detected() -> None:
    """A convergent edge C1 ∧ C2 → C3 found via the numeric (sum) signature."""
    claims = [
        _claim("c1", "Revenue was 100 for the quarter"),
        _claim("c2", "Costs were 40 for the quarter"),
        _claim("c3", "Profit was 60 for the quarter"),  # 100 - 40 = 60
    ]
    edges = _edges(claims)
    assert ("c1", "c3") in edges and ("c2", "c3") in edges  # both parents identified


def test_edges_are_acyclic_earlier_to_later() -> None:
    claims = [
        _claim("c1", "GDP fell three percent"),
        _claim("c2", "GDP fell three percent sharply"),
    ]
    order = {"c1": 0, "c2": 1}
    for src, dst in _edges(claims):
        assert order[src] < order[dst]  # every edge points earlier -> later (DAG)


def test_off_topic_pair_has_no_edge() -> None:
    claims = [
        _claim("c1", "Bananas are rich in potassium"),
        _claim("c2", "Real Madrid won the final"),  # unrelated -> no shared terms -> no edge
    ]
    assert _edges(claims) == set()


def test_recall_harness_reports_a_number() -> None:
    claims = [
        _claim("c1", "Revenue was 100 for the quarter"),
        _claim("c2", "Costs were 40 for the quarter"),
        _claim("c3", "Profit was 60 for the quarter"),
    ]
    gold = {("c1", "c3"), ("c2", "c3")}
    report = EdgeRecallHarness(_detector()).measure([EdgeCase(claims, gold)])
    assert report.gold == 2
    assert 0.0 <= report.recall <= 1.0 and 0.0 <= report.precision <= 1.0
    assert report.recall == 1.0  # the convergent arithmetic edge is found
