"""G7 — claim-graph visualization: a view over logged nodes/edges/verdicts."""

from __future__ import annotations

import json
from pathlib import Path

from rq_eval.contracts import AtomRecord, Claim
from rq_eval.pipeline.claim_graph import ClaimGraph, GraphNode
from rq_eval.pipeline.graph_viz import GraphVisualizer


def _claim(cid: str) -> Claim:
    return Claim(id=cid, text=cid, source_sentence=cid, verifiable=True, decontextualized=True,
                 extractor_version="claim-extractor-v1")


def _graph() -> ClaimGraph:
    g = ClaimGraph()
    for cid in ("c1", "c2", "c3"):
        g.add_node(GraphNode(_claim(cid), "independent", cid, context_incomplete=False))
    g.add_edge("c1", "c2", "supports")   # c1 -> c2 (chain)
    g.add_edge("c2", "c3", "contradicts")  # a broken/contradiction step
    return g


def test_render_reads_logged_verdicts_and_marks_broken_step(tmp_path: Path) -> None:
    graph = _graph()
    atoms = [
        AtomRecord.create(subject="c1", role="axiom", question="q", tier="code", verdict=True),
        AtomRecord.create(subject="c2", role="axiom", question="q", tier="code", verdict=False),
        AtomRecord.create(subject="c2", role="derived", question="q", tier="code", verdict=True),
        AtomRecord.create(subject="c3", role="axiom", question="q", tier="code", verdict=False),
    ]
    viz = GraphVisualizer()
    status = viz.status_from_atoms(graph, atoms)
    out = viz.render(graph, status, tmp_path / "graph.json")
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in data["nodes"]}
    assert by_id["c1"]["color"] == "green"          # passing axiom
    assert by_id["c2"]["color"] == "green"          # rescued (derived)
    assert by_id["c3"]["status"] == "failed" and by_id["c3"]["color"] == "red"  # broken step
    # the contradiction edge renders red
    contradiction = next(link for link in data["links"] if link["etype"] == "contradicts")
    assert contradiction["color"] == "red"
