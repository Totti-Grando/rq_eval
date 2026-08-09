"""G6 — relevance Layer 2 reads the SHARED graph (tree off by default; adds no edges)."""

from __future__ import annotations

from pathlib import Path

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.config import Config, load_config
from rq_eval.contracts import Claim, EvalInput
from rq_eval.dimensions.relevance.relevance import RelevanceDimension
from rq_eval.dimensions.responsiveness import ResponsivenessExport
from rq_eval.pipeline.claim_graph import ClaimGraph, GraphNode
from rq_eval.providers.factory import ProviderFactory

_Q = "Who won the championship match?"
_CLAIMS = [
    Claim(id="c1", text="A striker was signed in July", source_sentence="A striker was signed",
          verifiable=True, decontextualized=True, extractor_version="claim-extractor-v1"),
    Claim(id="c2", text="The striker scored the goal", source_sentence="The striker scored",
          verifiable=True, decontextualized=True, extractor_version="claim-extractor-v1"),
    Claim(id="c3", text="Real Madrid won the championship match", source_sentence="Madrid won",
          verifiable=True, decontextualized=True, extractor_version="claim-extractor-v1"),
]


def _cfg(tree: bool) -> Config:
    cfg = load_config()
    return cfg.model_copy(
        update={"relevance": cfg.relevance.model_copy(update={"tree_enabled": tree})}
    )


def _graph() -> ClaimGraph:
    g = ClaimGraph()
    for c in _CLAIMS:
        g.add_node(GraphNode(c, "independent", c.text, context_incomplete=False))
    g.add_edge("c1", "c2", "supports")  # premise chain c1 -> c2 -> c3(anchor)
    g.add_edge("c2", "c3", "supports")
    return g


def _run(cfg: Config, graph: ClaimGraph | None, path: Path):
    store = JsonlAtomStore(path)
    dim = RelevanceDimension(
        ProviderFactory(cfg).build(), cfg, AtomLogger(store, FixedClock()), _CLAIMS,
        ResponsivenessExport(), graph=graph,
    )
    ei = EvalInput(question=_Q, answer="Real Madrid won the championship match.")
    return dim.evaluate(ei), store


def test_tree_off_is_the_direct_core(tmp_path: Path) -> None:
    """Default (tree off) scores the direct core via relevance_capped_mean."""
    result, _ = _run(_cfg(tree=False), _graph(), tmp_path / "off.jsonl")
    assert result.formula_id == "relevance_capped_mean"


def test_tree_on_reads_shared_graph_premise_chain(tmp_path: Path) -> None:
    """With the tree on, off-topic premises attach via the SHARED graph's edges."""
    result, store = _run(_cfg(tree=True), _graph(), tmp_path / "on.jsonl")
    assert result.formula_id == "relevance_tree_capped_mean"
    graded = {a.subject: a for a in store.all() if a.role == "claim_relevance"}
    # c1 is off-topic on its own but reachable from the c3 anchor -> kept relevant
    assert graded["c1"].verdict is True
    assert graded["c1"].weight < graded["c3"].weight  # depth-decayed vs the anchor


def test_relevance_adds_no_edges_to_the_graph(tmp_path: Path) -> None:
    """Relevance READS the shared graph and adds zero edges of its own."""
    graph = _graph()
    before = len(graph.edges())
    _run(_cfg(tree=True), graph, tmp_path / "e.jsonl")
    assert len(graph.edges()) == before  # relevance built no edges
