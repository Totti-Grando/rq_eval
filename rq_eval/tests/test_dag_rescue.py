"""G5 — accuracy Layer 2 DAG derivation-rescue (additive, flag-gated)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.config import Config, load_config
from rq_eval.contracts import Claim, ContextChunk, EvalInput
from rq_eval.dimensions.accuracy.accuracy import AccuracyDimension
from rq_eval.graders.t1 import T1Tools
from rq_eval.pipeline.claim_graph import ClaimGraph, GraphNode
from rq_eval.pipeline.edge_detection import EdgeDetector
from rq_eval.providers.factory import ProviderFactory

# c1, c2 are grounded axioms; c3 (profit) is not directly grounded but = c1 - c2
_CLAIMS = [
    Claim(id="c1", text="Revenue was 100 for the quarter", source_sentence="Revenue was 100",
          verifiable=True, decontextualized=True, extractor_version="claim-extractor-v1"),
    Claim(id="c2", text="Costs were 40 for the quarter", source_sentence="Costs were 40",
          verifiable=True, decontextualized=True, extractor_version="claim-extractor-v1"),
    Claim(id="c3", text="Profit was 60 for the quarter", source_sentence="Profit was 60",
          verifiable=True, decontextualized=True, extractor_version="claim-extractor-v1"),
]
_CTX = [ContextChunk(
    id="s1", text="Revenue was 100 for the quarter. Costs were 40 for the quarter.",
)]
_EI = EvalInput(question="q", answer="...", context=_CTX)


def _cfg(rescue: bool) -> Config:
    cfg = load_config()
    return cfg.model_copy(
        update={"accuracy": cfg.accuracy.model_copy(update={"dag_rescue_enabled": rescue})}
    )


def _graph_with_edges(cfg: Config) -> ClaimGraph:
    graph = ClaimGraph()
    for c in _CLAIMS:
        graph.add_node(GraphNode(c, "independent", c.text, context_incomplete=False))
    EdgeDetector(
        ProviderFactory(cfg).build().grounding, T1Tools(), cfg.graph.edge_tau,
        cfg.graph.topical_min, cfg.graph.numeric_tolerance,
    ).detect(_CLAIMS, graph)
    return graph


def _score(cfg: Config, graph: ClaimGraph | None, path: Path) -> float:
    store = JsonlAtomStore(path)
    dim = AccuracyDimension(
        ProviderFactory(cfg).build(), cfg, AtomLogger(store, FixedClock()), _CLAIMS, graph=graph
    )
    return dim.evaluate(_EI).score


def test_flag_off_is_layer1_bit_identical(tmp_path: Path) -> None:
    """With rescue off, passing a graph does not change the score (= Layer 1)."""
    cfg = _cfg(rescue=False)
    no_graph = _score(cfg, None, tmp_path / "a.jsonl")
    with_graph = _score(cfg, _graph_with_edges(cfg), tmp_path / "b.jsonl")
    assert no_graph == with_graph == pytest.approx(2 / 3)  # c3 bare -> only 2 of 3 axioms


def test_derivation_rescue_counts_valid_derivation(tmp_path: Path) -> None:
    """With rescue on, the validly-derived profit claim now counts successful."""
    cfg = _cfg(rescue=True)
    graph = _graph_with_edges(cfg)
    assert ("c1", "c3") in {(s, d) for s, d, _ in graph.edges()}  # arithmetic edge present
    score = _score(cfg, graph, tmp_path / "c.jsonl")
    assert score == pytest.approx(1.0)  # c3 rescued via c1 - c2 = 60


def test_rescue_needs_the_graph(tmp_path: Path) -> None:
    """Rescue on but no graph -> degrades gracefully to the Layer-1 floor."""
    assert _score(_cfg(rescue=True), None, tmp_path / "d.jsonl") == pytest.approx(2 / 3)
