"""G3 — the shared ClaimGraph: claim typing + indexical binding (inert, no scoring)."""

from __future__ import annotations

from pathlib import Path

from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.config import load_config
from rq_eval.contracts import AtomRecord, Claim
from rq_eval.dimensions.groundedness.export import GroundednessExport
from rq_eval.fixtures import FixtureSuite
from rq_eval.graders.t1 import T1Tools
from rq_eval.pipeline.claim_graph import ClaimGraphBuilder
from rq_eval.providers.factory import ProviderFactory
from rq_eval.runner import Evaluator


def _claim(cid: str, text: str) -> Claim:
    return Claim(
        id=cid, text=text, source_sentence=text, verifiable=True, decontextualized=True,
        extractor_version="claim-extractor-v1",
    )


def _builder(grounded: GroundednessExport | None = None) -> ClaimGraphBuilder:
    providers = ProviderFactory(load_config()).build()
    return ClaimGraphBuilder(T1Tools(), providers.nlp, grounded or GroundednessExport())


def test_three_claim_types_are_tagged() -> None:
    # inference-dependent: well-formed but empty support set S
    grounded = GroundednessExport()
    grounded.add_triplet("t:c2", "N", "c2", set(), set())  # empty S
    grounded.set("c2", AtomRecord.create(subject="c2", role="grounded", question="g", tier="T2",
                                         verdict=False), [0.0])
    claims = [
        _claim("c1", "Real Madrid won the final."),   # independent (default)
        _claim("c2", "Margins therefore improved."),  # inference-dependent (empty S)
        _claim("c3", "It happened here."),             # indexical (deixis)
    ]
    graph = _builder(grounded).build(claims)
    types = {n.claim.id: n.ctype for n in graph.nodes()}
    assert types == {"c1": "independent", "c2": "inference", "c3": "indexical"}


def test_indexical_binds_to_sibling_filler() -> None:
    claims = [
        _claim("c1", "The match was played in London."),  # supplies the LOC filler
        _claim("c2", "It is dark now."),                   # indexical -> bind
    ]
    graph = _builder().build(claims)
    node = graph.node("c2")
    assert node.ctype == "indexical"
    assert node.context_incomplete is False
    assert "London" in node.bound_text  # slot filled from the sibling


def test_unbindable_indexical_is_flagged_context_incomplete() -> None:
    claims = [_claim("c1", "it is dark now")]  # no sibling supplies a filler
    node = _builder().build(claims).node("c1")
    assert node.ctype == "indexical"
    assert node.context_incomplete is True  # reported, not guessed


def test_graph_is_inert_in_a_run(tmp_path: Path) -> None:
    """The graph is built once and has no edges yet (G3); scores are unchanged."""
    cfg = load_config()
    case = next(c for c in FixtureSuite().cases() if c.name == "aligned")
    store = JsonlAtomStore(tmp_path / "a.jsonl")
    result = Evaluator(cfg, store=store, clock=FixedClock()).evaluate(case.to_input())
    assert result.graph is not None
    assert len(result.graph.nodes()) == len(result.claims)
    assert result.graph.edges() == []  # no edge detection until G4
