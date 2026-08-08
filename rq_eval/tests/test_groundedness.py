"""E3 — groundedness dimension (§1), offline/mock."""

from __future__ import annotations

from pathlib import Path

import pytest

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.audit.replay import ReplayVerifier
from rq_eval.config import load_config
from rq_eval.contracts import ContextChunk, EvalInput, Triplet
from rq_eval.dimensions.groundedness.export import GroundednessExport
from rq_eval.dimensions.groundedness.groundedness import GroundednessDimension
from rq_eval.dimensions.groundedness.prefilter import SimilarityPreFilter
from rq_eval.providers.factory import ProviderFactory
from rq_eval.scoring.formulas import default_registry


def _triplet(text: str, claim_id: str) -> Triplet:
    toks = text.split()
    return Triplet.create(
        claim_id=claim_id, subject=toks[0], predicate=toks[1] if len(toks) > 1 else "",
        obj=" ".join(toks[2:]), citation=None, source_pointer=text,
    )


def _dim(store_path: Path, triplets, export):
    cfg = load_config()
    store = JsonlAtomStore(store_path)
    dim = GroundednessDimension(
        ProviderFactory(cfg).build(), cfg, AtomLogger(store, FixedClock()), triplets, export
    )
    return dim, store


def test_grounded_triplets_score_high(tmp_path: Path) -> None:
    triplets = [_triplet("Real Madrid won final", "c1")]
    export = GroundednessExport()
    dim, _ = _dim(tmp_path / "a.jsonl", triplets, export)
    ctx = [ContextChunk(id="s1", text="Real Madrid won the final in 2024.")]
    result = dim.evaluate(EvalInput(question="q", answer="...", context=ctx))
    assert result.dimension == "groundedness"
    assert result.score == pytest.approx(1.0)
    assert export.has("c1") and export.grounded("c1") is True
    assert export.confidences("c1")  # per-triplet confidences recorded


def test_ungrounded_triplets_score_zero(tmp_path: Path) -> None:
    triplets = [_triplet("Barcelona won treble", "c1")]
    export = GroundednessExport()
    dim, _ = _dim(tmp_path / "a.jsonl", triplets, export)
    ctx = [ContextChunk(id="s1", text="The weather was sunny in Madrid.")]
    result = dim.evaluate(EvalInput(question="q", answer="...", context=ctx))
    assert result.score == pytest.approx(0.0)
    assert export.grounded("c1") is False


def test_score_replays(tmp_path: Path) -> None:
    triplets = [_triplet("Real Madrid won final", "c1"), _triplet("Barcelona lost match", "c2")]
    export = GroundednessExport()
    dim, store = _dim(tmp_path / "a.jsonl", triplets, export)
    ctx = [ContextChunk(id="s1", text="Real Madrid won the final. Barcelona lost the match.")]
    result = dim.evaluate(EvalInput(question="q", answer="...", context=ctx))
    assert ReplayVerifier(default_registry()).verify(result, store) is True


def test_prefilter_is_not_the_score(tmp_path: Path) -> None:
    # the pre-filter only focuses which chunks to entail; the score atoms are the entailments
    cfg = load_config()
    pf = SimilarityPreFilter(ProviderFactory(cfg).build().embedding)
    chunks = [
        ContextChunk(id="s1", text="Real Madrid won the final."),
        ContextChunk(id="s2", text="Bananas are yellow."),
    ]
    top = pf.select_k("Real Madrid won final", chunks, 1)
    assert [c.id for c in top] == ["s1"]
    spans = ["Real Madrid won the final.", "Bananas are yellow."]
    triplets = [_triplet("Real Madrid won final", "c1")]
    dim, store = _dim(tmp_path / "a.jsonl", triplets, GroundednessExport())
    result = dim.evaluate(EvalInput(
        question="q", answer="...", context=[ContextChunk(id="s1", text=spans[0])]
    ))
    # every score atom is a triplet entailment (role), none is a pre-filter atom
    score_atoms = [a for a in store.all() if a.id in set(result.atom_ids)]
    assert all(a.role == "triplet_grounded" for a in score_atoms)
