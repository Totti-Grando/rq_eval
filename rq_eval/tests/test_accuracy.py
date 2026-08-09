"""G2 — accuracy Layer 1: per-node axiom-truth floor (RQ §1), offline/mock."""

from __future__ import annotations

from pathlib import Path

import pytest

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.audit.replay import ReplayVerifier
from rq_eval.config import load_config
from rq_eval.contracts import AtomRecord, Claim, ContextChunk, EvalInput
from rq_eval.dimensions.accuracy.accuracy import AccuracyDimension
from rq_eval.dimensions.groundedness.export import GroundednessExport
from rq_eval.providers.factory import ProviderFactory
from rq_eval.scoring.formulas import default_registry


def _claim(cid: str, text: str, citation: str | None = None) -> Claim:
    return Claim(
        id=cid, text=text, source_sentence=text, verifiable=True, decontextualized=True,
        extractor_version="claim-extractor-v1", citation=citation,
    )


def _dim(store_path: Path, claims, grounded: GroundednessExport | None = None):
    cfg = load_config()
    store = JsonlAtomStore(store_path)
    logger = AtomLogger(store, FixedClock())
    dim = AccuracyDimension(
        ProviderFactory(cfg).build(), cfg, logger, claims, grounded_export=grounded
    )
    return dim, store


def test_grounded_claim_scores_high(tmp_path: Path) -> None:
    claims = [_claim("c1", "Real Madrid won the Champions League final.")]
    dim, _store = _dim(tmp_path / "a.jsonl", claims)
    ctx = [ContextChunk(id="chunk-1", text="Real Madrid won the Champions League final in 2024.")]
    result = dim.evaluate(EvalInput(question="who won?", answer="...", context=ctx))
    assert result.dimension == "accuracy"
    assert result.score == pytest.approx(1.0)
    assert result.formula_id == "dag_resolution"


def test_ungrounded_claim_scores_zero(tmp_path: Path) -> None:
    claims = [_claim("c1", "Barcelona won the treble in 2015.")]
    dim, _store = _dim(tmp_path / "a.jsonl", claims)
    ctx = [ContextChunk(id="chunk-1", text="The weather in Madrid was sunny.")]
    result = dim.evaluate(EvalInput(question="q", answer="...", context=ctx))
    assert result.score == pytest.approx(0.0)


def test_accuracy_is_truth_only_no_responsive(tmp_path: Path) -> None:
    """A true, well-sourced, correctly-cited but OFF-TOPIC claim counts accurate.

    G2's responsive decoupling: accuracy is truth, not relevance. There is no
    `responsive` atom on accuracy's path (relevance owns responsiveness).
    """
    claims = [_claim("c1", "Real Madrid won the Champions League final.", "chunk-1")]
    ctx = [ContextChunk(id="chunk-1", text="Real Madrid won the Champions League final in 2024.")]
    grounded = GroundednessExport()
    grounded.add_triplet("t:c1", "E", "c1", {"chunk-1"}, {"chunk-1"})
    grounded.set("c1", AtomRecord.create(subject="c1", role="grounded", question="g", tier="T2",
                                         verdict=True), [1.0])
    dim, store = _dim(tmp_path / "a.jsonl", claims, grounded)
    # question is unrelated to the claim -> off-topic, but accuracy ignores that
    result = dim.evaluate(EvalInput(question="What is the capital of France?", answer="...",
                                    context=ctx))
    assert result.score == pytest.approx(1.0)  # true+sourced+cited -> accurate axiom
    assert not any(a.role == "responsive" for a in store.all())  # decoupled


def test_grounded_imported_from_groundedness_export(tmp_path: Path) -> None:
    """Flipping the imported grounded verdict changes accuracy (§1 wire)."""
    claims = [_claim("c1", "Real Madrid won the Champions League final.")]
    ctx = [ContextChunk(id="chunk-1", text="Real Madrid won the Champions League final in 2024.")]
    ei = EvalInput(question="q", answer="...", context=ctx)

    def _run(grounded: bool, path: Path) -> float:
        gexport = GroundednessExport()
        gexport.set(
            "c1",
            AtomRecord.create(subject="c1", role="grounded", question="grounded?", tier="T2",
                              verdict=grounded),
            [1.0 if grounded else 0.0],
        )
        dim, _ = _dim(path, claims, gexport)
        return dim.evaluate(ei).score

    assert _run(True, tmp_path / "t.jsonl") == pytest.approx(1.0)
    assert _run(False, tmp_path / "f.jsonl") == pytest.approx(0.0)


def test_numeric_mismatch_fails_claim(tmp_path: Path) -> None:
    claims = [_claim("c1", "Revenue was $1.2B.")]
    dim, _store = _dim(tmp_path / "a.jsonl", claims)
    ctx = [ContextChunk(id="chunk-1", text="Revenue was $1.3B for the quarter.")]
    result = dim.evaluate(EvalInput(question="q", answer="...", context=ctx))
    # right topic, wrong number -> numeric exact-match fails the claim
    assert result.score == pytest.approx(0.0)


def test_accuracy_replays(tmp_path: Path) -> None:
    claims = [_claim("c1", "Real Madrid won the final."), _claim("c2", "The final was in London.")]
    dim, store = _dim(tmp_path / "a.jsonl", claims)
    ctx = [ContextChunk(id="chunk-1", text="Real Madrid won the final. The final was in London.")]
    result = dim.evaluate(EvalInput(question="q", answer="...", context=ctx))
    assert ReplayVerifier(default_registry()).verify(result, store) is True
