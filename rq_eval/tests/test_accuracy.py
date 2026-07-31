"""B7 — accuracy dimension (§1), offline/mock."""

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
from rq_eval.dimensions.accuracy.importance import ImportanceWeights
from rq_eval.dimensions.responsiveness import ResponsivenessExport
from rq_eval.providers.factory import ProviderFactory
from rq_eval.scoring.formulas import default_registry


def _claim(cid: str, text: str, citation: str | None = None) -> Claim:
    return Claim(
        id=cid, text=text, source_sentence=text, verifiable=True, decontextualized=True,
        extractor_version="claim-extractor-v1", citation=citation,
    )


def _export(claims: list[Claim], responsive: bool) -> ResponsivenessExport:
    export = ResponsivenessExport()
    for c in claims:
        export.set(
            c.id,
            AtomRecord.create(
                subject=c.id, role="responsive", question="on_topic AND on_ask",
                tier="T2", verdict=responsive,
            ),
        )
    return export


def _dim(store_path: Path, claims, export, weights=None):
    cfg = load_config()
    store = JsonlAtomStore(store_path)
    logger = AtomLogger(store, FixedClock())
    dim = AccuracyDimension(ProviderFactory(cfg).build(), cfg, logger, claims, export, weights)
    return dim, store


def test_grounded_claim_scores_high(tmp_path: Path) -> None:
    claims = [_claim("c1", "Real Madrid won the Champions League final.")]
    export = _export(claims, responsive=True)
    dim, _store = _dim(tmp_path / "a.jsonl", claims, export)
    ctx = [ContextChunk(id="chunk-1", text="Real Madrid won the Champions League final in 2024.")]
    result = dim.evaluate(EvalInput(question="who won?", answer="...", context=ctx))
    assert result.dimension == "accuracy"
    assert result.score == pytest.approx(1.0)


def test_ungrounded_claim_scores_zero(tmp_path: Path) -> None:
    claims = [_claim("c1", "Barcelona won the treble in 2015.")]
    export = _export(claims, responsive=True)
    dim, _store = _dim(tmp_path / "a.jsonl", claims, export)
    ctx = [ContextChunk(id="chunk-1", text="The weather in Madrid was sunny.")]
    result = dim.evaluate(EvalInput(question="q", answer="...", context=ctx))
    assert result.score == pytest.approx(0.0)


def test_responsive_is_imported_not_recomputed(tmp_path: Path) -> None:
    """Flipping the imported responsive verdict changes accuracy (proves import)."""
    claims = [_claim("c1", "Real Madrid won the Champions League final.")]
    ctx = [ContextChunk(id="chunk-1", text="Real Madrid won the Champions League final in 2024.")]
    ei = EvalInput(question="q", answer="...", context=ctx)

    dim_true, _ = _dim(tmp_path / "true.jsonl", claims, _export(claims, responsive=True))
    score_true = dim_true.evaluate(ei).score

    export_false = _export(claims, responsive=False)
    dim_false, _ = _dim(tmp_path / "false.jsonl", claims, export_false)
    result_false = dim_false.evaluate(ei)
    # the exported responsive atom id appears in accuracy's atom_ids (same atom)
    assert export_false.atom_id("c1") in result_false.atom_ids
    assert score_true == pytest.approx(1.0)
    assert result_false.score == pytest.approx(0.0)


def test_grounded_imported_from_groundedness_export(tmp_path: Path) -> None:
    """Flipping the imported grounded verdict changes accuracy (E3 wire)."""
    from rq_eval.dimensions.groundedness.export import GroundednessExport

    claims = [_claim("c1", "Real Madrid won the Champions League final.")]
    ctx = [ContextChunk(id="chunk-1", text="Real Madrid won the Champions League final in 2024.")]
    ei = EvalInput(question="q", answer="...", context=ctx)

    def _run(grounded: bool, path: Path) -> float:
        cfg = load_config()
        store = JsonlAtomStore(path)
        logger = AtomLogger(store, FixedClock())
        gexport = GroundednessExport()
        gexport.set(
            "c1",
            AtomRecord.create(subject="c1", role="grounded", question="grounded?", tier="T2",
                              verdict=grounded),
            [1.0 if grounded else 0.0],
        )
        dim = AccuracyDimension(
            ProviderFactory(cfg).build(), cfg, logger, claims, _export(claims, responsive=True),
            grounded_export=gexport,
        )
        return dim.evaluate(ei).score

    assert _run(True, tmp_path / "t.jsonl") == pytest.approx(1.0)
    assert _run(False, tmp_path / "f.jsonl") == pytest.approx(0.0)


def test_numeric_mismatch_fails_claim(tmp_path: Path) -> None:
    claims = [_claim("c1", "Revenue was $1.2B.")]
    export = _export(claims, responsive=True)
    dim, _store = _dim(tmp_path / "a.jsonl", claims, export)
    ctx = [ContextChunk(id="chunk-1", text="Revenue was $1.3B for the quarter.")]
    result = dim.evaluate(EvalInput(question="q", answer="...", context=ctx))
    # right topic, wrong number -> numeric exact-match fails the claim
    assert result.score == pytest.approx(0.0)


def test_accuracy_replays(tmp_path: Path) -> None:
    claims = [_claim("c1", "Real Madrid won the final."), _claim("c2", "The final was in London.")]
    export = _export(claims, responsive=True)
    dim, store = _dim(tmp_path / "a.jsonl", claims, export)
    ctx = [ContextChunk(id="chunk-1", text="Real Madrid won the final. The final was in London.")]
    result = dim.evaluate(EvalInput(question="q", answer="...", context=ctx))
    assert ReplayVerifier(default_registry()).verify(result, store) is True


def test_importance_weighting_changes_score(tmp_path: Path) -> None:
    claims = [
        _claim("c1", "Real Madrid won the Champions League final."),  # grounded -> correct
        _claim("c2", "Barcelona won the treble in 2015."),  # ungrounded -> incorrect
    ]
    export = _export(claims, responsive=True)
    ctx = [ContextChunk(id="chunk-1", text="Real Madrid won the Champions League final in 2024.")]
    ei = EvalInput(question="q", answer="...", context=ctx)

    # uniform weights: 1 of 2 correct -> 0.5
    dim_u, _ = _dim(tmp_path / "u.jsonl", claims, export, ImportanceWeights(enabled=False))
    assert dim_u.evaluate(ei).score == pytest.approx(0.5)

    # weight the correct (vital) claim heavily -> score rises
    weights = ImportanceWeights(enabled=True, weights={"c1": 3.0, "c2": 1.0})
    dim_w, _ = _dim(tmp_path / "w.jsonl", claims, export, weights)
    assert dim_w.evaluate(ei).score == pytest.approx(0.75)
