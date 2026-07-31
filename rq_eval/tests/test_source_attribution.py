"""E6 — source_attribution dimension (§4) + accuracy wiring, offline/mock."""

from __future__ import annotations

from pathlib import Path

import pytest

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.audit.replay import ReplayVerifier
from rq_eval.config import load_config
from rq_eval.contracts import Claim, ContextChunk, EvalInput
from rq_eval.dimensions.source_attribution.export import AttributionExport
from rq_eval.dimensions.source_attribution.source_attribution import SourceAttributionDimension
from rq_eval.providers.factory import ProviderFactory
from rq_eval.scoring.formulas import default_registry


def _claim(cid: str, text: str, citation: str | None) -> Claim:
    return Claim(
        id=cid, text=text, source_sentence=text, verifiable=True, decontextualized=True,
        extractor_version="claim-extractor-v1", citation=citation,
    )


def _dim(store_path: Path, claims):
    cfg = load_config()
    store = JsonlAtomStore(store_path)
    dim = SourceAttributionDimension(
        ProviderFactory(cfg).build(), cfg, AtomLogger(store, FixedClock()), claims,
        AttributionExport(),
    )
    return dim, store


def test_right_fact_wrong_citation_fails_attribution(tmp_path: Path) -> None:
    # claim is true, but cited chunk is about something else -> not attributable
    claims = [_claim("c1", "Real Madrid won the Champions League final.", "chunk-2")]
    ctx = [
        ContextChunk(id="chunk-1", text="Real Madrid won the Champions League final in 2024."),
        ContextChunk(id="chunk-2", text="Bananas are rich in potassium."),
    ]
    dim, _ = _dim(tmp_path / "a.jsonl", claims)
    result = dim.evaluate(EvalInput(question="q", answer="...", context=ctx))
    assert result.score == pytest.approx(0.0)  # cited chunk does not support it
    assert result.extra["citation_precision"] == pytest.approx(0.0)


def test_correct_citation_attributes(tmp_path: Path) -> None:
    claims = [_claim("c1", "Real Madrid won the Champions League final.", "chunk-1")]
    ctx = [ContextChunk(id="chunk-1", text="Real Madrid won the Champions League final in 2024.")]
    dim, _ = _dim(tmp_path / "a.jsonl", claims)
    result = dim.evaluate(EvalInput(question="q", answer="...", context=ctx))
    assert result.score == pytest.approx(1.0)
    assert result.extra["citation_recall"] == pytest.approx(1.0)


def test_no_citation_claims_excluded(tmp_path: Path) -> None:
    claims = [_claim("c1", "A cited claim.", "chunk-1"), _claim("c2", "No citation here.", None)]
    ctx = [ContextChunk(id="chunk-1", text="A cited claim.")]
    dim, _ = _dim(tmp_path / "a.jsonl", claims)
    result = dim.evaluate(EvalInput(question="q", answer="...", context=ctx))
    assert result.extra["cited_claims"] == pytest.approx(1.0)
    assert result.extra["excluded_no_citation"] == pytest.approx(1.0)
    assert result.n == 1


def test_attribution_replays(tmp_path: Path) -> None:
    claims = [_claim("c1", "Real Madrid won the final.", "chunk-1")]
    ctx = [ContextChunk(id="chunk-1", text="Real Madrid won the final in 2024.")]
    dim, store = _dim(tmp_path / "a.jsonl", claims)
    result = dim.evaluate(EvalInput(question="q", answer="...", context=ctx))
    assert ReplayVerifier(default_registry()).verify(result, store) is True


def test_accuracy_attributed_uses_real_provider(tmp_path: Path) -> None:
    """Right-fact/wrong-citation fails accuracy via the real attribution provider."""
    from rq_eval.contracts import AtomRecord
    from rq_eval.dimensions.accuracy.accuracy import AccuracyDimension
    from rq_eval.dimensions.responsiveness import ResponsivenessExport

    cfg = load_config()
    store = JsonlAtomStore(tmp_path / "acc.jsonl")
    logger = AtomLogger(store, FixedClock())
    claim = _claim("c1", "Real Madrid won the Champions League final.", "chunk-2")
    export = ResponsivenessExport()
    export.set("c1", AtomRecord.create(subject="c1", role="responsive", question="r",
                                       tier="T2", verdict=True))
    ctx = [
        ContextChunk(id="chunk-1", text="Real Madrid won the Champions League final."),
        ContextChunk(id="chunk-2", text="Bananas are rich in potassium."),
    ]
    dim = AccuracyDimension(ProviderFactory(cfg).build(), cfg, logger, [claim], export)
    dim.evaluate(EvalInput(question="q", answer="...", context=ctx))
    attributed = next(a for a in store.all() if a.role == "attributed")
    assert attributed.verdict is False  # wrong citation -> not attributed
    assert "label=" in attributed.evidence
