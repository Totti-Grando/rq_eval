"""E4 — hallucination dimension (§2), offline/mock."""

from __future__ import annotations

from pathlib import Path

import pytest

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.audit.replay import ReplayVerifier
from rq_eval.config import load_config
from rq_eval.contracts import Claim, ContextChunk, EvalInput, Triplet
from rq_eval.dimensions.groundedness.export import GroundednessExport
from rq_eval.dimensions.groundedness.groundedness import GroundednessDimension
from rq_eval.dimensions.hallucination.hallucination import HallucinationDimension
from rq_eval.providers.factory import ProviderFactory
from rq_eval.scoring.formulas import default_registry


def _triplet(text: str, claim_id: str) -> Triplet:
    toks = text.split()
    return Triplet.create(
        claim_id=claim_id, subject=toks[0], predicate=toks[1] if len(toks) > 1 else "",
        obj=" ".join(toks[2:]), citation=None, source_pointer=text,
    )


def _run(tmp_path: Path, triplets, claims, ctx, citations=None):
    cfg = load_config()
    store = JsonlAtomStore(tmp_path / "atoms.jsonl")
    logger = AtomLogger(store, FixedClock())
    providers = ProviderFactory(cfg).build()
    export = GroundednessExport()
    GroundednessDimension(providers, cfg, logger, triplets, export).evaluate(
        EvalInput(question="q", answer="ans", context=ctx)
    )
    dim = HallucinationDimension(providers, cfg, logger, claims, export)
    result = dim.evaluate(EvalInput(question="q", answer="ans", context=ctx,
                                    citations=citations or []))
    return result, store


def test_unsupported_rate_and_nc_split(tmp_path: Path) -> None:
    # one entailed, one contradicted -> unsupported 0.5, contradiction 0.5
    triplets = [_triplet("Real Madrid won final", "c1"), _triplet("sky is not blue", "c2")]
    ctx = [ContextChunk(id="s1", text="Real Madrid won the final. The sky is blue.")]
    result, _ = _run(tmp_path, triplets, [], ctx)
    assert result.score == pytest.approx(0.5)  # unsupported rate
    assert result.extra["contradiction_rate"] == pytest.approx(0.5)
    assert result.extra["neutral_rate"] == pytest.approx(0.0)
    assert result.extra["gate_failed"] == 0.0


def test_fabricated_citation_gates(tmp_path: Path) -> None:
    triplets = [_triplet("Real Madrid won final", "c1")]
    claims = [Claim(
        id="c1", text="Real Madrid won the final.", source_sentence="x", verifiable=True,
        decontextualized=True, extractor_version="claim-extractor-v1", citation="fabricated-42",
    )]
    ctx = [ContextChunk(id="chunk-1", text="Real Madrid won the final.")]
    result, _ = _run(tmp_path, triplets, claims, ctx)
    assert result.extra["gate_failed"] == 1.0
    assert result.band == "R"


def test_valid_citation_passes_gate(tmp_path: Path) -> None:
    triplets = [_triplet("Real Madrid won final", "c1")]
    claims = [Claim(
        id="c1", text="Real Madrid won the final.", source_sentence="x", verifiable=True,
        decontextualized=True, extractor_version="claim-extractor-v1", citation="chunk-1",
    )]
    ctx = [ContextChunk(id="chunk-1", text="Real Madrid won the final.")]
    result, _ = _run(tmp_path, triplets, claims, ctx)
    assert result.extra["gate_failed"] == 0.0


def test_unsupported_rate_replays(tmp_path: Path) -> None:
    triplets = [_triplet("Real Madrid won final", "c1"), _triplet("Barcelona lost match", "c2")]
    ctx = [ContextChunk(id="s1", text="Real Madrid won the final. Barcelona lost the match.")]
    result, store = _run(tmp_path, triplets, [], ctx)
    assert ReplayVerifier(default_registry()).verify(result, store) is True
