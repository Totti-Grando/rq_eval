"""B3 — contracts: atom id derivation and record validation (§0.5)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rq_eval.contracts import AtomRecord, Claim, ContextChunk, DimensionResult, EvalInput


def _atom(**over: object) -> AtomRecord:
    base: dict[str, object] = dict(
        subject="c1", role="grounded", question="q", tier="T2", verdict=True
    )
    base.update(over)
    return AtomRecord.create(**base)  # type: ignore[arg-type]


def test_atom_id_is_deterministic_and_excludes_timestamp() -> None:
    a = _atom(timestamp="2020-01-01T00:00:00+00:00")
    b = _atom(timestamp="2099-12-31T23:59:59+00:00")
    assert a.id == b.id  # timestamp not part of identity
    assert a.verdict is True


def test_atom_id_varies_by_subject_and_verdict() -> None:
    assert _atom(subject="c1").id != _atom(subject="c2").id
    assert _atom(verdict=True).id != _atom(verdict=False).id
    assert _atom(role="grounded").id != _atom(role="responsive").id


def test_dimension_result_bounds() -> None:
    ok = DimensionResult(
        dimension="accuracy",
        score=0.5,
        band="A",
        ci_low=0.3,
        ci_high=0.7,
        n=4,
        inputs_hash="h",
        atom_ids=["x"],
        formula_id="mean",
    )
    assert ok.score == 0.5
    with pytest.raises(ValidationError):
        DimensionResult(
            dimension="accuracy",
            score=1.5,  # out of [0,1]
            band="G",
            ci_low=0.0,
            ci_high=1.0,
            n=1,
            inputs_hash="h",
            atom_ids=[],
            formula_id="mean",
        )


def test_claim_and_eval_input_construct() -> None:
    claim = Claim(
        id="c1",
        text="The sky is blue.",
        source_sentence="The sky is blue today.",
        verifiable=True,
        decontextualized=True,
        extractor_version="claim-extractor-v1",
        citation="chunk-1",
    )
    assert claim.citation == "chunk-1"
    ei = EvalInput(
        question="why blue?",
        answer="rayleigh scattering",
        context=[ContextChunk(id="chunk-1", text="blue sky")],
    )
    assert ei.profile == "nexa"
    assert ei.context[0].id == "chunk-1"
