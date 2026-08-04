"""B4 — §0 claim-extraction pipeline (offline, mock)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.config import load_config
from rq_eval.graders.t1 import T1Tools
from rq_eval.pipeline.claim_extractor import ClaimExtractor
from rq_eval.pipeline.pipeline import ClaimPipeline
from rq_eval.pipeline.prompts import PromptLibrary
from rq_eval.providers.factory import ProviderFactory


def _pipeline(logger: AtomLogger | None = None) -> ClaimPipeline:
    cfg = load_config()
    return ClaimPipeline(ProviderFactory(cfg).build(), cfg, logger=logger)


def test_prompt_library_pins_version() -> None:
    lib = PromptLibrary(load_config())
    assert lib.version == "claim-extractor-v1"
    assert "{clause}" not in lib.realize("the sky is blue")


def test_verifiable_spans_kept_unverifiable_routed(tmp_path: Path) -> None:
    store = JsonlAtomStore(tmp_path / "atoms.jsonl")
    result = _pipeline(AtomLogger(store, FixedClock())).run(
        "Real Madrid won the final in 2024. Maybe they will win again."
    )
    # first sentence is a checkable claim; the hedged "maybe" one is routed out
    assert any("Real Madrid" in c.text for c in result.claims)
    assert any("Maybe" in s for s in result.routed_unverifiable)
    # every claim carries all fields
    for c in result.claims:
        assert c.verifiable is True
        assert c.extractor_version == "claim-extractor-v1"
        assert c.id.startswith("claim:")
    # atoms were logged for the verifiable decisions
    assert any(a.role == "verifiable" for a in store.all())


def test_citation_is_extracted() -> None:
    result = _pipeline().run("The revenue rose 12 percent [chunk-1].")
    assert result.claims
    assert result.claims[0].citation == "chunk-1"


def test_stability_is_one_under_mock() -> None:
    result = _pipeline().run("Barcelona signed a new striker. The deal closed in July.")
    assert result.stability == pytest.approx(1.0)


def test_abstractive_implied_is_flagged_not_generated() -> None:
    """A bracketed abstractive placeholder is flagged (routed), never generated."""
    cfg = load_config()
    lib = PromptLibrary(cfg)
    providers = ProviderFactory(cfg).build()
    extractor = ClaimExtractor(
        providers.nlp,
        T1Tools(),
        providers.generator,
        lib,
        ("mock-generator", "mock"),
        seed=1,
        realizer_enabled=False,
    )
    # a Claimify-style implied fact "[a celebrity]" -> flagged, no claims
    assert extractor.extract("Credits include a film starring [a celebrity].") == []
    # a plain factual sentence decomposes deterministically into a claim
    assert extractor.extract("Real Madrid won the final.") == ["Real Madrid won the final."]


def test_realizer_impact_agreement_justifies_default_off() -> None:
    """Parse-form vs realized claims yield the same NLI verdict → realizer stays off.

    The realizer-impact property (RQ §0.2): if the fixed verifier consumes
    parse-form units without changing its verdicts, the surface-realizer is
    droppable. We extract both ways, run the grounding verifier against a source,
    and require verdict agreement — the evidence for ``realizer_enabled: false``.
    """
    cfg = load_config()
    lib = PromptLibrary(cfg)
    providers = ProviderFactory(cfg).build()
    source = "Real Madrid won the final in 2024."
    sentence = "Real Madrid won the final and the crowd celebrated."

    def _verdicts(realize: bool) -> list[bool]:
        extractor = ClaimExtractor(
            providers.nlp, T1Tools(), providers.generator, lib,
            ("mock-generator", "mock"), seed=1, realizer_enabled=realize,
        )
        return [
            providers.grounding.entails(source, claim).supported
            for claim in extractor.extract(sentence)
        ]

    assert _verdicts(realize=False) == _verdicts(realize=True)


def test_decontextualization_carries_context_forward() -> None:
    result = _pipeline().run(
        "They won the champions cup final.", context="Lionel Messi played for Argentina."
    )
    assert result.claims
    # leading pronoun resolved to the carried subject
    assert "Argentina" in result.claims[0].text
