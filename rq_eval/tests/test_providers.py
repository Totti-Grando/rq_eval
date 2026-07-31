"""B2 — providers: determinism, factory-only construction, mock↔live selection."""

from __future__ import annotations

import pytest

from rq_eval.config import load_config
from rq_eval.providers.base import (
    EntailmentResult,
    GenerationResult,
    JudgeVerdict,
)
from rq_eval.providers.factory import ProviderFactory, Providers


@pytest.fixture
def providers() -> Providers:
    return ProviderFactory(load_config()).build()


def test_factory_builds_full_bundle(providers: Providers) -> None:
    assert isinstance(providers, Providers)
    for p in (
        providers.judge,
        providers.generator,
        providers.embedding,
        providers.grounding,
        providers.relevance,
        providers.nlp,
    ):
        assert p is not None


def test_judge_is_booleans_only(providers: Providers) -> None:
    v = providers.judge.binary("[[affirm]] anything?", "ctx")
    assert isinstance(v, JudgeVerdict)
    assert isinstance(v.verdict, bool)
    # the interface exposes exactly one public method: binary
    methods = [m for m in dir(type(providers.judge)) if not m.startswith("_")]
    assert methods == ["binary"]


def test_judge_tags_are_deterministic(providers: Providers) -> None:
    assert providers.judge.binary("[[affirm]] x", "y").verdict is True
    assert providers.judge.binary("[[deny]] x", "y").verdict is False
    # overlap: claim tokens fully present in context -> True; absent -> False
    assert providers.judge.binary("[[overlap]] alpha beta", "alpha beta gamma").verdict is True
    assert providers.judge.binary("[[overlap]] zeta theta", "alpha beta gamma").verdict is False
    # seeded default is stable across calls
    a = providers.judge.binary("no tag here", "ctx").verdict
    b = providers.judge.binary("no tag here", "ctx").verdict
    assert a == b


def test_grounding_three_way_entailment(providers: Providers) -> None:
    g = providers.grounding.entails("the sky is blue today", "sky is blue")
    assert isinstance(g, EntailmentResult)
    assert g.label in {"E", "N", "C"}
    assert 0.0 <= g.raw_score <= 1.0
    assert g.label == "E" and g.supported  # all hypothesis tokens covered
    # source silent on the hypothesis -> Neutral
    neutral = providers.grounding.entails("bananas grow in the tropics", "the sky is blue")
    assert neutral.label == "N"
    # negation mismatch with overlap -> Contradiction
    contra = providers.grounding.entails("the sky is blue", "the sky is not blue")
    assert contra.label == "C"
    r = providers.relevance.score("why is the sky blue", "the sky is blue")
    assert 0.0 <= r <= 1.0


def test_embedding_shape_and_determinism(providers: Providers) -> None:
    v1 = providers.embedding.embed(["alpha beta", "gamma"])
    v2 = providers.embedding.embed(["alpha beta", "gamma"])
    assert len(v1) == 2
    assert len({len(vec) for vec in v1}) == 1  # equal dimension
    assert v1 == v2  # deterministic


def test_generator_repeat_and_sentences(providers: Providers) -> None:
    rep = providers.generator.generate("[[repeat]] the answer", seed=1, n=3)
    assert isinstance(rep, GenerationResult)
    assert rep.items == ["the answer"] * 3
    sents = providers.generator.generate("[[sentences]] One. Two.", seed=1)
    assert sents.items == ["One.", "Two."]


def test_nlp_segment_and_coref(providers: Providers) -> None:
    assert providers.nlp.segment("First sentence. Second one!") == [
        "First sentence.",
        "Second one!",
    ]
    resolved = providers.nlp.resolve_coref("He scored.", context="Lionel Messi played.")
    assert resolved.resolved_text.startswith("Lionel Messi")


def _with(cfg, *, mode=None, nli=None):  # type: ignore[no-untyped-def]
    """Return a copy of cfg with providers.mode / models.nli overridden."""
    out = cfg
    if mode is not None:
        out = out.model_copy(update={"providers": out.providers.model_copy(update={"mode": mode})})
    if nli is not None:
        out = out.model_copy(update={"models": out.models.model_copy(update={"nli": nli})})
    return out


def _grounding_name(cfg) -> str:  # type: ignore[no-untyped-def]
    return type(ProviderFactory(cfg).build().grounding).__name__


def test_live_selection_is_config_only() -> None:
    """Flipping mode/nli changes provider classes without touching call sites."""
    cfg = load_config()
    live = _with(cfg, mode="live", nli="bedrock")
    bundle = ProviderFactory(live).build()
    # constructed lazily; no network/boto3 import until a method is called
    assert type(bundle.judge).__name__ == "BedrockJudgeProvider"
    assert type(bundle.embedding).__name__ == "TitanEmbeddingProvider"
    assert _grounding_name(live) == "GuardrailGroundingProvider"
    assert _grounding_name(_with(cfg, mode="live", nli="fairseq")) == "FairseqGroundingProvider"
    # nli: mock in live mode == partial-live (mock grounding), by design
    assert _grounding_name(_with(cfg, mode="live", nli="mock")) == "MockGroundingProvider"


def test_smoke_probes_pass_in_mock() -> None:
    results = ProviderFactory(load_config()).smoke_probes(lambda _name, fn: bool(fn()) or True)
    assert all(results)
