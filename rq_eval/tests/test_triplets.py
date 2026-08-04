"""E2 — claim-triplet decomposition (§0), offline/mock."""

from __future__ import annotations

from rq_eval.config import load_config
from rq_eval.contracts import Claim
from rq_eval.graders.t1 import T1Tools
from rq_eval.pipeline.triplets import ClaimTripletExtractor, TripletStabilityHarness
from rq_eval.providers.factory import ProviderFactory


def _claim(text: str, cid: str = "claim:1", citation: str | None = "chunk-1") -> Claim:
    return Claim(
        id=cid, text=text, source_sentence=text, verifiable=True, decontextualized=True,
        extractor_version="claim-extractor-v1", citation=citation,
    )


def _extractor() -> ClaimTripletExtractor:
    cfg = load_config()
    providers = ProviderFactory(cfg).build()
    return ClaimTripletExtractor(providers.generator, providers.nlp, T1Tools(), cfg)


def test_multi_triplet_claim_with_provenance() -> None:
    ext = _extractor()
    triplets = ext.extract(_claim("Einstein developed quantum mechanics and was in Berlin"))
    assert len(triplets) == 2  # two clauses -> two triplets
    for t in triplets:
        assert t.claim_id == "claim:1"
        assert t.citation == "chunk-1"
        assert t.source_pointer
        assert t.id.startswith("triplet:")
        assert t.text


def test_every_claim_yields_at_least_one_triplet() -> None:
    ext = _extractor()
    assert len(ext.extract(_claim("Short."))) >= 1


def test_version_pinned() -> None:
    assert _extractor().version == "triplet-extractor-v1"


def test_triplet_stability_is_one_under_mock() -> None:
    ext = _extractor()
    claims = [_claim("Barcelona signed a striker and paid a large fee", "claim:a")]
    assert TripletStabilityHarness(ext).measure(claims, runs=3) == 1.0


class _SpyGenerator:
    """Wraps the mock generator, counting generate() calls (residual path only)."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.calls = 0

    def generate(self, prompt: str, *, seed: int, n: int = 1):  # type: ignore[no-untyped-def]
        self.calls += 1
        return self._inner.generate(prompt, seed=seed, n=n)  # type: ignore[attr-defined]


def _spy_extractor() -> tuple[ClaimTripletExtractor, _SpyGenerator]:
    cfg = load_config()
    providers = ProviderFactory(cfg).build()
    spy = _SpyGenerator(providers.generator)
    return ClaimTripletExtractor(spy, providers.nlp, T1Tools(), cfg), spy  # type: ignore[arg-type]


def test_clean_claim_parses_with_no_generator_call() -> None:
    """A cleanly-parseable claim produces triplets without invoking the generator."""
    ext, spy = _spy_extractor()
    triplets = ext.extract(_claim("Einstein developed quantum mechanics"))
    assert triplets and spy.calls == 0  # parse-first, no [T3-gen]


def test_nested_claim_invokes_generator_residual() -> None:
    """A nested predicate the parser can't triple falls to the pinned generator."""
    ext, spy = _spy_extractor()
    ext.extract(_claim("Einstein believed that light bends near mass"))
    assert spy.calls >= 1  # residual only
