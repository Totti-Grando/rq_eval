"""E5 — source_quality dimension (§3) + accuracy wiring, offline/mock."""

from __future__ import annotations

from pathlib import Path

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.config import load_config
from rq_eval.contracts import ContextChunk, EvalInput
from rq_eval.dimensions.source_quality.reliability_list import ReliabilityList
from rq_eval.dimensions.source_quality.source_quality import SourceQualityDimension
from rq_eval.providers.factory import ProviderFactory


def _dim(store_path: Path):
    cfg = load_config()
    store = JsonlAtomStore(store_path)
    dim = SourceQualityDimension(ProviderFactory(cfg).build(), cfg, AtomLogger(store, FixedClock()))
    return dim, store


def test_reliability_list_allow_deny() -> None:
    rl = ReliabilityList(load_config())
    assert rl.version == "reliability-v1"
    assert rl.is_reliable("reuters.com") is True
    assert rl.is_reliable("fakenews.example") is False
    assert rl.is_reliable("unknown.example") is False  # non-empty allow-list -> not vetted
    assert rl.is_reliable(None) is True  # internal corpus


def test_coi_rule_flags_self_citation_and_denylist() -> None:
    """R4: disinterest decided by the [T1] COI rule (self-citation + denylist)."""
    from rq_eval.dimensions.source_quality.coi import CoiRule

    rule = CoiRule(load_config())
    # self-citation: a source about Acme published on acme.com -> affiliation conflict
    acme = ContextChunk(id="s1", text="Acme reported record profits.", domain="acme.com")
    verdict, reason = rule.decide(acme, "Acme reported record profits this quarter.")
    assert verdict is False and reason == "affiliation"
    # denylisted domain -> conflicted regardless of subject
    pr = ContextChunk(id="s2", text="x", domain="press.example")
    assert rule.decide(pr, "Beta Corp grew revenue.")[0] is False
    # independent web source -> ambiguous (rule not decisive; would sample the judge)
    ind = ContextChunk(id="s3", text="x", domain="reuters.com")
    assert rule.decide(ind, "Acme reported record profits.")[0] is None
    # internal corpus (no domain) -> decisively disinterested
    assert rule.decide(ContextChunk(id="s4", text="x"), "anything")[0] is True


def test_disinterest_is_t1_when_rule_decisive(tmp_path: Path) -> None:
    dim, store = _dim(tmp_path / "a.jsonl")
    ctx = [ContextChunk(id="chunk-1", text="Acme reported record profits.",
                        url="https://acme.com/pr", domain="acme.com")]
    dim.evaluate(EvalInput(question="q", answer="Acme reported record profits.", context=ctx))
    di = next(a for a in store.all() if a.role == "sq_disinterested")
    assert di.verdict is False and di.tier == "T1"  # COI rule decisive, no judge


def test_internal_source_scores_high(tmp_path: Path) -> None:
    # internal chunk (no url/domain) -> metadata checks satisfied by construction
    dim, store = _dim(tmp_path / "a.jsonl")
    text = "Revenue rose because input costs fell in the quarter."
    ctx = [ContextChunk(id="s1", text=text)]
    result = dim.evaluate(EvalInput(question="q", answer="Revenue rose; costs fell", context=ctx))
    assert result.dimension == "source_quality"
    assert result.score > 0.5
    # property atoms were logged for the source
    roles = {a.role for a in store.all() if a.subject == "source:s1"}
    assert {"sq_reachable", "sq_reputable", "sq_supports", "source_adequate"} <= roles


def test_bad_domain_source_scores_lower(tmp_path: Path) -> None:
    good, _ = _dim(tmp_path / "good.jsonl")
    text = "Revenue rose because input costs fell in the quarter."
    r_good = good.evaluate(EvalInput(
        question="q", answer=text,
        context=[ContextChunk(id="s1", text=text, url="https://reuters.com/x",
                              date="2026-01-01", author="J. Doe", domain="reuters.com")],
    ))
    bad, _ = _dim(tmp_path / "bad.jsonl")
    r_bad = bad.evaluate(EvalInput(
        question="q", answer=text,
        context=[ContextChunk(id="s1", text=text, url="https://fakenews.example/x",
                              domain="fakenews.example")],
    ))
    assert r_bad.score < r_good.score  # deny-listed, undated, unauthored


def test_accuracy_source_adequate_traces_to_real_properties(tmp_path: Path) -> None:
    """Accuracy's source-adequate atom now comes from the real provider (no stub)."""
    from rq_eval.contracts import AtomRecord, Claim
    from rq_eval.dimensions.accuracy.accuracy import AccuracyDimension
    from rq_eval.dimensions.groundedness.export import GroundednessExport

    cfg = load_config()
    store = JsonlAtomStore(tmp_path / "acc.jsonl")
    logger = AtomLogger(store, FixedClock())
    claim = Claim(
        id="c1", text="Revenue rose because costs fell.", source_sentence="x", verifiable=True,
        decontextualized=True, extractor_version="claim-extractor-v1", citation="chunk-1",
    )
    grounded = GroundednessExport()  # S supports c1 via chunk-1 (supports read off S)
    grounded.add_triplet("t:c1", "E", "c1", {"chunk-1"}, {"reuters.com"})
    grounded.set("c1", AtomRecord.create(subject="c1", role="grounded", question="g", tier="T2",
                                         verdict=True), [1.0])
    ctx = [ContextChunk(id="chunk-1", text="Revenue rose because input costs fell.",
                        url="https://reuters.com/x", date="2026-01-01", author="J. Doe",
                        domain="reuters.com")]
    dim = AccuracyDimension(
        ProviderFactory(cfg).build(), cfg, logger, [claim], grounded_export=grounded
    )
    dim.evaluate(EvalInput(question="q", answer="...", context=ctx))
    # the real property atoms are in the log (not a stub)
    roles = {a.role for a in store.all()}
    assert "sq_supports" in roles and "sq_reputable" in roles
    assert not any(a.model == "stub" for a in store.all())
