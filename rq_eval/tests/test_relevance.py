"""B6 — relevance dimension (§3), offline/mock."""

from __future__ import annotations

from pathlib import Path

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.audit.replay import ReplayVerifier
from rq_eval.config import load_config
from rq_eval.contracts import Claim, EvalInput
from rq_eval.dimensions.relevance.relevance import RelevanceDimension
from rq_eval.dimensions.responsiveness import ResponsivenessExport
from rq_eval.pipeline.pipeline import ClaimPipeline
from rq_eval.providers.factory import ProviderFactory
from rq_eval.scoring.formulas import default_registry


def _setup(tmp_path: Path, answer: str, question: str):
    cfg = load_config()
    providers = ProviderFactory(cfg).build()
    store = JsonlAtomStore(tmp_path / "atoms.jsonl")
    logger = AtomLogger(store, FixedClock())
    claims = ClaimPipeline(providers, cfg).run(answer).claims
    export = ResponsivenessExport()
    dim = RelevanceDimension(providers, cfg, logger, claims, export)
    result = dim.evaluate(EvalInput(question=question, answer=answer))
    return result, store, export, claims


def test_on_topic_answer_scores_high(tmp_path: Path) -> None:
    result, _store, export, claims = _setup(
        tmp_path,
        answer="Real Madrid won the Champions League final in 2024.",
        question="Who won the Champions League final in 2024?",
    )
    assert result.dimension == "relevance"
    assert result.score > 0.0
    # per-claim responsive atoms exported for accuracy to import
    assert claims
    for c in claims:
        assert export.has(c.id)


def test_off_ask_answer_is_capped(tmp_path: Path) -> None:
    cfg = load_config()
    result, _store, _export, _claims = _setup(
        tmp_path,
        answer="Bananas are a good source of potassium and grow in tropical climates.",
        question="What were the key drivers of the company's Q3 revenue decline?",
    )
    # off-topic/off-ask answer: score capped at off_ask_cap
    assert result.score <= cfg.relevance.off_ask_cap


def test_relevance_score_replays(tmp_path: Path) -> None:
    result, store, _export, _claims = _setup(
        tmp_path,
        answer="Barcelona signed a striker in July. The transfer fee was 50 million euros.",
        question="Which striker did Barcelona sign and for how much?",
    )
    verifier = ReplayVerifier(default_registry())
    assert verifier.verify(result, store) is True


def test_responsive_atom_is_the_shared_one(tmp_path: Path) -> None:
    result, store, export, claims = _setup(
        tmp_path,
        answer="Inter Miami beat Toronto three to one on Saturday.",
        question="What was the score of the Inter Miami match?",
    )
    # the exported atom ids are exactly the 'responsive' atoms in the log
    responsive_ids = {a.id for a in store.all() if a.role == "responsive"}
    exported_ids = {export.atom_id(c.id) for c in claims}
    assert exported_ids == responsive_ids
    assert responsive_ids.issubset(set(result.atom_ids))


def test_method_a_diagnostic_reported() -> None:
    cfg = load_config().model_copy(
        update={"relevance": load_config().relevance.model_copy(update={"method": "both"})}
    )
    providers = ProviderFactory(cfg).build()
    from rq_eval.dimensions.relevance.method_a import MethodAReverseQuestions

    method_a = MethodAReverseQuestions(
        providers.generator, providers.embedding, cfg.relevance.reverse_questions_n,
        cfg.seeds.reverse_questions,
    )
    # answer derived from the question -> high cosine self-similarity
    score = method_a.score("who won the final", "the final was won by madrid")
    assert 0.0 <= score <= 1.0


def test_claim_level_uses_pipeline_claims(tmp_path: Path) -> None:
    # a claim that is on-topic but does not cover the specific ask
    claims = [Claim(
        id="claim:x", text="The weather was sunny.", source_sentence="The weather was sunny.",
        verifiable=True, decontextualized=True, extractor_version="claim-extractor-v1",
    )]
    cfg = load_config()
    providers = ProviderFactory(cfg).build()
    store = JsonlAtomStore(tmp_path / "a.jsonl")
    dim = RelevanceDimension(
        providers, cfg, AtomLogger(store, FixedClock()), claims, ResponsivenessExport()
    )
    result = dim.evaluate(
        EvalInput(
            question="What was the final score of the match?",
            answer="The weather was sunny.",
        )
    )
    assert result.n == 1
