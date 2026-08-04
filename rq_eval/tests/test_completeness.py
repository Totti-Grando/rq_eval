"""B8 — completeness dimension (§2, two-tier), offline/mock."""

from __future__ import annotations

from pathlib import Path

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.audit.replay import ReplayVerifier
from rq_eval.config import load_config
from rq_eval.contracts import ContextChunk, EvalInput
from rq_eval.dimensions.completeness.admissibility_gate import UnitAdmissibilityGate
from rq_eval.dimensions.completeness.completeness import CompletenessDimension
from rq_eval.dimensions.completeness.requirement_templates import RequirementTemplates
from rq_eval.dimensions.completeness.unit import Unit
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.graders.judge_grader import JudgeGrader
from rq_eval.graders.t1 import T1Tools
from rq_eval.providers.factory import ProviderFactory
from rq_eval.scoring.formulas import default_registry

_QUESTION = "What were the key drivers of the revenue decline?"


def _dim(store_path: Path):
    cfg = load_config()
    store = JsonlAtomStore(store_path)
    logger = AtomLogger(store, FixedClock())
    dim = CompletenessDimension(ProviderFactory(cfg).build(), cfg, logger)
    return dim, store


def test_templates_classify_and_load() -> None:
    templates = RequirementTemplates(load_config())
    assert templates.version == "templates-v1"
    reqs = templates.requirements_for(_QUESTION)
    assert {r.id for r in reqs} == {"cost_drivers", "pricing", "one_time", "fx"}
    assert templates.classify("Compare option A versus option B") == "comparison"
    assert templates.classify("Tell me about bananas") == "default"


def test_complete_answer_scores_higher_than_partial(tmp_path: Path) -> None:
    ctx = [ContextChunk(id="s1", text=(
        "Cost drivers behind the change included higher input costs. "
        "Pricing actions taken were modest discounts. "
        "One-time or non-recurring items included a legal settlement. "
        "Foreign-exchange effects reduced revenue."
    ))]
    complete = (
        "Cost drivers behind the change included higher input costs. "
        "Pricing actions taken were modest discounts. "
        "One-time or non-recurring items included a legal settlement. "
        "Foreign-exchange effects reduced revenue."
    )
    partial = "Cost drivers behind the change included higher input costs."

    dim_c, _ = _dim(tmp_path / "complete.jsonl")
    r_complete = dim_c.evaluate(EvalInput(question=_QUESTION, answer=complete, context=ctx))

    dim_p, _ = _dim(tmp_path / "partial.jsonl")
    r_partial = dim_p.evaluate(EvalInput(question=_QUESTION, answer=partial, context=ctx))

    assert r_complete.score >= r_partial.score
    # a wholly-missing facet is caught by requirement coverage
    assert r_partial.extra["requirement_coverage"] < 1.0


def test_frozen_set_carries_version_and_corpus_hash(tmp_path: Path) -> None:
    text = "Cost drivers included higher input costs."
    ctx = [ContextChunk(id="s1", text=text)]
    dim, store = _dim(tmp_path / "atoms.jsonl")
    dim.evaluate(EvalInput(question=_QUESTION, answer=text, context=ctx))
    frozen = [a for a in store.all() if a.role == "frozen_set"]
    assert frozen
    assert "corpus_hash=" in frozen[0].evidence
    assert "template=templates-v1" in frozen[0].evidence


def test_completeness_replays(tmp_path: Path) -> None:
    text = "Cost drivers included higher input costs. Pricing actions were discounts."
    ctx = [ContextChunk(id="s1", text=text)]
    dim, store = _dim(tmp_path / "atoms.jsonl")
    result = dim.evaluate(EvalInput(question=_QUESTION, answer=text, context=ctx))
    assert ReplayVerifier(default_registry()).verify(result, store) is True


def test_admissibility_deterministic_double_nli(tmp_path: Path) -> None:
    """R3: atomic (split) + self-contained + double-NLI; world-knowledge unit rejected."""
    cfg = load_config()
    providers = ProviderFactory(cfg).build()
    store = JsonlAtomStore(tmp_path / "a.jsonl")
    logger = AtomLogger(store, FixedClock())
    gate = UnitAdmissibilityGate(
        T1Tools(), providers.nlp,
        GroundingGrader(providers.grounding, logger, ("mock-grounding", "mock"),
                        "completeness.decidable_nli", 1),
        JudgeGrader(providers.judge, logger, ("mock-judge", "mock"),
                    "completeness.decidability_residual", 1),
        logger,
    )
    answer = "Revenue rose sharply and costs fell steadily in the quarter."
    sources = "Paris is the capital of France."
    units = [
        # non-atomic -> repaired by conjunction split into two atomic parts (both in the answer)
        Unit.create("Revenue rose sharply and costs fell steadily", "r1", True, "top_down"),
        # world-knowledge: absent from the answer, present in the corpus -> labels flip -> rejected
        Unit.create("Paris is the capital of France", "r1", True, "bottom_up"),
    ]
    admitted = [u.text for u in gate.admit(units, answer=answer, sources=sources)]
    assert "Revenue rose sharply" in admitted
    assert "costs fell steadily" in admitted
    assert "Paris is the capital of France" not in admitted  # double-NLI disagreement -> deferred
    # the residual judge is the only [T3] here and only fired on the disagreement
    residual_atoms = [a for a in store.all() if a.role == "decidability_residual"]
    assert residual_atoms and all(a.tier == "T3" for a in residual_atoms)
