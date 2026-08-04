"""U4 — completeness reference modes: generated (default) / archetype / templated."""

from __future__ import annotations

from pathlib import Path

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.config import Config, load_config
from rq_eval.contracts import ContextChunk, EvalInput
from rq_eval.dimensions.completeness.archetype_templates import ArchetypeTemplates
from rq_eval.dimensions.completeness.completeness import CompletenessDimension
from rq_eval.dimensions.completeness.reference import ReferenceModeSelector
from rq_eval.dimensions.completeness.requirement_templates import RequirementTemplates
from rq_eval.providers.factory import ProviderFactory

_DRIVERS = "What were the key drivers of the revenue decline?"
_ARBITRARY = "Describe the migratory patterns of arctic terns."


def _cfg(mode: str) -> Config:
    cfg = load_config()
    return cfg.model_copy(
        update={"completeness": cfg.completeness.model_copy(update={"reference_mode": mode})}
    )


def _evaluate(mode: str, question: str, answer: str, tmp_path: Path):
    cfg = _cfg(mode)
    store = JsonlAtomStore(tmp_path / f"{mode}.jsonl")
    dim = CompletenessDimension(
        ProviderFactory(cfg).build(), cfg, AtomLogger(store, FixedClock())
    )
    ctx = [ContextChunk(id="s1", text=answer)]
    return dim.evaluate(EvalInput(question=question, answer=answer, context=ctx))


def test_default_reference_mode_is_generated() -> None:
    assert load_config().completeness.reference_mode == "generated"


def test_archetype_selector_classifies_and_instantiates() -> None:
    arch = ArchetypeTemplates(load_config())
    assert arch.version == "archetypes-v1"
    assert arch.classify("Why did revenue fall?") == "causal_explanation"
    assert arch.classify("Compare A versus B") == "comparison"
    reqs = arch.requirements_for("Compare A versus B")
    assert {r.id for r in reqs} == {"option_a", "option_b", "criteria", "verdict"}


def test_selector_routes_each_mode() -> None:
    cfg_gen = _cfg("generated")
    gen_sel = ReferenceModeSelector(
        cfg_gen, ProviderFactory(cfg_gen).build().generator, RequirementTemplates(cfg_gen)
    )
    assert gen_sel.mode == "generated"
    assert gen_sel.requirements_for(_ARBITRARY)  # generated works with no template match

    cfg_arch = _cfg("archetype")
    arch_sel = ReferenceModeSelector(
        cfg_arch, ProviderFactory(cfg_arch).build().generator, RequirementTemplates(cfg_arch)
    )
    assert arch_sel.requirements_for(_DRIVERS)  # causal_explanation shape


def test_all_three_modes_run_and_stamp_assurance(tmp_path: Path) -> None:
    answer = "Cost drivers included higher input costs."
    for mode in ("generated", "archetype", "templated"):
        result = _evaluate(mode, _DRIVERS, answer, tmp_path)
        assert result.assurance_mode == mode
        assert 0.0 <= result.score <= 1.0


def test_generated_mode_works_on_arbitrary_question(tmp_path: Path) -> None:
    """The open-domain default produces a score with no template for the question."""
    result = _evaluate(
        "generated", _ARBITRARY, "Arctic terns migrate from the Arctic to the Antarctic.", tmp_path
    )
    assert result.assurance_mode == "generated"
    assert result.dimension == "completeness"


def test_recall_sample_miss_rate_reported(tmp_path: Path) -> None:
    """The human should-contain sample reports a miss-rate error bar on the drivers Q."""
    partial = "Cost drivers included higher input costs."
    result = _evaluate("templated", _DRIVERS, partial, tmp_path)
    assert "recall_miss_rate" in result.extra
    assert 0.0 <= result.extra["recall_miss_rate"] <= 1.0
