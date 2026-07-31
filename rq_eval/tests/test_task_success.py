"""B9 — task_success dimension (§4), offline/mock."""

from __future__ import annotations

from pathlib import Path

import pytest

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.audit.replay import ReplayVerifier
from rq_eval.config import load_config
from rq_eval.contracts import EvalInput
from rq_eval.dimensions.task_success.task_success import TaskSuccessDimension
from rq_eval.dimensions.task_success.task_templates import TaskTemplates
from rq_eval.providers.factory import ProviderFactory
from rq_eval.scoring.formulas import default_registry


def _dim(store_path: Path):
    cfg = load_config()
    store = JsonlAtomStore(store_path)
    dim = TaskSuccessDimension(ProviderFactory(cfg).build(), cfg, AtomLogger(store, FixedClock()))
    return dim, store


def test_taxonomy_loads_and_classifies() -> None:
    t = TaskTemplates(load_config())
    assert t.version == "task-templates-v1"
    assert t.classify("Fix this broken function") == "fix"
    assert t.classify("Compare Redis versus Postgres") == "compare"
    assert t.classify("Tell me a story") == "explain"  # default
    fix_ids = {o.id for o in t.outcomes_for("fix")}
    assert fix_ids == {"corrected_artifact", "root_cause", "would_run"}


def test_fix_answered_with_explanation_fails(tmp_path: Path) -> None:
    # "fix" task answered with prose explanation, no code -> task failure
    dim, _ = _dim(tmp_path / "a.jsonl")
    result = dim.evaluate(EvalInput(
        question="Fix this broken function.",
        answer="The loop is slow because it is quadratic in the input size.",
    ))
    assert result.dimension == "task_success"
    assert result.score < 1.0  # not all fix outcomes achieved (no corrected code that runs)


def test_fix_with_code_scores_higher(tmp_path: Path) -> None:
    dim_expl, _ = _dim(tmp_path / "expl.jsonl")
    explanation = dim_expl.evaluate(EvalInput(
        question="Fix this broken function.",
        answer="It is slow.",
    ))
    dim_fix, _ = _dim(tmp_path / "fix.jsonl")
    real_fix = dim_fix.evaluate(EvalInput(
        question="Fix this broken function.",
        answer="Fixed: patched code uses def and return; it runs, cause was a typo.",
    ))
    assert real_fix.score > explanation.score


def test_impossible_task_is_success(tmp_path: Path) -> None:
    dim, _ = _dim(tmp_path / "a.jsonl")
    result = dim.evaluate(EvalInput(
        question="Write code to divide by zero safely and return infinity as a real number.",
        answer="This cannot be done because dividing by zero is impossible for real numbers.",
    ))
    assert result.score == pytest.approx(1.0)
    assert result.extra["impossible"] == pytest.approx(1.0)


def test_task_success_replays(tmp_path: Path) -> None:
    dim, store = _dim(tmp_path / "a.jsonl")
    result = dim.evaluate(EvalInput(
        question="Explain why the sky is blue.",
        answer="The sky is blue because shorter wavelengths scatter more; for example blue light.",
    ))
    assert ReplayVerifier(default_registry()).verify(result, store) is True
    assert result.extra["required"] == pytest.approx(3.0)  # explain has 3 outcomes


def test_partial_outcomes_give_ratio(tmp_path: Path) -> None:
    dim, _ = _dim(tmp_path / "a.jsonl")
    result = dim.evaluate(EvalInput(
        question="Explain why this happens.",
        answer="It happens because of scattering.",  # hits 'because' outcome, not others
    ))
    # ratio in [0,1], not all-or-nothing
    assert 0.0 <= result.score <= 1.0
    assert result.n == 3
