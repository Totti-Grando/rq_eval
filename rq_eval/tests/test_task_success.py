"""B9 (v2) — verifier-routed task_success (design §4), offline/mock."""

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
from rq_eval.dimensions.task_success.task_templates import Outcome, TaskTemplates
from rq_eval.dimensions.task_success.verifiers.base import VerifyContext
from rq_eval.dimensions.task_success.verifiers.constraint import ConstraintVerifier
from rq_eval.dimensions.task_success.verifiers.execution import ExecutionVerifier
from rq_eval.dimensions.task_success.verifiers.presence import PresenceVerifier
from rq_eval.dimensions.task_success.verifiers.state import StateVerifier
from rq_eval.providers.factory import ProviderFactory
from rq_eval.scoring.formulas import default_registry


def _dim(store_path: Path):
    cfg = load_config()
    store = JsonlAtomStore(store_path)
    dim = TaskSuccessDimension(ProviderFactory(cfg).build(), cfg, AtomLogger(store, FixedClock()))
    return dim, store


def _logger(tmp_path: Path) -> AtomLogger:
    return AtomLogger(JsonlAtomStore(tmp_path / "v.jsonl"), FixedClock())


def _outcome(verifier: str, params: dict[str, object], weight: float = 1.0) -> Outcome:
    return Outcome(id="o", text="t", verifier=verifier, weight=weight, params=params)


def _ctx(answer: str, question: str = "q", context: str = "") -> VerifyContext:
    return VerifyContext(question=question, answer=answer, context_text=context)


# --- templates -------------------------------------------------------------- #

def test_taxonomy_v2_tags_verifiers() -> None:
    t = TaskTemplates(load_config())
    assert t.version == "task-templates-v2"
    assert t.classify("Fix this broken function") == "fix"
    tags = {o.id: o.verifier for o in t.outcomes_for("fix")}
    assert tags == {
        "corrected_artifact": "artifact_presence",
        "would_run": "executable",
        "root_cause": "adequacy",
    }


# --- individual T1 verifiers ------------------------------------------------ #

def test_presence_verifier(tmp_path: Path) -> None:
    v = PresenceVerifier(_logger(tmp_path))
    oc = _outcome("artifact_presence", {"patterns": ["def "]})
    assert v.verify(oc, _ctx("def f(): ...")).verdict
    assert not v.verify(oc, _ctx("just prose")).verdict


def test_executable_heuristic(tmp_path: Path) -> None:
    v = ExecutionVerifier(_logger(tmp_path), sandbox_enabled=False)
    params = {"signals": ["def "], "run_claims": ["runs"]}
    assert v.verify(_outcome("executable", params), _ctx("def f(): ... it runs")).verdict
    assert not v.verify(_outcome("executable", params), _ctx("it runs but no code")).verdict
    assert not v.verify(_outcome("executable", params), _ctx("def f(): ...")).verdict


def test_state_and_constraint_verifiers(tmp_path: Path) -> None:
    sv = StateVerifier(_logger(tmp_path))
    assert sv.verify(_outcome("state", {"expected": "created"}), _ctx("record created")).verdict
    assert sv.verify(_outcome("state", {}), _ctx("anything")).verdict  # no expected -> satisfied
    cv = ConstraintVerifier(_logger(tmp_path))
    con = _outcome("constraint", {"includes": ["api"], "max_words": 5})
    assert cv.verify(con, _ctx("the api works")).verdict
    too_long = cv.verify(_outcome("constraint", {"max_words": 2}), _ctx("one two three four"))
    assert not too_long.verdict


# --- dimension end-to-end --------------------------------------------------- #

def test_fix_with_code_beats_explanation(tmp_path: Path) -> None:
    dim_e, _ = _dim(tmp_path / "e.jsonl")
    explanation = dim_e.evaluate(EvalInput(
        question="Fix this broken function.",
        answer="It is slow because the loop is quadratic.",
    ))
    dim_f, _ = _dim(tmp_path / "f.jsonl")
    real_fix = dim_f.evaluate(EvalInput(
        question="Fix this broken function.",
        answer="Fixed: def solve(): return x. It runs; the cause was a typo.",
    ))
    assert real_fix.score > explanation.score
    assert real_fix.dimension == "task_success"


def test_only_adequacy_uses_the_judge(tmp_path: Path) -> None:
    """For a 'fix' task, only the adequacy outcome is a T3 judge atom."""
    dim, store = _dim(tmp_path / "a.jsonl")
    dim.evaluate(EvalInput(
        question="Fix this broken function.",
        answer="Fixed: def solve(): return x. It runs; the root cause was a typo.",
    ))
    outcome_atoms = [a for a in store.all() if a.role == "outcome"]
    tiers = {a.grader_id: a.tier for a in outcome_atoms}
    assert tiers["task_success.presence"] == "T1"
    assert tiers["task_success.executable"] == "T1"
    assert tiers["task_success.adequacy"] == "T3"
    # exactly one T3 outcome atom (the adequacy one)
    assert sum(1 for a in outcome_atoms if a.tier == "T3") == 1


def test_impossible_task_is_success(tmp_path: Path) -> None:
    dim, _ = _dim(tmp_path / "a.jsonl")
    result = dim.evaluate(EvalInput(
        question="Write code to divide by zero and return a real number.",
        answer="This cannot be done because dividing by zero is impossible for real numbers.",
    ))
    assert result.score == pytest.approx(1.0)
    assert result.extra["impossible"] == pytest.approx(1.0)


def test_weighted_and_replays(tmp_path: Path) -> None:
    dim, store = _dim(tmp_path / "a.jsonl")
    result = dim.evaluate(EvalInput(
        question="Explain why the sky is blue.",
        answer="The sky is blue because short wavelengths scatter, for example blue light.",
    ))
    assert 0.0 <= result.score <= 1.0
    assert ReplayVerifier(default_registry()).verify(result, store) is True
