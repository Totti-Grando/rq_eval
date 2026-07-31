"""B10 — end-to-end runner, fixtures, report, whole-run replay (offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.audit.replay import ReplayVerifier
from rq_eval.config import load_config
from rq_eval.fixtures import FixtureSuite
from rq_eval.report import ReportRenderer
from rq_eval.runner import EvaluationResult, Evaluator, evaluate
from rq_eval.scoring.formulas import default_registry


def _run(case_name: str, tmp_path: Path) -> EvaluationResult:
    cfg = load_config()
    case = next(c for c in FixtureSuite().cases() if c.name == case_name)
    store = JsonlAtomStore(tmp_path / f"{case_name}.jsonl")
    return Evaluator(cfg, store=store, clock=FixedClock()).evaluate(case.to_input())


def test_all_four_dimensions_present(tmp_path: Path) -> None:
    result = _run("aligned", tmp_path)
    assert set(result.results) == {"accuracy", "completeness", "relevance", "task_success"}


def test_aligned_covers_every_facet(tmp_path: Path) -> None:
    # aligned answer covers all four drivers facets -> full requirement coverage
    result = _run("aligned", tmp_path)
    assert result.results["completeness"].extra["requirement_coverage"] == pytest.approx(1.0)
    assert not result.results["accuracy"].abstained
    # more complete than the missing-facet answer
    missing = _run("missing_facet", tmp_path)
    assert (
        result.results["completeness"].extra["requirement_coverage"]
        > missing.results["completeness"].extra["requirement_coverage"]
    )


def test_off_ask_relevance_is_capped(tmp_path: Path) -> None:
    result = _run("off_ask", tmp_path)
    assert result.results["relevance"].score <= load_config().relevance.off_ask_cap


def test_missing_facet_hurts_requirement_coverage(tmp_path: Path) -> None:
    result = _run("missing_facet", tmp_path)
    assert result.results["completeness"].extra["requirement_coverage"] < 1.0


def test_explanation_instead_of_fix_low_task_success(tmp_path: Path) -> None:
    result = _run("explanation_instead_of_fix", tmp_path)
    assert result.results["task_success"].score < 1.0


def test_whole_run_replays(tmp_path: Path) -> None:
    result = _run("aligned", tmp_path)
    verifier = ReplayVerifier(default_registry())
    assert verifier.verify_run(list(result.results.values()), result.store) is True


def test_report_renders(tmp_path: Path) -> None:
    result = _run("aligned", tmp_path)
    text = ReportRenderer().render(result)
    for dim in ("accuracy", "completeness", "relevance", "task_success"):
        assert dim in text
    assert "atoms by tier" in text


def test_convenience_wrapper(tmp_path: Path) -> None:
    # high lexical overlap -> grounded + responsive -> accuracy 1.0 end-to-end
    result = evaluate(
        question="Who won the final?",
        answer="Real Madrid won the final.",
        context=["Real Madrid won the final in 2024."],
        store=JsonlAtomStore(tmp_path / "w.jsonl"),
    )
    assert result.results["accuracy"].score == pytest.approx(1.0)
    assert len(result.atoms) > 0
