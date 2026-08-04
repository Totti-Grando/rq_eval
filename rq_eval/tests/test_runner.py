"""B10 + E9 — end-to-end runner, fixtures, report, whole-run replay (offline)."""

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

_DIMS = {
    "accuracy", "completeness", "relevance", "task_success",
    "groundedness", "hallucination", "source_quality", "source_attribution",
}


def _run(case_name: str, tmp_path: Path) -> EvaluationResult:
    cfg = load_config()
    case = next(c for c in FixtureSuite().cases() if c.name == case_name)
    store = JsonlAtomStore(tmp_path / f"{case_name}.jsonl")
    return Evaluator(cfg, store=store, clock=FixedClock()).evaluate(case.to_input())


def test_all_eight_dimensions_present(tmp_path: Path) -> None:
    assert set(_run("aligned", tmp_path).results) == _DIMS


def test_off_ask_relevance_is_capped(tmp_path: Path) -> None:
    result = _run("off_ask", tmp_path)
    assert result.results["relevance"].score <= load_config().relevance.off_ask_cap


def test_missing_facet_hurts_requirement_coverage(tmp_path: Path) -> None:
    # requirement coverage is a multi-facet signal -> pin the templated scaffold
    cfg = load_config()
    cfg = cfg.model_copy(
        update={"completeness": cfg.completeness.model_copy(update={"reference_mode": "templated"})}
    )
    case = next(c for c in FixtureSuite().cases() if c.name == "missing_facet")
    store = JsonlAtomStore(tmp_path / "missing_facet.jsonl")
    result = Evaluator(cfg, store=store, clock=FixedClock()).evaluate(case.to_input())
    assert result.results["completeness"].extra["requirement_coverage"] < 1.0
    assert result.results["completeness"].assurance_mode == "templated"


def test_explanation_instead_of_fix_low_task_success(tmp_path: Path) -> None:
    assert _run("explanation_instead_of_fix", tmp_path).results["task_success"].score < 1.0


def test_fabricated_citation_gates(tmp_path: Path) -> None:
    result = _run("fabricated_citation", tmp_path)
    assert result.results["hallucination"].extra["gate_failed"] == 1.0
    assert result.results["hallucination"].band == "R"


def test_wrong_citation_fails_attribution_but_grounds(tmp_path: Path) -> None:
    result = _run("wrong_citation", tmp_path)
    assert result.results["source_attribution"].score == pytest.approx(0.0)
    assert result.results["groundedness"].score > 0.0


def test_bad_source_fails_source_quality_but_grounds(tmp_path: Path) -> None:
    result = _run("bad_source", tmp_path)
    assert result.results["source_quality"].score < 0.6  # below adequacy threshold
    assert result.results["groundedness"].score > 0.0


def test_contradiction_reported(tmp_path: Path) -> None:
    result = _run("contradiction", tmp_path)
    assert result.results["hallucination"].extra["contradiction_rate"] > 0.0


def test_conformal_band_stamped(tmp_path: Path) -> None:
    result = _run("wrong_citation", tmp_path)
    extra = result.results["source_attribution"].extra
    assert extra["conformal_band_low"] == pytest.approx(1.0 - load_config().conformal.alpha)
    assert extra["conformal_band_high"] >= extra["conformal_band_low"]
    assert not result.conformal.abstained  # 7 factual calibration points >= min_n


def test_whole_run_replays(tmp_path: Path) -> None:
    result = _run("aligned", tmp_path)
    verifier = ReplayVerifier(default_registry())
    assert verifier.verify_run(list(result.results.values()), result.store) is True


def test_report_renders_eight(tmp_path: Path) -> None:
    text = ReportRenderer().render(_run("wrong_citation", tmp_path))
    for dim in _DIMS:
        assert dim in text
    assert "atoms by tier" in text


def test_convenience_wrapper(tmp_path: Path) -> None:
    result = evaluate(
        question="Who won the final?",
        answer="Real Madrid won the final.",
        context=["Real Madrid won the final in 2024."],
        store=JsonlAtomStore(tmp_path / "w.jsonl"),
    )
    assert result.results["accuracy"].score == pytest.approx(1.0)
    assert len(result.atoms) > 0
