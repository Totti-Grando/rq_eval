"""R5 — every remaining ScoringJudge call is reference-grounded (not None)."""

from __future__ import annotations

from pathlib import Path

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.dimensions.task_success.task_templates import Outcome
from rq_eval.dimensions.task_success.verifiers.adequacy import AdequacyVerifier
from rq_eval.dimensions.task_success.verifiers.base import VerifyContext
from rq_eval.graders.judge_grader import JudgeGrader
from rq_eval.providers.base import JudgeVerdict


class _SpyJudge:
    """Records the reference passed on the last binary() call."""

    def __init__(self) -> None:
        self.last_reference: str | None = "SENTINEL"

    def binary(self, question: str, context: str, reference: str | None = None) -> JudgeVerdict:
        self.last_reference = reference
        return JudgeVerdict(True, "spy")


def _grader(spy: _SpyJudge, tmp_path: Path) -> JudgeGrader:
    logger = AtomLogger(JsonlAtomStore(tmp_path / "a.jsonl"), FixedClock())
    return JudgeGrader(spy, logger, ("spy", "spy"), "test.spy", 1)  # type: ignore[arg-type]


def test_judge_grader_threads_reference(tmp_path: Path) -> None:
    spy = _SpyJudge()
    _grader(spy, tmp_path).judge(
        subject="s", role="r", question="q?", context="ctx", reference="THE-REFERENCE"
    )
    assert spy.last_reference == "THE-REFERENCE"


def test_adequacy_verifier_references_the_template(tmp_path: Path) -> None:
    spy = _SpyJudge()
    verifier = AdequacyVerifier(_grader(spy, tmp_path))
    outcome = Outcome(
        id="root_cause", text="the root cause is addressed", verifier="adequacy",
        weight=1.0, params={"cues": ["because", "cause"]},
    )
    verifier.verify(outcome, VerifyContext(question="q", answer="a", context_text=""))
    assert spy.last_reference == "the root cause is addressed"  # pinned template, not None
