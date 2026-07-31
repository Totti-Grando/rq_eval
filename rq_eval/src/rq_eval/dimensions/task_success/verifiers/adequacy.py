"""[T3] adequacy verifier — the judge residue (§4).

The ONLY judge call in task_success: a per-outcome binary for the soft
"addresses the *root* cause / at the *right* level / is *sound*" outcomes that no
rule can capture. Everything else is T1/T2/import.
"""

from __future__ import annotations

from rq_eval.contracts import AtomRecord
from rq_eval.dimensions.task_success.task_templates import Outcome
from rq_eval.dimensions.task_success.verifiers.base import Verifier, VerifyContext
from rq_eval.graders.judge_grader import JudgeGrader


class AdequacyVerifier(Verifier):
    """[T3] Per-outcome judge verdict on an adequacy outcome."""

    def __init__(self, judge: JudgeGrader) -> None:
        """Inject the judge grader."""
        self._judge = judge

    def verify(self, outcome: Outcome, ctx: VerifyContext) -> AtomRecord:
        """Judge whether the answer adequately achieves the outcome."""
        cues = " ".join(str(c) for c in outcome.params.get("cues", []))
        return self._judge.judge(
            subject=f"outcome:{outcome.id}", role="outcome",
            question=f"[[overlap:0.34]] {cues}", context=ctx.answer,
            weight=outcome.weight, tier="T3",
        )
