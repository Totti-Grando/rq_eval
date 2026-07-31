"""§4 task_success — goal accomplishment (build order B9; built to design §4).

Whether the user's *objective* would actually be achieved (fit to goal, not
question). Genuinely Tier-3: infer objective → classify + pull outcome template →
decompose outcomes → judge each achieved → ``task_success = |achieved| /
|required outcomes|``. A well-scoped impossibility ("can't be done because X") is
a success.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, DimensionResult, EvalInput
from rq_eval.dimensions.base import Dimension
from rq_eval.dimensions.task_success.objective import ObjectiveInference, OutcomeDecomposer
from rq_eval.dimensions.task_success.task_templates import TaskTemplates
from rq_eval.graders.judge_grader import JudgeGrader
from rq_eval.providers.model_stamp import ModelStamp
from rq_eval.scoring.bands import BandMapper
from rq_eval.scoring.formulas import default_registry
from rq_eval.scoring.wilson import WilsonInterval

if TYPE_CHECKING:
    from rq_eval.config import Config
    from rq_eval.providers.factory import Providers

_FORMULA = "achieved_ratio"
_IMPOSSIBLE = "[[overlap:0.6]] cannot done impossible because"


class TaskSuccessDimension(Dimension):
    """§4 — scores whether the answer achieves the inferred objective."""

    name = "task_success"

    def __init__(self, providers: Providers, cfg: Config, logger: AtomLogger) -> None:
        """Assemble the objective/decompose/judge chain from providers + config."""
        self._cfg = cfg
        self._logger = logger
        stamp = ModelStamp(cfg)
        seed = cfg.seeds.judge
        self._templates = TaskTemplates(cfg)
        self._objective = ObjectiveInference(providers.generator, seed)
        self._decomposer = OutcomeDecomposer(providers.generator, seed)
        self._outcome_judge = JudgeGrader(
            providers.judge, logger, stamp.judge(), "task_success.outcome", seed
        )
        self._impossible_judge = JudgeGrader(
            providers.judge, logger, stamp.judge(), "task_success.impossible", seed
        )
        self._registry = default_registry()
        self._bands = BandMapper(cfg.thresholds.bands.G, cfg.thresholds.bands.A)

    def evaluate(self, eval_input: EvalInput) -> DimensionResult:
        """Infer objective, judge each outcome, compute achieved/required."""
        q, a = eval_input.question, eval_input.answer
        inputs_hash = hashlib.sha256(f"{q}||{a}".encode()).hexdigest()[:16]

        impossible = self._impossible_judge.judge(
            subject="task", role="impossible_success", question=_IMPOSSIBLE, context=a, tier="T3"
        )
        if impossible.verdict:  # well-scoped "can't be done because X" == success
            score = self._registry.compute(_FORMULA, [impossible])
            return DimensionResult(
                dimension=self.name, score=score, band=self._bands.band(score),
                ci_low=0.0, ci_high=1.0, n=0, inputs_hash=inputs_hash,
                atom_ids=[impossible.id], formula_id=_FORMULA, abstained=False,
                extra={"achieved": 0.0, "required": 0.0, "impossible": 1.0},
            )

        objective = self._objective.infer(q)
        task_type = self._templates.classify(q)
        outcomes = self._decomposer.decompose(self._templates.outcomes_for(task_type), objective)
        atoms: list[AtomRecord] = [
            self._outcome_judge.judge(
                subject=f"outcome:{task_type}:{oc.id}", role="outcome",
                question=f"[[overlap:0.5]] {' '.join(oc.cues)}", context=a, tier="T3",
            )
            for oc in outcomes
        ]
        self._log_task_type(task_type)
        score = self._registry.compute(_FORMULA, atoms)
        achieved = sum(1 for x in atoms if x.verdict)
        low, high = WilsonInterval().interval(achieved, len(atoms))
        return DimensionResult(
            dimension=self.name, score=score, band=self._bands.band(score),
            ci_low=low, ci_high=high, n=len(atoms), inputs_hash=inputs_hash,
            atom_ids=[x.id for x in atoms], formula_id=_FORMULA, abstained=False,
            extra={"achieved": float(achieved), "required": float(len(atoms))},
        )

    def _log_task_type(self, task_type: str) -> None:
        """Record the classified task type for audit (not part of the score)."""
        self._logger.record(
            subject="task", role="task_type", question="task-type classification", tier="T3",
            verdict=True, evidence=f"type={task_type} taxonomy={self._templates.version}",
            grader_id="task_success.classify", model="code", model_version="rq_eval",
        )
