"""§4 task_success — verifier-routed goal accomplishment (design §4 v2).

Whether the user's *objective* would actually be achieved. Not irreducibly T3:
infer objective -> classify + pull a verifier-typed outcome template -> decompose
concrete outcomes -> route each outcome to its tagged verifier (T1 presence/
executable/state/constraint · T2 coverage · import grounded/responsive · T3
adequacy only) -> ``task_success = Σ achieved·w / Σ w``. A well-scoped
impossibility ("can't be done because X") is a success.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, DimensionResult, EvalInput
from rq_eval.dimensions.base import Dimension
from rq_eval.dimensions.task_success.objective import ObjectiveInference, OutcomeDecomposer
from rq_eval.dimensions.task_success.task_templates import TaskTemplates
from rq_eval.dimensions.task_success.verifiers.adequacy import AdequacyVerifier
from rq_eval.dimensions.task_success.verifiers.base import (
    Verifier,
    VerifierRouter,
    VerifyContext,
)
from rq_eval.dimensions.task_success.verifiers.constraint import ConstraintVerifier
from rq_eval.dimensions.task_success.verifiers.coverage import CoverageVerifier
from rq_eval.dimensions.task_success.verifiers.execution import ExecutionVerifier
from rq_eval.dimensions.task_success.verifiers.import_verifier import ImportVerifier
from rq_eval.dimensions.task_success.verifiers.presence import PresenceVerifier
from rq_eval.dimensions.task_success.verifiers.state import StateVerifier
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.graders.judge_grader import JudgeGrader
from rq_eval.graders.relevance_grader import RelevanceGrader
from rq_eval.providers.model_stamp import ModelStamp
from rq_eval.scoring.bands import BandMapper
from rq_eval.scoring.formulas import default_registry
from rq_eval.scoring.wilson import WilsonInterval

if TYPE_CHECKING:
    from rq_eval.config import Config
    from rq_eval.providers.factory import Providers

_FORMULA = "task_success_weighted"
# [T1] well-scoped impossibility: an impossibility marker + a stated reason.
_IMPOSSIBLE_MARKERS = ("cannot", "can't", "impossible", "not possible", "unable to")
_IMPOSSIBLE_REASON = ("because", "due to", "since ")


class TaskSuccessDimension(Dimension):
    """§4 — routes each required outcome to the cheapest verifier that fits."""

    name = "task_success"

    def __init__(self, providers: Providers, cfg: Config, logger: AtomLogger) -> None:
        """Assemble the objective chain + verifier router from providers + config."""
        self._cfg = cfg
        self._logger = logger
        stamp = ModelStamp(cfg)
        seed = cfg.seeds.judge
        self._templates = TaskTemplates(cfg)
        self._objective = ObjectiveInference(providers.generator, seed)
        self._decomposer = OutcomeDecomposer(providers.generator, seed)
        self._router = self._build_router(providers, cfg, logger, stamp, seed)
        self._registry = default_registry()
        self._bands = BandMapper(cfg.thresholds.bands.G, cfg.thresholds.bands.A)

    def _build_router(
        self, providers: Providers, cfg: Config, logger: AtomLogger, stamp: ModelStamp, seed: int
    ) -> VerifierRouter:
        rtau = cfg.thresholds.relevance_tau
        coverage_grader = GroundingGrader(
            providers.grounding, logger, stamp.grounding(), "task_success.coverage", seed
        )
        import_grounding = GroundingGrader(
            providers.grounding, logger, stamp.grounding(), "task_success.import_grounded", seed
        )
        import_relevance = RelevanceGrader(
            providers.relevance, rtau, logger, stamp.relevance(),
            "task_success.import_responsive", seed,
        )
        adequacy_judge = JudgeGrader(
            providers.judge, logger, stamp.judge(), "task_success.adequacy", seed
        )
        verifiers: dict[str, Verifier] = {
            "artifact_presence": PresenceVerifier(logger),
            "executable": ExecutionVerifier(logger, cfg.task_success.execution_sandbox),
            "state": StateVerifier(logger),
            "constraint": ConstraintVerifier(logger),
            "coverage": CoverageVerifier(coverage_grader),
            "import": ImportVerifier(import_grounding, import_relevance, logger),
            "adequacy": AdequacyVerifier(adequacy_judge),
        }
        return VerifierRouter(verifiers)

    def evaluate(self, eval_input: EvalInput) -> DimensionResult:
        """Infer, decompose, route each outcome, compute Σ achieved·w / Σ w."""
        q, a = eval_input.question, eval_input.answer
        inputs_hash = hashlib.sha256(f"{q}||{a}".encode()).hexdigest()[:16]
        ctx = VerifyContext(
            question=q, answer=a, context_text=" ".join(c.text for c in eval_input.context)
        )

        impossible = self._impossible(a)
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
        atoms: list[AtomRecord] = [self._router.route(oc, ctx) for oc in outcomes]
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

    def _impossible(self, answer: str) -> AtomRecord:
        """[T1] Well-scoped impossibility: an impossibility marker + a stated reason."""
        low = answer.lower()
        verdict = any(m in low for m in _IMPOSSIBLE_MARKERS) and any(
            r in low for r in _IMPOSSIBLE_REASON
        )
        return self._logger.record(
            subject="task", role="impossible_success", question="well-scoped impossibility?",
            tier="T1", verdict=verdict, evidence="marker+reason",
            grader_id="task_success.impossible", model="code", model_version="rq_eval",
        )

    def _log_task_type(self, task_type: str) -> None:
        """Record the classified task type for audit (not part of the score)."""
        self._logger.record(
            subject="task", role="task_type", question="task-type classification", tier="T1",
            verdict=True, evidence=f"type={task_type} taxonomy={self._templates.version}",
            grader_id="task_success.classify", model="code", model_version="rq_eval",
        )
