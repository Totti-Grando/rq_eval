"""§3 relevance — on-topic + responsiveness (build first; §1 imports it).

Owns the responsiveness signal. Answer-level relevance (Method A diagnostic /
Method B gate) + per-claim responsive atoms, combined as a mean with an off-ask
cap; a proper decline to an unanswerable question scores relevant (abstention).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, Claim, DimensionResult, EvalInput
from rq_eval.dimensions.base import Dimension
from rq_eval.dimensions.relevance.claim_responsiveness import ClaimResponsiveness
from rq_eval.dimensions.relevance.method_a import MethodAReverseQuestions
from rq_eval.dimensions.relevance.method_b import MethodBGuardrail
from rq_eval.dimensions.responsiveness import ResponsivenessExport
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.graders.judge_grader import JudgeGrader
from rq_eval.graders.relevance_grader import RelevanceGrader
from rq_eval.graders.t1 import T1Tools
from rq_eval.providers.model_stamp import ModelStamp
from rq_eval.scoring.bands import BandMapper
from rq_eval.scoring.formulas import default_registry
from rq_eval.scoring.wilson import WilsonInterval

if TYPE_CHECKING:
    from rq_eval.config import Config
    from rq_eval.providers.factory import Providers

_DECLINE = "[[overlap:0.4]] cannot answer decline refuse unable insufficient information"
_UNANSWERABLE = "[[deny]] Is this question impossible to answer from any available source?"


class RelevanceDimension(Dimension):
    """§3 — scores fit to the question and exports per-claim responsiveness."""

    name = "relevance"

    def __init__(
        self,
        providers: Providers,
        cfg: Config,
        logger: AtomLogger,
        claims: list[Claim],
        export: ResponsivenessExport,
    ) -> None:
        """Assemble graders + methods from injected providers/config."""
        self._cfg = cfg
        self._logger = logger
        self._claims = claims
        self._export = export
        self._method = cfg.relevance.method
        self._cap = cfg.relevance.off_ask_cap
        stamp = ModelStamp(cfg)
        seed = cfg.seeds.judge
        self._seed = seed
        self._t1 = T1Tools()
        self._lex_min = cfg.relevance.lexical_min_overlap
        self._grounding_stamp = stamp.grounding()
        self._on_topic = RelevanceGrader(
            providers.relevance, cfg.thresholds.relevance_tau, logger, stamp.relevance(),
            "relevance.on_topic", seed,
        )
        # on-ask is now fixed NLI + lexical (DIVER-QA) — no judge on the on-ask path
        self._on_ask_nli = GroundingGrader(
            providers.grounding, logger, stamp.grounding(), "relevance.on_ask_nli", seed
        )
        self._decline = JudgeGrader(
            providers.judge, logger, stamp.judge(), "relevance.decline", seed
        )
        self._unans = JudgeGrader(
            providers.judge, logger, stamp.judge(), "relevance.unanswerable", seed
        )
        self._responsiveness = ClaimResponsiveness(
            self._on_topic, self._on_ask_nli, self._t1, logger, stamp.relevance(), seed,
            self._lex_min,
        )
        self._method_a = MethodAReverseQuestions(
            providers.generator, providers.embedding, cfg.relevance.reverse_questions_n,
            cfg.seeds.reverse_questions,
        )
        self._method_b = MethodBGuardrail(self._on_topic)
        self._registry = default_registry()
        self._bands = BandMapper(cfg.thresholds.bands.G, cfg.thresholds.bands.A)
        self._abstain_stamp = stamp.judge()

    def evaluate(self, eval_input: EvalInput) -> DimensionResult:
        """Compute the relevance score + band + CI; export responsiveness."""
        q, a = eval_input.question, eval_input.answer
        inputs_hash = self._hash(q, a)

        abstain = self._maybe_abstain(q, a)
        if abstain is not None:
            score = self._registry.compute("relevance_capped_mean", [abstain])
            return DimensionResult(
                dimension=self.name, score=score, band=self._bands.band(score),
                ci_low=0.0, ci_high=1.0, n=len(self._claims), inputs_hash=inputs_hash,
                atom_ids=[abstain.id], formula_id="relevance_capped_mean", abstained=True,
            )

        extra = self._answer_level_scores(q, a)
        answer_ask = self._answer_on_ask(q, a)
        responsive = self._responsiveness.compute(q, self._claims, self._export)
        atoms: list[AtomRecord] = [answer_ask, *responsive]
        score = self._registry.compute("relevance_capped_mean", atoms)

        num_responsive = sum(1 for x in responsive if x.verdict)
        low, high = WilsonInterval().interval(num_responsive, len(self._claims))
        return DimensionResult(
            dimension=self.name, score=score, band=self._bands.band(score),
            ci_low=low, ci_high=high, n=len(self._claims), inputs_hash=inputs_hash,
            atom_ids=[x.id for x in atoms], formula_id="relevance_capped_mean",
            abstained=False, extra=extra,
        )

    def _maybe_abstain(self, question: str, answer: str) -> AtomRecord | None:
        """Return an abstain-relevant atom iff a proper decline to unanswerable."""
        is_decline = self._decline.judge(
            subject="answer", role="decline", question=_DECLINE, context=answer
        ).verdict
        is_unanswerable = self._unans.judge(
            subject="answer", role="unanswerable", question=_UNANSWERABLE, context=question
        ).verdict
        if is_decline and is_unanswerable:
            return self._logger.record(
                subject="answer", role="abstain_relevant",
                question="proper decline to an unanswerable question", tier="T3", verdict=True,
                grader_id="relevance.abstain", model=self._abstain_stamp[0],
                model_version=self._abstain_stamp[1], seed=self._cfg.seeds.judge,
            )
        return None

    def _answer_on_ask(self, question: str, answer: str) -> AtomRecord:
        """Answer-level on-ask atom (fixed NLI ∨ lexical); weight carries the cap."""
        ask = self._t1.ask_hypothesis(question)
        nli = self._on_ask_nli.classify(premise=answer, hypothesis=ask).supported
        lex = self._t1.key_term_overlap(question, answer) >= self._lex_min
        return self._logger.record(
            subject="answer", role="on_ask_answer", question="answer addresses the ask?",
            tier="T2", verdict=(nli or lex), weight=self._cap,
            evidence=f"nli={nli} lex={lex}", grader_id="relevance.answer_on_ask",
            model=self._grounding_stamp[0], model_version=self._grounding_stamp[1], seed=self._seed,
        )

    def _answer_level_scores(self, question: str, answer: str) -> dict[str, float]:
        """Method-A diagnostic and/or Method-B gate scores (reported in extra)."""
        extra: dict[str, float] = {}
        if self._method in ("A", "both"):
            extra["method_a"] = self._method_a.score(question, answer)
        if self._method in ("B", "both"):
            extra["method_b"] = self._method_b.score(question, answer)
        return extra

    @staticmethod
    def _hash(question: str, answer: str) -> str:
        return hashlib.sha256(f"{question}||{answer}".encode()).hexdigest()[:16]
