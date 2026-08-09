"""§3 relevance — direct core + scaffolded support-tree (build first; §1 imports it).

Two layers, de-risked. **Core (Layer 1, the built score):** the per-claim direct
on-topic + on-ask check (fixed NLI + lexical, DIVER-QA) — a mean with an off-ask
cap and abstention. **Extension (Layer 2, `relevance.tree_enabled`, default off):**
a support-tree over the answer that rescues indirectly-relevant claims — but it
reads the edges of the **one shared `ClaimGraph`** (§0.3), it does **not** build
its own. Anchors (on-ask seed + centrality + conformal recall) → reachability over
the shared support edges → orphan resolution (off-topic penalized; stranded /
veracity kept + routed to the ConsistencyProvider; background kept). When off,
relevance = the direct core. The per-claim ``responsive`` atom is exported for
accuracy either way.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, Claim, DimensionResult, EvalInput
from rq_eval.dimensions.base import Dimension
from rq_eval.dimensions.relevance.anchors import AnchorSelector
from rq_eval.dimensions.relevance.claim_responsiveness import ClaimResponsiveness, ClaimSignals
from rq_eval.dimensions.relevance.edges import Edge
from rq_eval.dimensions.relevance.method_a import MethodAReverseQuestions
from rq_eval.dimensions.relevance.method_b import MethodBGuardrail
from rq_eval.dimensions.relevance.orphans import OFF_TOPIC, STRANDED_CONTRADICTION, OrphanResolver
from rq_eval.dimensions.relevance.tree import SupportTree
from rq_eval.dimensions.responsiveness import ResponsivenessExport
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.graders.judge_grader import JudgeGrader
from rq_eval.graders.relevance_grader import RelevanceGrader
from rq_eval.graders.t1 import T1Tools
from rq_eval.pipeline.claim_graph import ClaimGraph
from rq_eval.providers.model_stamp import ModelStamp
from rq_eval.scoring.bands import BandMapper
from rq_eval.scoring.conformal import ConformalCalibrator
from rq_eval.scoring.formulas import default_registry
from rq_eval.scoring.wilson import WilsonInterval

if TYPE_CHECKING:
    from rq_eval.config import Config
    from rq_eval.providers.factory import Providers

_DECLINE = "[[overlap:0.4]] cannot answer decline refuse unable insufficient information"
_UNANSWERABLE = "[[deny]] Is this question impossible to answer from any available source?"
_CORE_FORMULA = "relevance_capped_mean"  # Layer 1 direct core (default)
_TREE_FORMULA = "relevance_tree_capped_mean"  # Layer 2 depth-graded (tree_enabled)
_CODE = ("code", "rq_eval")


class RelevanceDimension(Dimension):
    """§3 — anchor-tree relevance; exports per-claim responsiveness."""

    name = "relevance"

    def __init__(
        self,
        providers: Providers,
        cfg: Config,
        logger: AtomLogger,
        claims: list[Claim],
        export: ResponsivenessExport,
        graph: ClaimGraph | None = None,
    ) -> None:
        """Assemble graders + the direct core; the tree layer reads the shared graph."""
        self._cfg = cfg
        self._logger = logger
        self._claims = claims
        self._export = export
        self._method = cfg.relevance.method
        self._cap = cfg.relevance.off_ask_cap
        self._tree_enabled = cfg.relevance.tree_enabled
        self._graph = graph
        stamp = ModelStamp(cfg)
        seed = cfg.seeds.judge
        self._seed = seed
        self._t1 = T1Tools()
        self._grounding = providers.grounding
        self._lex_min = cfg.relevance.lexical_min_overlap
        self._grounding_stamp = stamp.grounding()
        self._on_topic = RelevanceGrader(
            providers.relevance, cfg.thresholds.relevance_tau, logger, stamp.relevance(),
            "relevance.on_topic", seed,
        )
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
        # Layer-2 tree pipeline (§3) — reads the shared graph's edges, builds none
        self._anchors = AnchorSelector(
            ConformalCalibrator(cfg.relevance.anchor_alpha, cfg.conformal.min_calibration_n),
            cfg.relevance.anchor_centrality_min,
        )
        self._tree = SupportTree(cfg.relevance.max_hops, cfg.relevance.depth_decay)
        self._orphans = OrphanResolver(providers.grounding, providers.consistency)
        self._method_a = MethodAReverseQuestions(
            providers.generator, providers.embedding, cfg.relevance.reverse_questions_n,
            cfg.seeds.reverse_questions,
        )
        self._method_b = MethodBGuardrail(self._on_topic)
        self._registry = default_registry()
        self._bands = BandMapper(cfg.thresholds.bands.G, cfg.thresholds.bands.A)
        self._abstain_stamp = stamp.judge()

    def evaluate(self, eval_input: EvalInput) -> DimensionResult:
        """Compute the tree-graded relevance score + band + CI; export responsiveness."""
        q, a = eval_input.question, eval_input.answer
        inputs_hash = self._hash(q, a)

        formula = _TREE_FORMULA if self._tree_active() else _CORE_FORMULA
        abstain = self._maybe_abstain(q, a)
        if abstain is not None:
            score = self._registry.compute(formula, [abstain])
            return DimensionResult(
                dimension=self.name, score=score, band=self._bands.band(score),
                ci_low=0.0, ci_high=1.0, n=len(self._claims), inputs_hash=inputs_hash,
                atom_ids=[abstain.id], formula_id=formula, abstained=True,
            )

        extra = self._answer_level_scores(q, a)
        answer_ask = self._answer_on_ask(q, a)
        signals = self._responsiveness.compute(q, self._claims, self._export)
        responsive_atoms = [s.responsive for s in signals]

        if self._tree_active():  # Layer 2: depth-graded over the shared graph
            graded = self._grade_claims(q, signals, extra)
            atoms: list[AtomRecord] = [answer_ask, *responsive_atoms, *graded]
            score = self._registry.compute(_TREE_FORMULA, [answer_ask, *graded])
            num_relevant = sum(1 for g in graded if g.verdict)
        else:  # Layer 1 direct core: mean of responsive with off-ask cap
            atoms = [answer_ask, *responsive_atoms]
            score = self._registry.compute(_CORE_FORMULA, atoms)
            num_relevant = sum(1 for s in responsive_atoms if s.verdict)

        low, high = WilsonInterval().interval(num_relevant, len(self._claims))
        return DimensionResult(
            dimension=self.name, score=score, band=self._bands.band(score),
            ci_low=low, ci_high=high, n=len(self._claims), inputs_hash=inputs_hash,
            atom_ids=[x.id for x in atoms], formula_id=formula, abstained=False, extra=extra,
        )

    def _tree_active(self) -> bool:
        """The scaffolded tree runs only when enabled AND the shared graph is present."""
        return self._tree_enabled and self._graph is not None

    def _grade_claims(
        self, question: str, signals: list[ClaimSignals], extra: dict[str, float]
    ) -> list[AtomRecord]:
        """Resolve the tree over the SHARED graph's edges + orphans; log one atom/claim.

        Relevance **reads** the shared graph's support edges — it constructs none.
        """
        by_id = {s.claim.id: s for s in signals}
        edges = self._shared_edges()
        seed_ids = {s.claim.id for s in signals if s.on_ask}
        # anchor confidence = the on-ask lexical coverage of the question's terms
        confidences = {
            s.claim.id: self._t1.key_term_overlap(question, s.claim.text) for s in signals
        }
        anchor_result = self._anchors.select(
            [s.claim.id for s in signals], seed_ids, edges, confidences
        )
        extra["anchor_band_low"] = anchor_result.band_low
        extra["anchor_count"] = float(len(anchor_result.anchor_ids))
        depth = self._tree.build(anchor_result.anchor_ids, edges)
        anchors = [by_id[i].claim for i in anchor_result.anchor_ids if i in by_id]

        graded: list[AtomRecord] = []
        for s in signals:
            graded.append(self._grade_one(s, depth, anchors))
        return graded

    def _shared_edges(self) -> list[Edge]:
        """Read the shared graph's confirmed support edges (relevance builds none)."""
        if self._graph is None:
            return []
        return [
            Edge(src=src, dst=dst, raw_score=1.0, marker=False)
            for src, dst, etype in self._graph.edges()
            if etype == "supports"
        ]

    def _grade_one(
        self, s: ClaimSignals, depth: dict[str, int], anchors: list[Claim]
    ) -> AtomRecord:
        """Grade a single claim by its tree depth, or by orphan classification."""
        if s.claim.id in depth:
            weight = self._tree.relevance_weight(depth[s.claim.id])
            return self._log_relevance(
                s.claim.id, verdict=True, weight=weight,
                evidence=f"in_tree depth={depth[s.claim.id]}",
            )
        verdict = self._orphans.classify(s.claim, s.on_topic, anchors)
        if verdict.kind == STRANDED_CONTRADICTION:
            self._logger.record(
                subject=s.claim.id, role="routed_contradiction",
                question="stranded contradiction routed to Reasoning", tier="T1",
                verdict=True, evidence=verdict.route_reason or "",
                grader_id="relevance.routed_contradiction", model=_CODE[0], model_version=_CODE[1],
            )
        weight = 0.0 if verdict.kind == OFF_TOPIC else 1.0
        return self._log_relevance(
            s.claim.id, verdict=verdict.relevant, weight=weight, evidence=f"orphan={verdict.kind}"
        )

    def _log_relevance(
        self, claim_id: str, *, verdict: bool, weight: float, evidence: str
    ) -> AtomRecord:
        """Log a code-graded ``claim_relevance`` atom (the score reads its weight)."""
        return self._logger.record(
            subject=claim_id, role="claim_relevance", question="relevance grade (tree depth)",
            tier="T2", verdict=verdict, weight=weight, evidence=evidence,
            grader_id="relevance.claim_relevance", model=self._grounding_stamp[0],
            model_version=self._grounding_stamp[1], seed=self._seed,
        )

    def _maybe_abstain(self, question: str, answer: str) -> AtomRecord | None:
        """Return an abstain-relevant atom iff a proper decline to unanswerable."""
        is_decline = self._decline.judge(
            subject="answer", role="decline", question=_DECLINE, context=answer,
            reference=question,
        ).verdict
        is_unanswerable = self._unans.judge(
            subject="answer", role="unanswerable", question=_UNANSWERABLE, context=question,
            reference=answer,
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
