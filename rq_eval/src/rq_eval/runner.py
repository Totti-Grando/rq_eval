"""End-to-end runner (build order B10 + E9, Phase D assembly).

Runs the §0 pipeline + triplet decomposition once, calibrates the conformal
layer, then the eight dimensions in dependency order — relevance (exports
responsive) and groundedness (exports grounded) before accuracy; hallucination
after groundedness — returning all eight results plus the atom log.
``python -m rq_eval.runner`` runs the fixture suite and prints reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.atom_store import AtomStore
from rq_eval.audit.atom_store_factory import AtomStoreFactory
from rq_eval.audit.calibration import CalibrationStore
from rq_eval.audit.clock import Clock, SystemClock
from rq_eval.config import Config, load_config
from rq_eval.contracts import AtomRecord, Claim, ContextChunk, DimensionResult, EvalInput, Profile
from rq_eval.dimensions.accuracy.accuracy import AccuracyDimension
from rq_eval.dimensions.completeness.completeness import CompletenessDimension
from rq_eval.dimensions.groundedness.export import GroundednessExport
from rq_eval.dimensions.groundedness.groundedness import GroundednessDimension
from rq_eval.dimensions.hallucination.hallucination import HallucinationDimension
from rq_eval.dimensions.relevance.relevance import RelevanceDimension
from rq_eval.dimensions.responsiveness import ResponsivenessExport
from rq_eval.dimensions.source_attribution.export import AttributionExport
from rq_eval.dimensions.source_attribution.source_attribution import SourceAttributionDimension
from rq_eval.dimensions.source_quality.source_quality import SourceQualityDimension
from rq_eval.dimensions.task_success.task_success import TaskSuccessDimension
from rq_eval.graders.t1 import T1Tools
from rq_eval.pipeline.claim_graph import ClaimGraph, ClaimGraphBuilder
from rq_eval.pipeline.edge_detection import EdgeDetector
from rq_eval.pipeline.pipeline import ClaimPipeline
from rq_eval.pipeline.triplets import ClaimTripletExtractor
from rq_eval.providers.factory import ProviderFactory, Providers
from rq_eval.scoring.conformal import ConformalCalibrator, ConformalResult, ConformalStratifier

_ORDER = (
    "accuracy", "completeness", "relevance", "task_success",
    "groundedness", "hallucination", "source_quality", "source_attribution",
)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One evaluation's eight dimension results + provenance."""

    results: dict[str, DimensionResult]
    claims: list[Claim]
    stability: float | None
    store: AtomStore
    conformal: ConformalResult
    atoms: list[AtomRecord] = field(default_factory=list)
    summary: str = ""  # read-only ExplanationJudge prose; never an input to any score
    graph: ClaimGraph | None = None  # the one shared claim graph (§0.3); read by projections


class Evaluator:
    """Orchestrates the pipeline + eight dimensions for one evaluation."""

    def __init__(
        self,
        cfg: Config,
        providers: Providers | None = None,
        store: AtomStore | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Inject config + (optionally) providers/store/clock; else build defaults."""
        self._cfg = cfg
        self._providers = providers or ProviderFactory(cfg).build()
        self._store = store or AtomStoreFactory(cfg).build()
        self._logger = AtomLogger(self._store, clock or SystemClock())

    def evaluate(self, eval_input: EvalInput) -> EvaluationResult:
        """Run §0 + triplets + conformal, then all eight dimensions in order."""
        p, cfg, log = self._providers, self._cfg, self._logger
        context_text = " ".join(c.text for c in eval_input.context)
        pres = ClaimPipeline(p, cfg, log).run(eval_input.answer, context_text)
        claims = pres.claims
        triplets = ClaimTripletExtractor(p.generator, p.nlp, T1Tools(), cfg).extract_all(claims)
        conformal = self._calibrate_conformal()

        responsive = ResponsivenessExport()
        grounded = GroundednessExport()
        attribution = AttributionExport()

        # groundedness first (builds the support set S), then the one shared claim
        # graph (§0.3), which relevance (reachability) and accuracy (derivation) both
        # read as projections. Edges are detected only when a Layer-2 flag reads them.
        groundedness = GroundednessDimension(p, cfg, log, triplets, grounded).evaluate(eval_input)
        graph = ClaimGraphBuilder(T1Tools(), p.nlp, grounded).build(claims)
        if cfg.accuracy.dag_rescue_enabled or cfg.relevance.tree_enabled:
            EdgeDetector(
                p.grounding, T1Tools(), cfg.graph.edge_tau, cfg.graph.topical_min,
                cfg.graph.numeric_tolerance,
            ).detect(claims, graph)
        relevance = RelevanceDimension(
            p, cfg, log, claims, responsive, graph=graph
        ).evaluate(eval_input)
        conformal_gate = None if conformal.abstained else conformal.threshold
        accuracy = AccuracyDimension(
            p, cfg, log, claims, grounded_export=grounded,
            attribution_conformal_threshold=conformal_gate, graph=graph,
        ).evaluate(eval_input)
        completeness = CompletenessDimension(p, cfg, log).evaluate(eval_input)
        task_success = TaskSuccessDimension(p, cfg, log).evaluate(eval_input)
        hallucination = HallucinationDimension(p, cfg, log, claims, grounded).evaluate(eval_input)
        source_quality = SourceQualityDimension(p, cfg, log, grounded).evaluate(eval_input)
        source_attribution = SourceAttributionDimension(
            p, cfg, log, claims, attribution, grounded_export=grounded
        ).evaluate(eval_input)
        source_attribution = self._stamp_conformal(source_attribution, conformal)
        groundedness = self._stamp_conformal(groundedness, conformal)

        results = {
            r.dimension: r
            for r in (
                accuracy, completeness, relevance, task_success,
                groundedness, hallucination, source_quality, source_attribution,
            )
        }
        atoms = self._store.all()
        # read-only summary AFTER all scores are final; never feeds a formula
        summary = p.explanation.summarize(results, atoms)
        return EvaluationResult(
            results=results, claims=claims, stability=pres.stability, store=self._store,
            conformal=conformal, atoms=atoms, summary=summary, graph=graph,
        )

    def _calibrate_conformal(self) -> ConformalResult:
        """Calibrate the (global) conformal threshold from the calibration set."""
        cal = self._cfg.conformal
        entails = self._providers.grounding.entails
        points = [
            (entails(ex.context, ex.claim).raw_score, ex.label, ex.stratum)
            for ex in CalibrationStore(self._cfg).examples()
        ]
        stratifier = ConformalStratifier(ConformalCalibrator(cal.alpha, cal.min_calibration_n))
        return stratifier.calibrate(points, cal.per_stratum)["__global__"]

    @staticmethod
    def _stamp_conformal(result: DimensionResult, conformal: ConformalResult) -> DimensionResult:
        """Stamp the conformal guarantee band onto a dimension result's extra."""
        extra = dict(result.extra)
        extra.update(
            conformal_band_low=conformal.band_low, conformal_band_high=conformal.band_high,
            conformal_threshold=conformal.threshold,
            conformal_abstained=1.0 if conformal.abstained else 0.0,
        )
        return result.model_copy(update={"extra": extra})


def evaluate(
    question: str,
    answer: str,
    context: list[str] | None = None,
    citations: list[str] | None = None,
    profile: Profile = "nexa",
    *,
    cfg: Config | None = None,
    store: AtomStore | None = None,
) -> EvaluationResult:
    """Convenience wrapper: build the EvalInput and run the Evaluator.

    ``context`` is a list of chunk texts (auto-assigned ids ``chunk-1..n``).
    """
    cfg = cfg or load_config()
    chunks = [ContextChunk(id=f"chunk-{i + 1}", text=t) for i, t in enumerate(context or [])]
    eval_input = EvalInput(
        question=question, answer=answer, context=chunks,
        citations=citations or [], profile=profile,
    )
    return Evaluator(cfg, store=store).evaluate(eval_input)


def _main() -> None:
    """Run the fixture suite end-to-end and print a report per case."""
    from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
    from rq_eval.fixtures import FixtureSuite
    from rq_eval.report import ReportRenderer

    cfg = load_config()
    renderer = ReportRenderer()
    runs_dir = cfg.resolve(cfg.paths.runs_dir)
    for case in FixtureSuite().cases():
        store = JsonlAtomStore(runs_dir / f"fixture_{case.name}.jsonl")
        result = Evaluator(cfg, store=store).evaluate(case.to_input())
        print(f"\n=== fixture: {case.name} — {case.note} ===")
        print(renderer.render(result))


if __name__ == "__main__":
    _main()
