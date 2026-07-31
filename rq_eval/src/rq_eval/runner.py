"""End-to-end runner (build order B10, Phase D assembly).

Runs the §0 pipeline once, then the four dimensions in the required order —
relevance first (it exports per-claim responsiveness), then accuracy (imports
it), completeness, and task_success — returning all four results plus the atom
log. ``python -m rq_eval.runner`` runs the fixture suite and prints reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.atom_store import AtomStore
from rq_eval.audit.atom_store_factory import AtomStoreFactory
from rq_eval.audit.clock import Clock, SystemClock
from rq_eval.config import Config, load_config
from rq_eval.contracts import AtomRecord, Claim, ContextChunk, DimensionResult, EvalInput, Profile
from rq_eval.dimensions.accuracy.accuracy import AccuracyDimension
from rq_eval.dimensions.completeness.completeness import CompletenessDimension
from rq_eval.dimensions.relevance.relevance import RelevanceDimension
from rq_eval.dimensions.responsiveness import ResponsivenessExport
from rq_eval.dimensions.task_success.task_success import TaskSuccessDimension
from rq_eval.pipeline.pipeline import ClaimPipeline
from rq_eval.providers.factory import ProviderFactory, Providers

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One evaluation's four dimension results + provenance."""

    results: dict[str, DimensionResult]
    claims: list[Claim]
    stability: float | None
    store: AtomStore
    atoms: list[AtomRecord] = field(default_factory=list)


class Evaluator:
    """Orchestrates the pipeline + four dimensions for one evaluation."""

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
        """Run §0 then the four dimensions (relevance before accuracy)."""
        context_text = " ".join(c.text for c in eval_input.context)
        pipeline = ClaimPipeline(self._providers, self._cfg, self._logger)
        pres = pipeline.run(eval_input.answer, context_text)
        claims = pres.claims

        export = ResponsivenessExport()
        relevance = RelevanceDimension(
            self._providers, self._cfg, self._logger, claims, export
        ).evaluate(eval_input)
        accuracy = AccuracyDimension(
            self._providers, self._cfg, self._logger, claims, export
        ).evaluate(eval_input)
        completeness = CompletenessDimension(
            self._providers, self._cfg, self._logger
        ).evaluate(eval_input)
        task_success = TaskSuccessDimension(
            self._providers, self._cfg, self._logger
        ).evaluate(eval_input)

        results = {
            r.dimension: r for r in (accuracy, completeness, relevance, task_success)
        }
        return EvaluationResult(
            results=results, claims=claims, stability=pres.stability,
            store=self._store, atoms=self._store.all(),
        )


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
