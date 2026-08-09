"""G9 — end-to-end: both Layer-2 flags on still runs, replays, and stays whole."""

from __future__ import annotations

from pathlib import Path

from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.audit.replay import ReplayVerifier
from rq_eval.config import Config, load_config
from rq_eval.fixtures import FixtureSuite
from rq_eval.runner import Evaluator
from rq_eval.scoring.formulas import default_registry

_DIMS = {
    "accuracy", "completeness", "relevance", "task_success",
    "groundedness", "hallucination", "source_quality", "source_attribution",
}


def _layer2_on() -> Config:
    cfg = load_config()
    return cfg.model_copy(
        update={
            "accuracy": cfg.accuracy.model_copy(update={"dag_rescue_enabled": True}),
            "relevance": cfg.relevance.model_copy(update={"tree_enabled": True}),
        }
    )


def _run(cfg: Config, case_name: str, tmp_path: Path):
    case = next(c for c in FixtureSuite().cases() if c.name == case_name)
    store = JsonlAtomStore(tmp_path / f"{case_name}.jsonl")
    return Evaluator(cfg, store=store, clock=FixedClock()).evaluate(case.to_input())


def test_layer2_on_runs_all_eight_and_builds_edges(tmp_path: Path) -> None:
    """With both flags on, the shared graph gets edges and all eight dimensions score."""
    result = _run(_layer2_on(), "missing_facet", tmp_path)
    assert set(result.results) == _DIMS
    assert result.graph is not None  # the one shared graph, now with detected edges


def test_whole_run_replays_with_layer2_on(tmp_path: Path) -> None:
    """The layered paths (edges, DAG rescue, tree) still replay from atoms — no model call."""
    result = _run(_layer2_on(), "aligned", tmp_path)
    verifier = ReplayVerifier(default_registry())
    assert verifier.verify_run(list(result.results.values()), result.store) is True
