"""R6 — determinism ledger: lock the score-affecting [T3] judge surface.

After the reforms the only score-affecting generative-judge calls are the five
named residuals; the reference-construction pipeline (`pipeline.*`, [T3-gen]) is
excluded, and no reformed path (on-ask, admissibility non-residual, disinterest
rule, impossible detection) may log a judge atom. This test fails if a new judge
silently creeps onto the scoring path.
"""

from __future__ import annotations

from pathlib import Path

from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.config import load_config
from rq_eval.fixtures import FixtureSuite
from rq_eval.runner import Evaluator

# the five reform residuals (abstention = decline + unanswerable mechanism)
_ALLOWED_T3 = {
    "accuracy.residual",                    # unsourced residual
    "task_success.adequacy",                # adequacy
    "relevance.decline",                    # \_ abstention
    "relevance.unanswerable",               # /
    "completeness.decidability_residual",   # decidability residual (disagreement only)
    "source_quality.disinterest_residual",  # disinterest residual (ambiguous + sampled)
}
# grader_ids that the reforms removed entirely (must not appear anywhere)
_REMOVED = {
    "relevance.on_ask", "relevance.residual",
    "completeness.decidable", "source_quality.disinterest",
}


def test_t3_scoring_surface_is_locked(tmp_path: Path) -> None:
    cfg = load_config()
    observed_t3: set[str] = set()
    all_graders: set[str] = set()
    for i, case in enumerate(FixtureSuite().cases()):
        store = JsonlAtomStore(tmp_path / f"c{i}.jsonl")
        Evaluator(cfg, store=store, clock=FixedClock()).evaluate(case.to_input())
        for atom in store.all():
            all_graders.add(atom.grader_id)
            if atom.tier == "T3" and not atom.grader_id.startswith("pipeline."):
                observed_t3.add(atom.grader_id)

    assert observed_t3 <= _ALLOWED_T3, f"new score-affecting judge(s): {observed_t3 - _ALLOWED_T3}"
    leaked = all_graders & _REMOVED
    assert not leaked, f"removed judge grader_ids present: {leaked}"


def test_reformed_paths_emit_no_judge_atoms(tmp_path: Path) -> None:
    """on-ask, impossible, and task-type classification are now T1/T2 (no judge)."""
    cfg = load_config()
    case = next(c for c in FixtureSuite().cases() if c.name == "contradiction")
    store = JsonlAtomStore(tmp_path / "a.jsonl")
    Evaluator(cfg, store=store, clock=FixedClock()).evaluate(case.to_input())
    by_role = {a.role: a.tier for a in store.all()}
    assert by_role.get("on_ask_nli") == "T2"
    assert by_role.get("on_ask_lex") == "T1"
    assert by_role.get("impossible_success") == "T1"
    assert by_role.get("task_type") == "T1"


def test_extraction_and_relevance_are_deterministic(tmp_path: Path) -> None:
    """U1/U3: claim decomposition + relevance scoring emit no judge on their paths.

    The pipeline (extraction) now logs only T1 atoms — the old pipeline.verifiable/
    disambiguate/decontextualized judge atoms are gone — and the relevance
    anchor-tree scoring path is fixed NLI + code (on-ask, tree grade, routing).
    """
    cfg = load_config()
    for i, case in enumerate(FixtureSuite().cases()):
        store = JsonlAtomStore(tmp_path / f"d{i}.jsonl")
        Evaluator(cfg, store=store, clock=FixedClock()).evaluate(case.to_input())
        atoms = store.all()
        # no pipeline.* extraction atom is a judge (T3)
        pipeline_t3 = [a for a in atoms if a.grader_id.startswith("pipeline.") and a.tier == "T3"]
        assert not pipeline_t3, f"pipeline judge atom leaked: {[a.role for a in pipeline_t3]}"
        # the reformed relevance scoring roles are fixed NLI + code, never a judge
        by_role = {a.role: a.tier for a in atoms}
        assert by_role.get("verifiable") == "T1"  # was a judge before U1
        if "claim_relevance" in by_role:
            assert by_role["claim_relevance"] == "T2"  # depth-graded, no judge (U3)
        if "routed_contradiction" in by_role:
            assert by_role["routed_contradiction"] == "T1"
