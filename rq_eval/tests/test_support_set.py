"""G1 — the Evidence support-set model: one S, derived by §3/§4 (no new NLI)."""

from __future__ import annotations

from pathlib import Path

from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.config import load_config
from rq_eval.fixtures import FixtureSuite
from rq_eval.runner import Evaluator


def _run(case_name: str, tmp_path: Path):
    cfg = load_config()
    case = next(c for c in FixtureSuite().cases() if c.name == case_name)
    store = JsonlAtomStore(tmp_path / f"{case_name}.jsonl")
    return Evaluator(cfg, store=store, clock=FixedClock()).evaluate(case.to_input()), store


def test_groundedness_builds_support_set_in_atoms(tmp_path: Path) -> None:
    """The per-chunk pass logs each triplet's support set S (chunk-ids)."""
    _result, store = _run("aligned", tmp_path)
    triplet_atoms = [a for a in store.all() if a.role == "triplet_grounded"]
    assert triplet_atoms
    assert all("S=" in a.evidence for a in triplet_atoms)  # support set logged per triplet
    assert all(a.tier == "T2" for a in triplet_atoms)


def test_attributed_subset_of_grounded(tmp_path: Path) -> None:
    """C∩S≠∅ ⟹ S≠∅: every attributed claim is necessarily grounded (built-in check)."""
    result, store = _run("wrong_citation", tmp_path)
    # the mis-cited fixture: attribution finds the cited chunk not in S -> precision 0
    assert result.results["source_attribution"].score == 0.0
    # and a mis-citation diagnostic (C−S) is reported
    assert result.results["source_attribution"].extra["mis_cited"] >= 1.0


def test_source_quality_and_attribution_make_no_new_nli(tmp_path: Path) -> None:
    """§3/§4 derive from S: their atoms are set-ops (T1), not fresh entailments (T2 NLI)."""
    _result, store = _run("bad_source", tmp_path)
    sq = [a for a in store.all() if a.grader_id == "source_quality.property"]
    assert sq and all(a.tier in ("T1", "T3") for a in sq)  # supports/corroborated now T1
    cite = [a for a in store.all() if a.grader_id == "source_attribution.cite"]
    assert all(a.tier == "T1" for a in cite)  # set-op over S, not an NLI pass
