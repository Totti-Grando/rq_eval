"""B3 — audit store + replay guarantee (§0.5.4).

Every grader call produces an AtomRecord; scores replay bit-for-bit from the
log with no model call; a tampered atom makes replay fail; both store backends
round-trip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.audit.atom_store import AtomStore
from rq_eval.audit.clock import FixedClock
from rq_eval.audit.jsonl_atom_store import JsonlAtomStore
from rq_eval.audit.replay import ReplayVerifier
from rq_eval.audit.sqlite_atom_store import SqliteAtomStore
from rq_eval.contracts import AtomRecord, DimensionResult
from rq_eval.scoring.formulas import default_registry


def _logger(store: AtomStore) -> AtomLogger:
    return AtomLogger(store, clock=FixedClock())


def _result(score: float, atom_ids: list[str], formula_id: str) -> DimensionResult:
    return DimensionResult(
        dimension="accuracy",
        score=score,
        band="A",
        ci_low=0.0,
        ci_high=1.0,
        n=len(atom_ids),
        inputs_hash="h",
        atom_ids=atom_ids,
        formula_id=formula_id,
    )


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_round_trip_and_replay(tmp_path: Path, backend: str) -> None:
    store: AtomStore = (
        JsonlAtomStore(tmp_path / "atoms.jsonl")
        if backend == "jsonl"
        else SqliteAtomStore(tmp_path / "atoms.db")
    )
    logger = _logger(store)
    # claim c1: grounded + responsive both true -> correct; c2: one false -> incorrect
    a1 = logger.record(subject="c1", role="grounded", question="q", tier="T2", verdict=True)
    a2 = logger.record(subject="c1", role="responsive", question="q", tier="T2", verdict=True)
    a3 = logger.record(subject="c2", role="grounded", question="q", tier="T2", verdict=True)
    a4 = logger.record(subject="c2", role="responsive", question="q", tier="T2", verdict=False)

    assert all(isinstance(a, AtomRecord) for a in (a1, a2, a3, a4))
    assert len(store.all()) == 4

    registry = default_registry()
    atom_ids = [a1.id, a2.id, a3.id, a4.id]
    score = registry.compute("conjunction_weighted_mean", store.by_ids(atom_ids))
    assert score == pytest.approx(0.5)  # 1 of 2 subjects correct

    result = _result(score, atom_ids, "conjunction_weighted_mean")
    verifier = ReplayVerifier(registry)
    assert verifier.verify(result, store) is True
    assert verifier.recompute(result, store) == result.score  # bit-for-bit


def test_tampered_atom_fails_replay(tmp_path: Path) -> None:
    path = tmp_path / "atoms.jsonl"
    store = JsonlAtomStore(path)
    logger = _logger(store)
    a1 = logger.record(subject="c1", role="grounded", question="q", tier="T2", verdict=True)
    a2 = logger.record(subject="c2", role="grounded", question="q", tier="T2", verdict=True)

    registry = default_registry()
    ids = [a1.id, a2.id]
    result = _result(registry.compute("mean", store.by_ids(ids)), ids, "mean")
    assert result.score == pytest.approx(1.0)

    # tamper: flip a stored verdict on disk (keeps the same id -> resolved by id)
    tampered = path.read_text(encoding="utf-8").replace('"verdict":true', '"verdict":false', 1)
    path.write_text(tampered, encoding="utf-8")

    verifier = ReplayVerifier(registry)
    assert verifier.verify(result, store) is False  # recomputed 0.5 != stored 1.0


def test_weighted_mean_and_mean_formulas(tmp_path: Path) -> None:
    store = JsonlAtomStore(tmp_path / "a.jsonl")
    logger = _logger(store)
    hi = logger.record(
        subject="v1", role="unit", question="q", tier="T2", verdict=True, weight=3.0
    )
    lo = logger.record(
        subject="v2", role="unit", question="q", tier="T2", verdict=False, weight=1.0
    )
    registry = default_registry()
    atoms = store.by_ids([hi.id, lo.id])
    assert registry.compute("mean", atoms) == pytest.approx(0.5)
    assert registry.compute("weighted_mean", atoms) == pytest.approx(0.75)


def test_duplicate_formula_id_rejected() -> None:
    reg = default_registry()
    with pytest.raises(ValueError, match="duplicate"):
        from rq_eval.scoring.formulas import MeanFormula

        reg.register(MeanFormula())
