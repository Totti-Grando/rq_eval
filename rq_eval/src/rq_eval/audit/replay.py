"""Replay verifier — the determinism guarantee made checkable (§0.5.4).

Recomputes a :class:`DimensionResult`'s score from its logged ``atom_ids`` +
``formula_id`` using the formula registry, with **no model call**, and asserts
equality. T1/T2/code atoms replay bit-for-bit; a T3 atom replays from its logged
verdict (its model+version is stamped so drift is detectable, not silent). If
recomputation doesn't reproduce the stored score, that is a defect — and a
tampered atom is caught the same way.
"""

from __future__ import annotations

from rq_eval.audit.atom_store import AtomStore
from rq_eval.contracts import DimensionResult
from rq_eval.scoring.registry import FormulaRegistry


class ReplayVerifier:
    """Recomputes stored scores from atoms to prove they are reproducible."""

    def __init__(self, registry: FormulaRegistry) -> None:
        """Inject the formula registry used to recompute scores."""
        self._registry = registry

    def recompute(self, result: DimensionResult, store: AtomStore) -> float:
        """Recompute ``result.score`` from its atoms + formula (no model call)."""
        atoms = store.by_ids(result.atom_ids)
        return self._registry.compute(result.formula_id, atoms)

    def verify(self, result: DimensionResult, store: AtomStore) -> bool:
        """True iff the recomputed score equals the stored score exactly."""
        return self.recompute(result, store) == result.score

    def verify_run(self, results: list[DimensionResult], store: AtomStore) -> bool:
        """True iff every result in a run replays."""
        return all(self.verify(r, store) for r in results)
