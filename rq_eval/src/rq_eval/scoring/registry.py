"""Formula registry — the replay backbone (§0.5.4).

A :class:`Formula` recomputes a score purely from a list of :class:`AtomRecord`s
(their verdicts, weights, and subjects). Dimensions compute their score *through*
a registered formula and store its ``formula_id``; the replay verifier then
recomputes the same score from the logged atoms with no model call. Keeping the
formula id + the atoms is therefore sufficient to reproduce any score.

No model/provider code is imported here (enforced by
``tests/test_scoring_pure.py`` in B5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rq_eval.contracts import AtomRecord


class Formula(ABC):
    """A pure score function over atoms; identified by :attr:`formula_id`."""

    formula_id: str

    @abstractmethod
    def compute(self, atoms: list[AtomRecord]) -> float:
        """Recompute the score from ``atoms`` only. Deterministic."""


class FormulaRegistry:
    """Maps ``formula_id`` -> :class:`Formula`; the single lookup for replay."""

    def __init__(self) -> None:
        """Start empty; register formulas via :meth:`register`."""
        self._formulas: dict[str, Formula] = {}

    def register(self, formula: Formula) -> Formula:
        """Register ``formula`` under its id (rejects duplicate ids)."""
        if formula.formula_id in self._formulas:
            raise ValueError(f"duplicate formula_id: {formula.formula_id}")
        self._formulas[formula.formula_id] = formula
        return formula

    def get(self, formula_id: str) -> Formula:
        """Return the formula for ``formula_id`` (KeyError if unknown)."""
        if formula_id not in self._formulas:
            raise KeyError(f"unknown formula_id: {formula_id}")
        return self._formulas[formula_id]

    def compute(self, formula_id: str, atoms: list[AtomRecord]) -> float:
        """Recompute a score: ``get(formula_id).compute(atoms)``."""
        return self.get(formula_id).compute(atoms)
