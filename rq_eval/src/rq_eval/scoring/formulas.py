"""Replay-critical composition formulas (§0.5.4; expanded in B5).

Each formula recomputes a score from atoms alone. Definitions:

* ``mean``                       — score = (Σ verdict) / n            (n atoms)
* ``weighted_mean``              — score = (Σ verdict·w) / (Σ w)
* ``conjunction_weighted_mean``  — group atoms by subject; per subject
                                   correct = AND(verdicts); w = the subject's
                                   (shared) weight; score = (Σ correct·w)/(Σ w)

All return 0.0 on an empty/zero-weight atom set. These are pure functions of the
atoms, so any score computed through them replays bit-for-bit.
"""

from __future__ import annotations

from rq_eval.contracts import AtomRecord
from rq_eval.scoring.registry import Formula, FormulaRegistry


class MeanFormula(Formula):
    """score = fraction of true verdicts (achieved/required, unit recall)."""

    formula_id = "mean"

    def compute(self, atoms: list[AtomRecord]) -> float:
        """Return (Σ verdict) / n, or 0.0 if there are no atoms."""
        if not atoms:
            return 0.0
        return sum(1 for a in atoms if a.verdict) / len(atoms)


class WeightedMeanFormula(Formula):
    """score = Σ verdict·weight / Σ weight."""

    formula_id = "weighted_mean"

    def compute(self, atoms: list[AtomRecord]) -> float:
        """Return the weight-normalized mean of verdicts, or 0.0 if Σw == 0."""
        total = sum(a.weight for a in atoms)
        if total == 0.0:
            return 0.0
        return sum(a.weight for a in atoms if a.verdict) / total


class ConjunctionWeightedMeanFormula(Formula):
    """Accuracy's formula: per-subject AND of verdicts, then weighted mean.

    correct(subject) = AND over that subject's atom verdicts;
    score = Σ correct·w / Σ w, weight taken per subject (first atom seen).
    """

    formula_id = "conjunction_weighted_mean"

    def compute(self, atoms: list[AtomRecord]) -> float:
        """Group by subject, AND verdicts, weight-average the results."""
        correct: dict[str, bool] = {}
        weight: dict[str, float] = {}
        for a in atoms:
            correct[a.subject] = correct.get(a.subject, True) and a.verdict
            weight.setdefault(a.subject, a.weight)
        total = sum(weight.values())
        if total == 0.0:
            return 0.0
        return sum(weight[s] for s, ok in correct.items() if ok) / total


def default_registry() -> FormulaRegistry:
    """Build a registry with the replay-critical formulas registered."""
    registry = FormulaRegistry()
    registry.register(MeanFormula())
    registry.register(WeightedMeanFormula())
    registry.register(ConjunctionWeightedMeanFormula())
    return registry
