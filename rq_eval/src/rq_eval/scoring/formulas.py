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


class RelevanceCappedMeanFormula(Formula):
    """§3 relevance: mean of responsive atoms, off-ask capped, abstain-aware.

    Reads three atom roles so it replays purely from the logged atoms:
    * ``abstain_relevant`` (verdict True) -> score 1.0 (proper decline);
    * ``responsive`` atoms -> ``base = mean(verdicts)``;
    * ``on_ask_answer`` atom -> if its verdict is False, cap: ``min(base, w)``
      where the cap value travels as that atom's ``weight``.
    """

    formula_id = "relevance_capped_mean"

    def compute(self, atoms: list[AtomRecord]) -> float:
        """Compute the capped, abstain-aware mean from the atoms alone."""
        if any(a.role == "abstain_relevant" and a.verdict for a in atoms):
            return 1.0
        responsive = [a for a in atoms if a.role == "responsive"]
        base = (sum(1 for a in responsive if a.verdict) / len(responsive)) if responsive else 0.0
        on_ask = [a for a in atoms if a.role == "on_ask_answer"]
        if on_ask and not on_ask[0].verdict:
            return min(base, on_ask[0].weight)
        return base


class AchievedRatioFormula(Formula):
    """§4 task_success: ``|achieved| / |required outcomes|``, impossible-aware.

    * an ``impossible_success`` atom (verdict True) -> 1.0 (a well-scoped "can't
      be done because X" is a success, like relevance's abstention);
    * otherwise ``mean`` over the ``outcome`` atom verdicts.
    Replays purely from atoms.
    """

    formula_id = "achieved_ratio"

    def compute(self, atoms: list[AtomRecord]) -> float:
        """Compute achieved/required (or 1.0 for a well-scoped impossibility)."""
        if any(a.role == "impossible_success" and a.verdict for a in atoms):
            return 1.0
        outcomes = [a for a in atoms if a.role == "outcome"]
        if not outcomes:
            return 0.0
        return sum(1 for a in outcomes if a.verdict) / len(outcomes)


class TaskSuccessWeightedFormula(Formula):
    """§4 task_success (v2): ``Σ achieved·w / Σ w`` over outcome atoms.

    * an ``impossible_success`` atom (verdict True) -> 1.0;
    * otherwise the weighted mean of the ``outcome`` atom verdicts, where each
      outcome's weight is its atom weight. Replays purely from atoms.
    """

    formula_id = "task_success_weighted"

    def compute(self, atoms: list[AtomRecord]) -> float:
        """Weighted achieved/required (or 1.0 for a well-scoped impossibility)."""
        if any(a.role == "impossible_success" and a.verdict for a in atoms):
            return 1.0
        outcomes = [a for a in atoms if a.role == "outcome"]
        total = sum(a.weight for a in outcomes)
        if total == 0.0:
            return 0.0
        return sum(a.weight for a in outcomes if a.verdict) / total


class UnsupportedRateFormula(Formula):
    """§2 hallucination: ``unsupported = 1 − |supported| / |total|`` over triplet atoms."""

    formula_id = "unsupported_rate"

    def compute(self, atoms: list[AtomRecord]) -> float:
        """Return 1 − mean(verdicts), or 0.0 if there are no atoms."""
        if not atoms:
            return 0.0
        return 1.0 - sum(1 for a in atoms if a.verdict) / len(atoms)


def default_registry() -> FormulaRegistry:
    """Build a registry with the replay-critical formulas registered."""
    registry = FormulaRegistry()
    registry.register(MeanFormula())
    registry.register(WeightedMeanFormula())
    registry.register(ConjunctionWeightedMeanFormula())
    registry.register(RelevanceCappedMeanFormula())
    registry.register(AchievedRatioFormula())
    registry.register(TaskSuccessWeightedFormula())
    registry.register(UnsupportedRateFormula())
    return registry
