"""§2 step 7 — two-level completeness scoring [code].

Pure composition over the per-unit support atoms:
* per-requirement recall = supported units / that requirement's units
  (normalized per requirement so a big facet doesn't drown a small one);
* requirement coverage = requirements with ≥1 supported unit / total;
* weighted recall = vital-weighted mean of per-requirement recall.

The headline score (strict vital recall) is computed separately via the `mean`
formula over the vital support atoms, so it replays from atoms.
"""

from __future__ import annotations

from rq_eval.contracts import AtomRecord
from rq_eval.dimensions.completeness.requirement_templates import Requirement
from rq_eval.dimensions.completeness.unit import Unit


class TwoLevelScoring:
    """Computes requirement-level completeness diagnostics (pure)."""

    def requirement_coverage(
        self, units: list[Unit], atoms: list[AtomRecord], requirements: list[Requirement]
    ) -> float:
        """Fraction of requirements with at least one supported unit."""
        if not requirements:
            return 0.0
        supported = self._supported_requirement_ids(units, atoms)
        return sum(1 for r in requirements if r.id in supported) / len(requirements)

    def weighted_recall(
        self,
        units: list[Unit],
        atoms: list[AtomRecord],
        requirements: list[Requirement],
        vital_weighting: bool,
    ) -> float:
        """Vital-weighted mean of per-requirement recall (0 if no requirements)."""
        total_w = 0.0
        acc = 0.0
        for req in requirements:
            recall = self._requirement_recall(req.id, units, atoms)
            weight = 2.0 if (vital_weighting and req.vital) else 1.0
            acc += recall * weight
            total_w += weight
        return acc / total_w if total_w else 0.0

    def _requirement_recall(
        self, req_id: str, units: list[Unit], atoms: list[AtomRecord]
    ) -> float:
        verdict = {a.subject: a.verdict for a in atoms}
        req_units = [u for u in units if u.requirement_id == req_id]
        if not req_units:
            return 0.0
        return sum(1 for u in req_units if verdict.get(u.id, False)) / len(req_units)

    @staticmethod
    def _supported_requirement_ids(units: list[Unit], atoms: list[AtomRecord]) -> set[str]:
        verdict = {a.subject: a.verdict for a in atoms}
        return {u.requirement_id for u in units if verdict.get(u.id, False)}
