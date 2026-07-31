"""§2 step 6 — assign units to the answer [T2 NLI].

Per unit, one binary: is it *fully* supported by the answer (answer = premise,
unit = hypothesis; partial = unsupported). Uses the grounding grader; the
per-unit support atom is weighted by the unit's materiality.
"""

from __future__ import annotations

from rq_eval.contracts import AtomRecord
from rq_eval.dimensions.completeness.unit import Unit
from rq_eval.graders.grounding_grader import GroundingGrader


class UnitAssigner:
    """[T2] Decides, per unit, whether the answer supports it."""

    def __init__(self, grounding: GroundingGrader, vital_weight: float, okay_weight: float) -> None:
        """Inject the grounding grader and the vital/okay unit weights."""
        self._grounding = grounding
        self._vital_weight = vital_weight
        self._okay_weight = okay_weight

    def assign(self, units: list[Unit], answer: str) -> list[AtomRecord]:
        """Return one support atom per unit (verdict = fully supported)."""
        atoms: list[AtomRecord] = []
        for unit in units:
            weight = self._vital_weight if unit.vital else self._okay_weight
            atoms.append(
                self._grounding.check(
                    subject=unit.id, role="unit_support", source=answer, claim=unit.text,
                    weight=weight,
                )
            )
        return atoms
