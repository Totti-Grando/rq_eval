"""§2 step 3 — unit admissibility gate [T1 + T2 + T3-once].

A unit joins the frozen set only if it is atomic (parse/conjunction-split [T1]),
self-contained (coref-resolved [T2+T1]), and entailment-decidable from the
answer text alone (one-time admission [T3]). Non-atomic units are repaired by
splitting; the rest are rejected. This guards against evaluator over-acceptance.
"""

from __future__ import annotations

import re

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.dimensions.completeness.unit import Unit
from rq_eval.graders.judge_grader import JudgeGrader
from rq_eval.graders.t1 import T1Tools
from rq_eval.providers.base import NlpProvider

_LEADING_PRONOUN = re.compile(r"^(he|she|it|they|this|that|these|those)\b", re.IGNORECASE)
_DECIDABLE = (
    "[[verifiable]] Is this unit decidable from the answer text alone (no world knowledge)?"
)


class UnitAdmissibilityGate:
    """[T1/T2/T3] Repairs, checks, and freezes the admissible unit set."""

    def __init__(
        self, t1: T1Tools, nlp: NlpProvider, decidable: JudgeGrader, logger: AtomLogger
    ) -> None:
        """Inject T1 tools, NLP (coref), the decidability judge, and logger."""
        self._t1 = t1
        self._nlp = nlp
        self._decidable = decidable
        self._logger = logger

    def admit(self, units: list[Unit], context: str) -> list[Unit]:
        """Return the frozen, admissible unit set (repairing non-atomic units)."""
        frozen: list[Unit] = []
        for unit in units:
            for text in self._atomic_parts(unit.text):
                resolved = self._nlp.resolve_coref(text, context).resolved_text
                self_contained = _LEADING_PRONOUN.match(resolved) is None
                decidable = self._decidable.judge(
                    subject=unit.id, role="decidable", question=_DECIDABLE, context=resolved
                ).verdict
                admitted = self_contained and decidable
                self._logger.record(
                    subject=unit.id, role="admissible", question="admissibility gate", tier="T2",
                    verdict=admitted,
                    evidence=f"self_contained={self_contained} decidable={decidable}",
                    grader_id="completeness.admissibility", model="code", model_version="rq_eval",
                )
                if admitted:
                    frozen.append(
                        Unit.create(resolved, unit.requirement_id, unit.vital, unit.origin)
                    )
        return frozen

    def _atomic_parts(self, text: str) -> list[str]:
        """Return atomic parts: the text itself, or its conjunction-split repair."""
        return [text] if self._t1.is_atomic(text) else self._t1.conjunction_split(text)
