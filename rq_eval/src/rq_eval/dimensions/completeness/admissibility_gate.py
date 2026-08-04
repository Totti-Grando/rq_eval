"""§2 step 3 — unit admissibility gate (deterministic-first, R3).

A unit joins the frozen set only if it is:
* **atomic** `[T1]` — one predicate / one 5W1H slot (structural; conjunction-split
  repairs multi-predicate units) — the Warrant-Gap typed check, no judge;
* **self-contained** `[T1]` — no unresolved mention after coref (Molecular Facts /
  FactCoref), no judge;
* **entailment-decidable** `[T2]` — **double-NLI agreement**: run the grounding
  verifier on the unit with premise = answer vs premise = answer+corpus; agreement
  ⟹ decidable from the answer alone (world-knowledge units flip and route out).
  Only genuine disagreements fall to a reference-grounded `ScoringJudge` residual.
"""

from __future__ import annotations

import re

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.dimensions.completeness.unit import Unit
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.graders.judge_grader import JudgeGrader
from rq_eval.graders.t1 import T1Tools
from rq_eval.providers.base import NlpProvider

_LEADING_PRONOUN = re.compile(r"^(he|she|it|they|this|that|these|those)\b", re.IGNORECASE)
# mock: [[deny]] -> disagreements default to "not decidable" (world-knowledge units
# route out); live strips the tag and the judge decides against the reference.
_RESIDUAL = "[[deny]] Is this unit decidable from the answer text alone (no world knowledge)?"


class UnitAdmissibilityGate:
    """[T1/T2] Repairs, checks (double-NLI), and freezes the admissible unit set."""

    def __init__(
        self,
        t1: T1Tools,
        nlp: NlpProvider,
        grounding: GroundingGrader,
        residual: JudgeGrader,
        logger: AtomLogger,
        double_nli: bool = True,
    ) -> None:
        """Inject T1 tools, NLP (coref), the double-NLI grounding grader + residual."""
        self._t1 = t1
        self._nlp = nlp
        self._grounding = grounding
        self._residual = residual
        self._logger = logger
        self._double_nli = double_nli

    def admit(self, units: list[Unit], answer: str, sources: str) -> list[Unit]:
        """Return the frozen admissible unit set (atomic + self-contained + decidable)."""
        frozen: list[Unit] = []
        for unit in units:
            for text in self._atomic_parts(unit.text):  # atomic [T1] (split repair)
                resolved = self._nlp.resolve_coref(text, sources).resolved_text
                self_contained = _LEADING_PRONOUN.match(resolved) is None  # [T1]
                decidable, via = self._decidable(unit, resolved, answer, sources)
                admitted = self_contained and decidable
                self._logger.record(
                    subject=unit.id, role="admissible", question="admissibility gate", tier="T2",
                    verdict=admitted,
                    evidence=f"self_contained={self_contained} decidable={decidable} via={via}",
                    grader_id="completeness.admissibility", model="code", model_version="rq_eval",
                )
                if admitted:
                    frozen.append(
                        Unit.create(resolved, unit.requirement_id, unit.vital, unit.origin)
                    )
        return frozen

    def _decidable(self, unit: Unit, text: str, answer: str, sources: str) -> tuple[bool, str]:
        """Double-NLI: agree ⟹ decidable; disagreement ⟹ reference-grounded residual."""
        label_answer = self._grounding.classify(premise=answer, hypothesis=text).label
        if not self._double_nli:  # single-NLI mode: decidable ⟺ answer entails the unit
            return label_answer == "E", "single_nli"
        label_corpus = self._grounding.classify(
            premise=f"{answer} {sources}".strip(), hypothesis=text
        ).label
        if label_answer == label_corpus:
            return True, "double_nli"
        residual = self._residual.judge(  # only genuine disagreements reach the judge
            subject=unit.id, role="decidability_residual", question=_RESIDUAL,
            context=text, reference=sources, tier="T3",
        )
        return residual.verdict, "residual"

    def _atomic_parts(self, text: str) -> list[str]:
        """Return atomic parts: the text itself, or its conjunction-split repair."""
        return [text] if self._t1.is_atomic(text) else self._t1.conjunction_split(text)
