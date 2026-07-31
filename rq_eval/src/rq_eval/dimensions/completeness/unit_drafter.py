"""§2 step 2 — Tier-2 unit drafting [T3-gen].

Within each requirement, draft atomic units top-down (from the requirement) and
bottom-up (from the sources — catching drop-from-source omissions). Each unit is
one checkable nugget; units inherit their requirement's vitality (the oracle).
"""

from __future__ import annotations

from rq_eval.dimensions.completeness.requirement_templates import Requirement
from rq_eval.dimensions.completeness.unit import Unit
from rq_eval.providers.base import GeneratorProvider, NlpProvider
from rq_eval.providers.mock.deterministic_text import DeterministicText


class UnitDrafter:
    """[T3-gen] Drafts candidate units per requirement (top-down + bottom-up)."""

    def __init__(self, generator: GeneratorProvider, nlp: NlpProvider, seed: int) -> None:
        """Inject the generator + NLP; seed the bottom-up relevance filter."""
        self._generator = generator
        self._nlp = nlp
        self._seed = seed
        self._dt = DeterministicText(seed)

    def draft(self, requirement: Requirement, sources: str) -> list[Unit]:
        """Return top-down + bottom-up candidate units for ``requirement``."""
        units: list[Unit] = []
        # top-down: the requirement phrased as a checkable statement
        top = self._generator.generate(
            f"[[sentences]] {{{{ {requirement.text} }}}}", seed=self._seed
        )
        for text in top.items or [requirement.text]:
            units.append(Unit.create(text, requirement.id, requirement.vital, "top_down"))
        # bottom-up: source sentences relevant to the requirement (drop-from-source)
        for sentence in self._nlp.segment(sources):
            if self._dt.overlap(requirement.text, sentence) > 0.0:
                units.append(Unit.create(sentence, requirement.id, requirement.vital, "bottom_up"))
        return units
