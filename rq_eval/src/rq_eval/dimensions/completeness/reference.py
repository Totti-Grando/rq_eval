"""§2 — completeness reference-mode selector (generation is primary, stamped).

Completeness must score an *absence*, so its reference is generated — and this is
unavoidable, not a shortcut. The system supports three assurance modes and stamps
which one produced the score:

* **templated** (strongest, closed-domain) — a human per-type checklist
  (`requirement_templates.yaml`); coverage is a real guarantee.
* **archetype** (recommended middle) — the requirement *skeleton* instantiated
  from a fixed set of ~8–12 domain-independent question shapes.
* **generated** (primary, open-domain default) — requirement facets generated
  per-question `[T3-gen]`.

The Tier-2 bottom-up units stay extractive from source spans regardless of mode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rq_eval.dimensions.completeness.archetype_templates import ArchetypeTemplates
from rq_eval.dimensions.completeness.requirement_templates import Requirement, RequirementTemplates
from rq_eval.providers.base import GeneratorProvider

if TYPE_CHECKING:
    from rq_eval.config import Config


class ReferenceModeSelector:
    """Selects the Tier-1 requirement set per ``completeness.reference_mode``."""

    def __init__(
        self, cfg: Config, generator: GeneratorProvider, templates: RequirementTemplates
    ) -> None:
        """Inject config, the generator (generated mode), and the templated oracle."""
        self._mode = cfg.completeness.reference_mode
        self._generator = generator
        self._templates = templates
        self._archetypes = ArchetypeTemplates(cfg)
        self._seed = cfg.seeds.judge

    @property
    def mode(self) -> str:
        """The active assurance mode (stamped on the DimensionResult)."""
        return self._mode

    def requirements_for(self, question: str) -> list[Requirement]:
        """Return the Tier-1 requirements for ``question`` under the active mode."""
        if self._mode == "templated":
            return self._templates.requirements_for(question)
        if self._mode == "archetype":
            return self._archetypes.requirements_for(question)
        return self._generated(question)

    def _generated(self, question: str) -> list[Requirement]:
        """[T3-gen] Generate per-question requirement facets (open-domain default).

        Bounded, pinned generation: the generator proposes the facets a complete
        answer to *this* question must cover; they are frozen and τ-validated like
        any generated reference. Every generated facet is treated as vital.
        """
        result = self._generator.generate(
            f"[[sentences]] {{{{ requirement facets a complete answer must cover: {question} }}}}",
            seed=self._seed,
        )
        items = [t.strip() for t in (result.items or [question]) if t.strip()]
        return [
            Requirement(id=f"gen-{i}", text=text, vital=True) for i, text in enumerate(items)
        ]
