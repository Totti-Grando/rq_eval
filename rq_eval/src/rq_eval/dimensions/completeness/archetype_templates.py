"""§2 — question-shape archetypes (the middle assurance mode).

A small, fixed, domain-independent set of ~8–12 question shapes, each with a
generic requirement pattern instantiated per question. The shapes are human-fixed
and finite even when topics aren't, so the model instantiates a fixed structure
rather than inventing facets — bounded structure against a scaffold. Loaded from
the pinned ``question_archetypes.yaml`` oracle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rq_eval.config import load_yaml
from rq_eval.dimensions.completeness.requirement_templates import Requirement

if TYPE_CHECKING:
    from rq_eval.config import Config


class ArchetypeTemplates:
    """Loads the archetype oracle and instantiates a shape's requirements."""

    def __init__(self, cfg: Config) -> None:
        """Load + validate the pinned question-archetype file."""
        data = load_yaml(cfg.resolve(cfg.paths.question_archetypes))
        if not isinstance(data, dict):
            raise ValueError("question_archetypes.yaml must be a mapping")
        self._version = str(data["version"])
        self._shapes: dict[str, Any] = data["archetypes"]

    @property
    def version(self) -> str:
        """The pinned archetype-set version."""
        return self._version

    def classify(self, question: str) -> str:
        """Return the archetype key (first keyword match, else 'default')."""
        q = question.lower()
        for key, spec in self._shapes.items():
            if key == "default":
                continue
            if any(kw.lower() in q for kw in spec.get("match", [])):
                return key
        return "default"

    def requirements_for(self, question: str) -> list[Requirement]:
        """Instantiate the matched archetype's generic requirement pattern."""
        spec = self._shapes[self.classify(question)]
        return [
            Requirement(id=r["id"], text=r["text"], vital=bool(r["vital"]))
            for r in spec["requirements"]
        ]
