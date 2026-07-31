"""§2 step 1 — Tier-1 requirement templates (the structural oracle).

A fixed, human-reviewed scaffold: the facets a complete answer must cover, per
question-type. This is where the coverage guarantee lives — not an AI recall
step. The question-type is selected by config-defined ``match`` keywords.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rq_eval.config import load_yaml

if TYPE_CHECKING:
    from rq_eval.config import Config


@dataclass(frozen=True, slots=True)
class Requirement:
    """One Tier-1 facet: id, human text, and materiality (vital/okay)."""

    id: str
    text: str
    vital: bool


class RequirementTemplates:
    """Loads the versioned YAML oracle and selects requirements by question."""

    def __init__(self, cfg: Config) -> None:
        """Load + validate the pinned requirement-template file."""
        data = load_yaml(cfg.resolve(cfg.paths.requirement_templates))
        if not isinstance(data, dict):
            raise ValueError("requirement_templates.yaml must be a mapping")
        self._version = str(data["version"])
        self._types: dict[str, Any] = data["question_types"]

    @property
    def version(self) -> str:
        """The pinned template-set version."""
        return self._version

    def classify(self, question: str) -> str:
        """Return the question-type key (first keyword match, else 'default')."""
        q = question.lower()
        for key, spec in self._types.items():
            if key == "default":
                continue
            if any(kw.lower() in q for kw in spec.get("match", [])):
                return key
        return "default"

    def requirements_for(self, question: str) -> list[Requirement]:
        """Return the Tier-1 requirements for this question's type."""
        spec = self._types[self.classify(question)]
        return [
            Requirement(id=r["id"], text=r["text"], vital=bool(r["vital"]))
            for r in spec["requirements"]
        ]
