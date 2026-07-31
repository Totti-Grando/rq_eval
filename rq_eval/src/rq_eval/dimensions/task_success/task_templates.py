"""§4 step 2 — task-type taxonomy + outcome templates (pinned reference).

A fixed, versioned taxonomy: each task type and the required outcomes it implies.
The question is classified by config-defined ``match`` keywords (deterministic);
the genuine Tier-3 judgment is left to the per-outcome verdicts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rq_eval.config import load_yaml

if TYPE_CHECKING:
    from rq_eval.config import Config


@dataclass(frozen=True, slots=True)
class Outcome:
    """One required outcome, tagged with the verifier that decides it (§4 v2)."""

    id: str
    text: str
    # verifier tag: artifact_presence|executable|state|constraint|coverage|import|adequacy
    verifier: str
    weight: float
    params: Mapping[str, Any]


class TaskTemplates:
    """Loads the versioned task taxonomy and classifies questions."""

    def __init__(self, cfg: Config) -> None:
        """Load + validate the pinned task-template file."""
        data = load_yaml(cfg.resolve(cfg.paths.task_templates))
        if not isinstance(data, dict):
            raise ValueError("task_templates.yaml must be a mapping")
        self._version = str(data["version"])
        self._default = str(data["default"])
        self._types: dict[str, Any] = data["task_types"]

    @property
    def version(self) -> str:
        """The pinned taxonomy version."""
        return self._version

    def classify(self, question: str) -> str:
        """Return the task-type key (first keyword match, else the default)."""
        q = question.lower()
        for key, spec in self._types.items():
            if any(kw.lower() in q for kw in spec.get("match", [])):
                return key
        return self._default

    def outcomes_for(self, task_type: str) -> list[Outcome]:
        """Return the verifier-tagged required outcomes for ``task_type``."""
        return [
            Outcome(
                id=o["id"], text=o["text"], verifier=o["verifier"],
                weight=float(o.get("weight", 1.0)), params=dict(o.get("params", {})),
            )
            for o in self._types[task_type]["outcomes"]
        ]
