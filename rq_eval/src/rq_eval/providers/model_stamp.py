"""Model+version stamps for atoms (§0.5.2, drift detection).

Maps the configured provider mode/backends to the ``(model, version)`` pair
stamped on every :class:`AtomRecord`, so a verdict records *what decided it* and
drift in a live model is detectable rather than silent. All ids are read from
the typed config object — no literals here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rq_eval.config import Config


class ModelStamp:
    """Returns the ``(model, version)`` to stamp per provider kind."""

    def __init__(self, cfg: Config) -> None:
        """Store config; stamps are derived from mode + model ids."""
        self._cfg = cfg
        self._live = cfg.providers.mode == "live"

    def judge(self) -> tuple[str, str]:
        """Judge model stamp."""
        return (self._cfg.models.judge_id, "live") if self._live else ("mock-judge", "mock")

    def generator(self) -> tuple[str, str]:
        """Generator model stamp (same base model as the judge, live)."""
        return (self._cfg.models.judge_id, "live") if self._live else ("mock-generator", "mock")

    def embedding(self) -> tuple[str, str]:
        """Embedding model stamp."""
        return (self._cfg.models.embed_id, "live") if self._live else ("mock-embed", "mock")

    def grounding(self) -> tuple[str, str]:
        """Grounding backend stamp (guardrail / fairseq / mock)."""
        if not self._live or self._cfg.models.nli == "mock":
            return ("mock-grounding", "mock")
        if self._cfg.models.nli == "fairseq":
            return ("fairseq-roberta-mnli", "live")
        return (self._cfg.models.guardrail_id, self._cfg.models.guardrail_version)

    def relevance(self) -> tuple[str, str]:
        """Relevance backend stamp (guardrail / mock)."""
        if not self._live:
            return ("mock-relevance", "mock")
        return (self._cfg.models.guardrail_id, self._cfg.models.guardrail_version)
