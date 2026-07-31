"""§3 step 4 — domain reliability allow/deny oracle.

Human-maintained, pinned YAML (structural oracle). deny → unreliable; allow →
reliable; neither (with a non-empty allow-list) → not-yet-vetted (unreliable);
internal-corpus chunks (no domain) → reliable by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rq_eval.config import load_yaml

if TYPE_CHECKING:
    from rq_eval.config import Config


class ReliabilityList:
    """Loads + queries the pinned domain reliability oracle."""

    def __init__(self, cfg: Config) -> None:
        """Load + validate the reliability YAML."""
        data = load_yaml(cfg.resolve(cfg.source_quality.reliability_list))
        if not isinstance(data, dict):
            raise ValueError("reliability_list.yaml must be a mapping")
        self._version = str(data["version"])
        self._allow = {str(d).lower() for d in data.get("allow", [])}
        self._deny = {str(d).lower() for d in data.get("deny", [])}

    @property
    def version(self) -> str:
        """The pinned reliability-list version."""
        return self._version

    def is_reliable(self, domain: str | None) -> bool:
        """Return whether ``domain`` is reputable (None == internal → True)."""
        if domain is None:
            return True
        d = domain.lower()
        if d in self._deny:
            return False
        if d in self._allow:
            return True
        return not self._allow  # unknown domain: reliable only if no allow-list is set
