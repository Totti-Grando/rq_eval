"""Live resolver — URL reachability + optional DOI registry (§2, live path).

urllib is used for URLs; DOI validation against a registry is config-gated
(``hallucination.doi_registry_enabled``). Imports are lazy/stdlib; network calls
only fire on the target machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rq_eval.providers.base import ResolverProvider

if TYPE_CHECKING:
    from rq_eval.config import Config

_DOI_REGISTRY = "https://doi.org/"


class LiveResolverProvider(ResolverProvider):
    """Resolves URLs (urllib HEAD) and, if enabled, DOIs via the DOI registry."""

    def __init__(self, cfg: Config) -> None:
        """Store config (DOI toggle)."""
        self._cfg = cfg

    def resolve(self, reference: str) -> bool:
        """True iff the URL resolves (or the DOI validates when enabled)."""
        ref = reference.strip()
        if ref.startswith("10.") or ref.lower().startswith("doi:"):
            if not self._cfg.hallucination.doi_registry_enabled:
                return True  # not in scope -> don't gate on it
            doi = ref.split(":", 1)[1] if ref.lower().startswith("doi:") else ref
            return self._head(_DOI_REGISTRY + doi)
        if ref.startswith(("http://", "https://")):
            return self._head(ref)
        return True  # non-URL/DOI references are checked by set-membership in code

    @staticmethod
    def _head(url: str) -> bool:
        import urllib.error  # noqa: PLC0415 - stdlib, lazy
        import urllib.request  # noqa: PLC0415

        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - target-only
                return 200 <= int(resp.status) < 400
        except (urllib.error.URLError, ValueError, OSError):
            return False
