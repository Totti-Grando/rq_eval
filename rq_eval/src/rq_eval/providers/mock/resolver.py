"""Mock resolver — deterministic offline reference existence (§2)."""

from __future__ import annotations

from rq_eval.providers.base import ResolverProvider

_FABRICATED = ("fabricated", "nonexistent", "doesnotexist", "fake", "madeup")


class MockResolverProvider(ResolverProvider):
    """Deterministic: a reference 'exists' unless it looks fabricated."""

    def resolve(self, reference: str) -> bool:
        """Return False iff the reference contains a fabricated-marker token."""
        low = reference.lower()
        return not any(marker in low for marker in _FABRICATED)
