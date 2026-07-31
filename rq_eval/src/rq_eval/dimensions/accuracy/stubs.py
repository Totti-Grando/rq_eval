"""Interface stub for the one category accuracy still imports as a placeholder.

``source_quality`` is now the real provider (Evidence & Truthfulness §3). The
shared ``inference-validity`` check (``logical_consistency``) remains a separate
category, out of scope here, so it stays a typed stub (inference assumed valid),
wired so the real module drops in later without touching :class:`AccuracyDimension`.
"""

from __future__ import annotations


class InferenceValidityStub:
    """Stub for the shared inference-validity check (logical_consistency)."""

    def valid(self, claim: str, context: str) -> bool:
        """Return whether an inferred claim's inference is valid (stub: True)."""
        return True
