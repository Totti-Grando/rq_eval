"""Interface stubs for categories accuracy imports but that live elsewhere.

``source_quality`` (source-adequate?) and the shared ``inference-validity`` check
are other evaluation categories, out of scope for this build. The design imports
them; here they are typed interface stubs with the Nexa-profile default
(trusted corpus → source-adequacy ≈ 1, inference assumed valid), wired so the
real modules drop in later without touching :class:`AccuracyDimension`.
"""

from __future__ import annotations

from rq_eval.contracts import Profile


class SourceQualityStub:
    """Stub for ``source_quality`` — is the grounding source itself adequate?"""

    def __init__(self, profile: Profile) -> None:
        """Store the profile; Nexa (trusted corpus) => adequate ≈ always true."""
        self._profile = profile

    def adequate(self, claim: str, source: str) -> bool:
        """Return whether the source is trustworthy enough to count.

        Nexa profile: True (trusted corpus). RavenPack: also True in the stub
        (placeholder) until the real source_quality module is wired in.
        """
        return True


class InferenceValidityStub:
    """Stub for the shared inference-validity check (logical_consistency)."""

    def valid(self, claim: str, context: str) -> bool:
        """Return whether an inferred claim's inference is valid (stub: True)."""
        return True
