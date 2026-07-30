"""Live grounding — Bedrock Guardrails contextual grounding (B2, live path).

ApplyGuardrail returns a contextual-grounding GROUNDING filter score for the
claim against the source. We return that raw score; our code thresholds it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rq_eval.providers.base import GroundingProvider, GroundingResult
from rq_eval.providers.live.bedrock_session import BedrockSession

if TYPE_CHECKING:
    from rq_eval.config import Config


class GuardrailGroundingProvider(GroundingProvider):
    """Grounding score from a Bedrock Guardrail's contextual-grounding policy."""

    def __init__(self, cfg: Config, session: BedrockSession) -> None:
        """Store config + shared Bedrock session (no network yet)."""
        self._cfg = cfg
        self._session = session

    def check(self, source: str, claim: str) -> GroundingResult:
        """ApplyGuardrail(source as grounding_source, claim as output)→ score."""
        resp = self._session.runtime().apply_guardrail(
            guardrailIdentifier=self._cfg.models.guardrail_id,
            guardrailVersion=self._cfg.models.guardrail_version,
            source="OUTPUT",
            content=[
                {"text": {"text": source, "qualifiers": ["grounding_source"]}},
                {"text": {"text": claim}},
            ],
        )
        return GroundingResult(raw_score=_filter_score(resp, "GROUNDING"))


def _filter_score(resp: dict[str, Any], filter_type: str) -> float:
    """Extract a contextual-grounding filter score by type (0.0 if absent)."""
    for assessment in resp.get("assessments", []):
        policy = assessment.get("contextualGroundingPolicy", {})
        for filt in policy.get("filters", []):
            if filt.get("type") == filter_type:
                return float(filt.get("score", 0.0))
    return 0.0
