"""Live grounding — Bedrock Guardrails contextual grounding (design §1, live).

ApplyGuardrail returns a contextual-grounding GROUNDING filter score. Guardrails
can't distinguish Neutral from Contradiction, so this maps score→{E, N} via
``entail_tau`` (use the fairseq backend for native three-way with C).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rq_eval.providers.base import EntailmentLabel, EntailmentResult, GroundingProvider
from rq_eval.providers.live.bedrock_session import BedrockSession

if TYPE_CHECKING:
    from rq_eval.config import Config


class GuardrailGroundingProvider(GroundingProvider):
    """Three-way (E/N only) entailment from a Bedrock contextual-grounding policy."""

    def __init__(self, cfg: Config, session: BedrockSession) -> None:
        """Store config + shared Bedrock session (no network yet)."""
        self._cfg = cfg
        self._session = session

    def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        """ApplyGuardrail(premise as grounding_source)→ score→ E if ≥ entail_tau else N."""
        resp = self._session.runtime().apply_guardrail(
            guardrailIdentifier=self._cfg.models.guardrail_id,
            guardrailVersion=self._cfg.models.guardrail_version,
            source="OUTPUT",
            content=[
                {"text": {"text": premise, "qualifiers": ["grounding_source"]}},
                {"text": {"text": hypothesis}},
            ],
        )
        score = _filter_score(resp, "GROUNDING")
        label: EntailmentLabel = "E" if score >= self._cfg.thresholds.entail_tau else "N"
        return EntailmentResult(label=label, raw_score=score)


def _filter_score(resp: dict[str, Any], filter_type: str) -> float:
    """Extract a contextual-grounding filter score by type (0.0 if absent)."""
    for assessment in resp.get("assessments", []):
        policy = assessment.get("contextualGroundingPolicy", {})
        for filt in policy.get("filters", []):
            if filt.get("type") == filter_type:
                return float(filt.get("score", 0.0))
    return 0.0
