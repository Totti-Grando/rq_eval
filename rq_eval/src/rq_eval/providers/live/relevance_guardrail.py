"""Live relevance — Bedrock Guardrails contextual-grounding RELEVANCE (B2).

Method B: a fixed query↔response relevance score from ApplyGuardrail, returned
raw and thresholded in our code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rq_eval.providers.base import RelevanceProvider
from rq_eval.providers.live.bedrock_session import BedrockSession
from rq_eval.providers.live.grounding_guardrail import _filter_score

if TYPE_CHECKING:
    from rq_eval.config import Config


class GuardrailRelevanceProvider(RelevanceProvider):
    """Query/response relevance from a Bedrock Guardrail (contextual grounding)."""

    def __init__(self, cfg: Config, session: BedrockSession) -> None:
        """Store config + shared Bedrock session (no network yet)."""
        self._cfg = cfg
        self._session = session

    def score(self, query: str, response: str) -> float:
        """ApplyGuardrail(query as query qualifier, response as output)→ score."""
        resp = self._session.runtime().apply_guardrail(
            guardrailIdentifier=self._cfg.models.guardrail_id,
            guardrailVersion=self._cfg.models.guardrail_version,
            source="OUTPUT",
            content=[
                {"text": {"text": query, "qualifiers": ["query"]}},
                {"text": {"text": response}},
            ],
        )
        return _filter_score(resp, "RELEVANCE")
