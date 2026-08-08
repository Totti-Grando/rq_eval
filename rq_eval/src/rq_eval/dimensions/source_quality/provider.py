"""§3/§6 — SourceQualityProvider implementation (accuracy imports this)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rq_eval.contracts import ContextChunk
from rq_eval.dimensions.source_quality.scorer import SourceQualityScorer
from rq_eval.providers.base import SourceQualityProvider

if TYPE_CHECKING:
    from rq_eval.config import Config


class SourceQualityProviderImpl(SourceQualityProvider):
    """``adequate`` = source_quality score ≥ config threshold (properties logged)."""

    def __init__(self, cfg: Config, scorer: SourceQualityScorer) -> None:
        """Inject config (threshold) and the property scorer."""
        self._threshold = cfg.source_quality.adequacy_threshold
        self._scorer = scorer

    def adequate(
        self, source: ContextChunk, claim: str, sources: list[ContextChunk], claim_id: str = ""
    ) -> bool:
        """Score the source's properties and return score ≥ threshold."""
        score, _atoms = self._scorer.score(source, claim, sources, claim_id=claim_id)
        return score >= self._threshold
