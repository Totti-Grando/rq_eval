"""Live grounding — fairseq RoBERTa-MNLI via torch.hub (B2, optional live path).

Non-HF NLI: entailment probability of ``claim`` (hypothesis) given ``source``
(premise). Returned raw; thresholded in our code. Selected by models.nli:
fairseq. torch/fairseq are imported lazily so this file is import-safe without
them installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rq_eval.providers.base import GroundingProvider, GroundingResult

if TYPE_CHECKING:
    from rq_eval.config import Config


class FairseqGroundingProvider(GroundingProvider):
    """Entailment score from a torch.hub fairseq RoBERTa-large-MNLI model."""

    def __init__(self, cfg: Config) -> None:
        """Store config; the model is loaded lazily on first check()."""
        self._cfg = cfg
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            import torch  # noqa: PLC0415 - lazy optional dependency

            model = torch.hub.load("pytorch/fairseq", "roberta.large.mnli")
            model.eval()
            self._model = model
        return self._model

    def check(self, source: str, claim: str) -> GroundingResult:
        """P(entailment) of claim given source via RoBERTa-MNLI (label idx 2)."""
        model = self._load()
        tokens = model.encode(source, claim)
        logits = model.predict("mnli", tokens)
        probs = logits.exp() / logits.exp().sum()
        # fairseq MNLI label order: 0=contradiction, 1=neutral, 2=entailment
        return GroundingResult(raw_score=float(probs[0][2]))
