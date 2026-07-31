"""Live grounding — fairseq RoBERTa-MNLI via torch.hub (design §1, optional live).

Native three-way NLI (Contradiction / Neutral / Entailment). Selected by
``models.nli: fairseq``. torch/fairseq are imported lazily so this file is
import-safe without them installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rq_eval.providers.base import EntailmentLabel, EntailmentResult, GroundingProvider

if TYPE_CHECKING:
    from rq_eval.config import Config

# fairseq MNLI label order: 0=contradiction, 1=neutral, 2=entailment
_LABELS: tuple[EntailmentLabel, ...] = ("C", "N", "E")


class FairseqGroundingProvider(GroundingProvider):
    """Native three-way entailment from a torch.hub fairseq RoBERTa-large-MNLI."""

    def __init__(self, cfg: Config) -> None:
        """Store config; the model is loaded lazily on first call."""
        self._cfg = cfg
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            import torch  # noqa: PLC0415 - lazy optional dependency

            model = torch.hub.load("pytorch/fairseq", "roberta.large.mnli")
            model.eval()
            self._model = model
        return self._model

    def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        """Argmax over {C, N, E}; raw_score = P(entailment)."""
        model = self._load()
        tokens = model.encode(premise, hypothesis)
        logits = model.predict("mnli", tokens)
        probs = logits.exp() / logits.exp().sum()
        idx = int(probs.argmax())
        return EntailmentResult(label=_LABELS[idx], raw_score=float(probs[0][2]))
