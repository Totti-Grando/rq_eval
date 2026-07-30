"""§0 step 2 — select verifiable spans [T3] (VeriScore filter).

Keep only spans that can plausibly be proven true or false; route opinions,
hedges, and hypotheticals away from truth scoring. One boolean per sentence,
each logged as an atom.
"""

from __future__ import annotations

import hashlib

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.pipeline.prompts import PromptLibrary
from rq_eval.providers.base import JudgeProvider


class VerifiableSpanSelector:
    """[T3] Classifies each sentence as verifiable (keep) or not (route)."""

    grader_id = "pipeline.verifiable"

    def __init__(
        self, judge: JudgeProvider, prompts: PromptLibrary, stamp: tuple[str, str], seed: int
    ) -> None:
        """Inject judge, prompts, the model stamp, and the judge seed."""
        self._judge = judge
        self._prompts = prompts
        self._model, self._version = stamp
        self._seed = seed

    def classify(
        self, sentences: list[str], logger: AtomLogger | None = None
    ) -> list[tuple[str, bool]]:
        """Return ``(sentence, verifiable)`` pairs; log one atom per sentence."""
        out: list[tuple[str, bool]] = []
        for sentence in sentences:
            verdict = self._judge.binary(self._prompts.verifiable(), sentence)
            if logger is not None:
                logger.record(
                    subject="span:" + hashlib.sha256(sentence.encode()).hexdigest()[:12],
                    role="verifiable",
                    question=self._prompts.verifiable(),
                    tier="T3",
                    verdict=verdict.verdict,
                    evidence=verdict.reason,
                    grader_id=self.grader_id,
                    model=self._model,
                    model_version=self._version,
                    seed=self._seed,
                )
            out.append((sentence, verdict.verdict))
        return out
