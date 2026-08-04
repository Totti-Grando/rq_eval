"""§0 step 2 — select verifiable spans [T1] (VeriScore filter).

Keep only spans that can plausibly be proven true or false; route opinions,
hedges, and hypotheticals away from truth scoring. The decision is a pure
lexical/syntactic check (``T1Tools.is_verifiable``) — no judge — with the
ambiguous remainder left for an optional fixed ``[T2]`` classifier when live.
One boolean per sentence, each logged as an atom.
"""

from __future__ import annotations

import hashlib

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.graders.t1 import T1Tools


class VerifiableSpanSelector:
    """[T1] Classifies each sentence as verifiable (keep) or not (route)."""

    grader_id = "pipeline.verifiable"

    def __init__(self, t1: T1Tools, seed: int) -> None:
        """Inject the T1 toolbox and the pipeline seed (for atom parity)."""
        self._t1 = t1
        self._seed = seed

    def classify(
        self, sentences: list[str], logger: AtomLogger | None = None
    ) -> list[tuple[str, bool]]:
        """Return ``(sentence, verifiable)`` pairs; log one T1 atom per sentence."""
        out: list[tuple[str, bool]] = []
        for sentence in sentences:
            verifiable = self._t1.is_verifiable(sentence)
            if logger is not None:
                logger.record(
                    subject="span:" + hashlib.sha256(sentence.encode()).hexdigest()[:12],
                    role="verifiable",
                    question="provable true/false? (opinions/hedges routed)",
                    tier="T1",
                    verdict=verifiable,
                    evidence="lexical hedge/opinion filter",
                    grader_id=self.grader_id,
                    model="code",
                    model_version="rq_eval",
                    seed=self._seed,
                )
            out.append((sentence, verifiable))
        return out
