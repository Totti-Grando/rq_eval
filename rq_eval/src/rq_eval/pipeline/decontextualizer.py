"""§0.2 — decontextualize a proposition [T1/T2] (coref, no judge).

Coreference substitution (``NlpProvider.resolve_coref`` — coreferee live, a
leading-pronoun rule in the mock) carries the surrounding context into the claim,
per Molecular Facts / DnDScore. Self-containedness is then a **structural** check
— the resolved claim must not still open with an unresolved pronoun — mirroring
the admissibility gate; no judge is consulted.
"""

from __future__ import annotations

import hashlib

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.graders.t1 import T1Tools
from rq_eval.providers.base import NlpProvider


class Decontextualizer:
    """[T1/T2] Resolve references, then structurally verify self-containedness."""

    grader_id = "pipeline.decontextualized"

    def __init__(self, nlp: NlpProvider, t1: T1Tools, seed: int) -> None:
        """Inject the NLP provider (coref) + T1 tools + the pipeline seed."""
        self._nlp = nlp
        self._t1 = t1
        self._seed = seed

    def decontextualize(
        self, proposition: str, context: str, logger: AtomLogger | None = None
    ) -> tuple[str, bool]:
        """Return ``(resolved_text, self_contained)``; log one T1 atom.

        ``self_contained`` is True iff the resolved text no longer begins with an
        unresolved pronoun (a pure structural check, not a judge call).
        """
        resolved = self._nlp.resolve_coref(proposition, context).resolved_text
        self_contained = not self._t1.has_leading_pronoun(resolved)
        if logger is not None:
            logger.record(
                subject="claim:" + hashlib.sha256(resolved.encode()).hexdigest()[:12],
                role="decontextualized",
                question="self-contained after coref? (no leading pronoun)",
                tier="T1",
                verdict=self_contained,
                evidence="structural leading-pronoun check",
                grader_id=self.grader_id,
                model="code",
                model_version="rq_eval",
                seed=self._seed,
            )
        return resolved, self_contained
