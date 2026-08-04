"""§0 step 4 — decontextualize each claim [T2 coref + T3].

Resolve pronouns/referents (carrying context forward, per Molecular Facts /
DnDScore) so the verifier sees a self-contained claim, then confirm it reads as
self-contained. One boolean logged per claim.
"""

from __future__ import annotations

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.pipeline.prompts import PromptLibrary
from rq_eval.providers.base import NlpProvider, ScoringJudge


class Decontextualizer:
    """[T2/T3] Coref-resolve a proposition, then verify it is self-contained."""

    grader_id = "pipeline.decontextualized"

    def __init__(
        self,
        nlp: NlpProvider,
        judge: ScoringJudge,
        prompts: PromptLibrary,
        stamp: tuple[str, str],
        seed: int,
    ) -> None:
        """Inject NLP + judge, prompts, the judge model stamp, and the seed."""
        self._nlp = nlp
        self._judge = judge
        self._prompts = prompts
        self._model, self._version = stamp
        self._seed = seed

    def decontextualize(
        self, proposition: str, context: str, logger: AtomLogger | None = None
    ) -> tuple[str, bool]:
        """Return ``(resolved_text, is_self_contained)``; carry context forward."""
        resolved = self._nlp.resolve_coref(proposition, context).resolved_text
        check = self._judge.binary(self._prompts.decontextualized(), resolved)
        if logger is not None:
            logger.record(
                subject="claim:" + resolved[:40],
                role="decontextualized",
                question=self._prompts.decontextualized(),
                tier="T3",
                verdict=check.verdict,
                evidence=check.reason,
                grader_id=self.grader_id,
                model=self._model,
                model_version=self._version,
                seed=self._seed,
            )
        return resolved, check.verdict
