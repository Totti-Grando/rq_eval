"""§0 step 3 — Claimify extract atomic claims [T3 / T3-gen].

selection (already done by the verifiable filter) -> disambiguation (flag, don't
guess, when context can't resolve) -> extraction (one proposition per claim).
"""

from __future__ import annotations

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.pipeline.prompts import PromptLibrary
from rq_eval.providers.base import GeneratorProvider, ScoringJudge


class ClaimExtractor:
    """[T3] Disambiguate then [T3-gen] extract atomic propositions."""

    grader_id = "pipeline.disambiguate"

    def __init__(
        self,
        judge: ScoringJudge,
        generator: GeneratorProvider,
        prompts: PromptLibrary,
        judge_stamp: tuple[str, str],
        gen_stamp: tuple[str, str],
        seed: int,
    ) -> None:
        """Inject judge + generator, prompts, both model stamps, and the seed."""
        self._judge = judge
        self._generator = generator
        self._prompts = prompts
        self._judge_model, self._judge_version = judge_stamp
        self._gen_model, self._gen_version = gen_stamp
        self._seed = seed

    def extract(self, sentence: str, logger: AtomLogger | None = None) -> list[str]:
        """Return atomic propositions; empty list if flagged ambiguous.

        Ambiguity is flagged (not guessed): a YES verdict routes the sentence
        out and yields no claims.
        """
        flagged = self._judge.binary(self._prompts.disambiguate(), sentence)
        if logger is not None:
            logger.record(
                subject="span:" + sentence[:40],
                role="disambiguate",
                question=self._prompts.disambiguate(),
                tier="T3",
                verdict=flagged.verdict,
                evidence=flagged.reason,
                grader_id=self.grader_id,
                model=self._judge_model,
                model_version=self._judge_version,
                seed=self._seed,
            )
        if flagged.verdict:  # ambiguous + unresolvable -> flag, don't guess
            return []
        result = self._generator.generate(self._prompts.extract(sentence), seed=self._seed)
        return [p for p in result.items if p.strip()]
