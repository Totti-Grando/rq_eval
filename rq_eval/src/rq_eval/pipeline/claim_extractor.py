"""§0.2 — deterministic claim decomposition [T1] (+ optional pinned realizer).

Decomposition is *parsing, not generation*: a sentence is split into content-unit
clauses by ``NlpProvider.parse_clauses`` (ClausIE/PredPatt-style live; a rule
splitter in the mock). Abstractive-*implied* content (Claimify's bracketed
``[a celebrity]``) is **flagged and routed**, never generated. The optional
surface-realizer — a pinned ``[T2]`` generation that turns parse-form units into
fluent standalone claims — runs only when ``extraction.realizer_enabled`` is set;
by default the whole path calls no GeneratorProvider and no judge.
"""

from __future__ import annotations

import hashlib

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.graders.t1 import T1Tools
from rq_eval.pipeline.prompts import PromptLibrary
from rq_eval.providers.base import GeneratorProvider, NlpProvider


class ClaimExtractor:
    """[T1] Parse a sentence into atomic propositions (optionally realized)."""

    grader_id = "pipeline.decompose"

    def __init__(
        self,
        nlp: NlpProvider,
        t1: T1Tools,
        generator: GeneratorProvider,
        prompts: PromptLibrary,
        gen_stamp: tuple[str, str],
        seed: int,
        realizer_enabled: bool,
    ) -> None:
        """Inject NLP (parse) + T1 tools; the generator is used only if realizing."""
        self._nlp = nlp
        self._t1 = t1
        self._generator = generator
        self._prompts = prompts
        self._gen_model, self._gen_version = gen_stamp
        self._seed = seed
        self._realizer_enabled = realizer_enabled

    def extract(self, sentence: str, logger: AtomLogger | None = None) -> list[str]:
        """Return atomic propositions; empty list if flagged abstractive-implied.

        Abstractive-implied spans are flagged (not guessed): they yield no claims
        so the pipeline never fabricates the unstated fact.
        """
        flagged = self._t1.is_abstractive_implied(sentence)
        if logger is not None:
            logger.record(
                subject="span:" + hashlib.sha256(sentence.encode()).hexdigest()[:12],
                role="decompose",
                question="abstractive-implied? (flag, don't generate)",
                tier="T1",
                verdict=not flagged,
                evidence="flagged abstractive" if flagged else "parse-decomposed",
                grader_id=self.grader_id,
                model="code",
                model_version="rq_eval",
                seed=self._seed,
            )
        if flagged:
            return []
        clauses = [c for c in self._nlp.parse_clauses(sentence) if c.strip()]
        if self._realizer_enabled:
            clauses = [self._realize(c) for c in clauses]
        return [c for c in clauses if c.strip()]

    def _realize(self, clause: str) -> str:
        """[T2, pinned] Turn a parse-form clause into a fluent standalone claim."""
        result = self._generator.generate(self._prompts.realize(clause), seed=self._seed)
        return result.text.strip() or clause
