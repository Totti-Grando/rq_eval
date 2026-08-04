"""§3 step 4 — per-claim responsive atom (on-topic ∧ on-ask) [T2/T1].

Both signals are fixed (no judge, per the DIVER-QA reform):
* on-topic — `RelevanceProvider.score(question, claim) ≥ relevance_tau` `[T2]`;
* on-ask  — `on_ask_nli ∨ on_ask_lex`, where `on_ask_nli = entails(premise=claim,
  hypothesis=ask) == E` `[T2]` (ask = `T1Tools.ask_hypothesis(question)`) and
  `on_ask_lex = key_term_overlap(question, claim) ≥ lexical_min_overlap` `[T1]`.

`responsive = on_topic ∧ on_ask` is the boolean accuracy imports.
"""

from __future__ import annotations

from dataclasses import dataclass

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, Claim
from rq_eval.dimensions.responsiveness import ResponsivenessExport
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.graders.relevance_grader import RelevanceGrader
from rq_eval.graders.t1 import T1Tools

_CODE = ("code", "rq_eval")


@dataclass(frozen=True, slots=True)
class ClaimSignals:
    """Per-claim relevance signals used to seed anchors + classify orphans.

    ``on_ask`` seeds the anchor set (a direct question hit); ``on_topic`` splits
    off-topic orphans from background; ``responsive`` (= on_topic ∧ on_ask) is
    the atom accuracy imports.
    """

    claim: Claim
    on_topic: bool
    on_ask: bool
    responsive: AtomRecord


class ClaimResponsiveness:
    """Computes + logs the per-claim responsive atom (fixed NLI + lexical)."""

    def __init__(
        self,
        on_topic: RelevanceGrader,
        on_ask_nli: GroundingGrader,
        t1: T1Tools,
        logger: AtomLogger,
        stamp: tuple[str, str],
        seed: int,
        lexical_min_overlap: float,
    ) -> None:
        """Inject the on-topic (relevance) + on-ask (NLI) graders, T1 tools, config."""
        self._on_topic = on_topic
        self._on_ask_nli = on_ask_nli
        self._t1 = t1
        self._logger = logger
        self._model, self._version = stamp
        self._seed = seed
        self._lex_min = lexical_min_overlap

    def compute(
        self, question: str, claims: list[Claim], export: ResponsivenessExport
    ) -> list[ClaimSignals]:
        """Return per-claim signals; log + publish the responsive atom to ``export``."""
        ask = self._t1.ask_hypothesis(question)
        signals: list[ClaimSignals] = []
        for claim in claims:
            on_topic = self._on_topic.check(
                subject=claim.id, role="on_topic", query=question, response=claim.text
            )
            nli_atom, _ = self._on_ask_nli.assess(
                subject=claim.id, role="on_ask_nli", premise=claim.text, hypothesis=ask
            )
            lex = self._t1.key_term_overlap(question, claim.text) >= self._lex_min
            self._logger.record(
                subject=claim.id, role="on_ask_lex", question="lexical key-term overlap",
                tier="T1", verdict=lex, evidence=f"min={self._lex_min}",
                grader_id="relevance.on_ask_lex", model=_CODE[0], model_version=_CODE[1],
            )
            on_ask = nli_atom.verdict or lex
            responsive = on_topic.verdict and on_ask
            atom = self._logger.record(
                subject=claim.id, role="responsive", question="on_topic AND on_ask",
                tier="T2", verdict=responsive,
                evidence=f"on_topic={on_topic.verdict} nli={nli_atom.verdict} lex={lex}",
                grader_id="relevance.responsive", model=self._model,
                model_version=self._version, seed=self._seed,
            )
            export.set(claim.id, atom)
            signals.append(
                ClaimSignals(
                    claim=claim, on_topic=on_topic.verdict, on_ask=on_ask, responsive=atom
                )
            )
        return signals
