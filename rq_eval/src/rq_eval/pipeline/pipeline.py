"""§0 — the shared claim-extraction pipeline (build order B4).

Orchestrates the five steps into cached, decontextualized, verifiable
:class:`Claim`s: segment [T1] -> select verifiable spans [T3] -> Claimify
disambiguate + extract [T3/T3-gen] -> decontextualize [T2/T3] -> pin & measure
stability. Unverifiable spans are excluded and routed; the stability metric is
computed and reported.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import Claim
from rq_eval.pipeline.claim_extractor import ClaimExtractor
from rq_eval.pipeline.decontextualizer import Decontextualizer
from rq_eval.pipeline.prompts import PromptLibrary
from rq_eval.pipeline.segmenter import Segmenter
from rq_eval.pipeline.span_selector import VerifiableSpanSelector
from rq_eval.pipeline.stability import StabilityHarness
from rq_eval.providers.model_stamp import ModelStamp

if TYPE_CHECKING:
    from rq_eval.config import Config
    from rq_eval.providers.factory import Providers

_CITATION = re.compile(r"\[([^\]]+)\]")


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Cached §0 output: claims + routed spans + stability agreement."""

    claims: list[Claim]
    routed_unverifiable: list[str] = field(default_factory=list)
    flagged_ambiguous: list[str] = field(default_factory=list)
    stability: float | None = None


class ClaimPipeline:
    """Builds and runs the §0 pipeline; providers injected via the bundle."""

    def __init__(
        self,
        providers: Providers,
        cfg: Config,
        logger: AtomLogger | None = None,
    ) -> None:
        """Assemble the step objects from the injected providers + config."""
        self._cfg = cfg
        self._logger = logger
        self._extractor_version = cfg.pins.extractor_version
        stamp = ModelStamp(cfg)
        prompts = PromptLibrary(cfg)
        seed = cfg.seeds.judge
        self._segmenter = Segmenter(providers.nlp)
        self._selector = VerifiableSpanSelector(providers.judge, prompts, stamp.judge(), seed)
        self._claimer = ClaimExtractor(
            providers.judge, providers.generator, prompts, stamp.judge(), stamp.generator(), seed
        )
        self._decon = Decontextualizer(providers.nlp, providers.judge, prompts, stamp.judge(), seed)

    def run(self, answer: str, context: str = "") -> PipelineResult:
        """Extract claims (logging atoms) and attach the stability metric."""
        result = self._build(answer, context, self._logger)
        stability = StabilityHarness(self).measure(
            answer, context, runs=self._cfg.pipeline.stability_runs
        )
        return PipelineResult(
            claims=result.claims,
            routed_unverifiable=result.routed_unverifiable,
            flagged_ambiguous=result.flagged_ambiguous,
            stability=stability,
        )

    def claim_ids(self, answer: str, context: str = "") -> list[str]:
        """Claim ids from a logger-less pass (used by the stability harness)."""
        return [c.id for c in self._build(answer, context, None).claims]

    def _build(self, answer: str, context: str, logger: AtomLogger | None) -> PipelineResult:
        classified = self._selector.classify(self._segmenter.segment(answer), logger)
        claims: list[Claim] = []
        routed: list[str] = []
        flagged: list[str] = []
        carried = context
        for sentence, verifiable in classified:
            if not verifiable:
                routed.append(sentence)
                carried = f"{carried} {sentence}".strip()
                continue
            propositions = self._claimer.extract(sentence, logger)
            if not propositions:
                flagged.append(sentence)
            for prop in propositions:
                resolved, decon = self._decon.decontextualize(prop, carried, logger)
                claims.append(self._make_claim(resolved, sentence, decon))
            carried = f"{carried} {sentence}".strip()
        return PipelineResult(claims=claims, routed_unverifiable=routed, flagged_ambiguous=flagged)

    def _make_claim(self, text: str, source: str, decontextualized: bool) -> Claim:
        citation = _CITATION.search(source)
        return Claim(
            id="claim:" + hashlib.sha256(text.encode()).hexdigest()[:12],
            text=text,
            source_sentence=source,
            verifiable=True,
            decontextualized=decontextualized,
            extractor_version=self._extractor_version,
            citation=citation.group(1) if citation else None,
        )
