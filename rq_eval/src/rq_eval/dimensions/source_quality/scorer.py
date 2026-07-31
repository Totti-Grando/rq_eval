"""§3 — the seven source-quality property checks + mean score.

Per source: reachable [T1], dated&fresh [T1], authored [T1], reputable-domain
[T1], corroborated≥N [T1 count], supports-claim [T2 via entails], disinterested
[T3 sampled]. ``source_quality = mean(property booleans)``. Each property is an
AtomRecord so the score (and accuracy's imported source-adequate) is auditable.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, ContextChunk, Tier
from rq_eval.dimensions.source_quality.reliability_list import ReliabilityList
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.graders.judge_grader import JudgeGrader

if TYPE_CHECKING:
    from rq_eval.config import Config

_CODE = ("code", "rq_eval")


class SourceQualityScorer:
    """Runs the seven property checks for one source and logs their atoms."""

    def __init__(
        self,
        cfg: Config,
        logger: AtomLogger,
        grounding: GroundingGrader,
        judge: JudgeGrader,
        reliability: ReliabilityList,
        resolver_resolve: object,  # ResolverProvider.resolve bound method
    ) -> None:
        """Inject config, logger, grounding/judge graders, reliability list, resolver."""
        self._cfg = cfg
        self._logger = logger
        self._grounding = grounding
        self._judge = judge
        self._reliability = reliability
        self._resolve = resolver_resolve

    def score(
        self, source: ContextChunk, claim: str, sources: list[ContextChunk]
    ) -> tuple[float, list[AtomRecord]]:
        """Return (mean(properties), property atoms) for ``source`` vs ``claim``."""
        internal = source.url is None and source.domain is None
        checks: list[tuple[str, bool, Tier]] = [
            ("reachable", True if internal else bool(self._resolve(source.url)), "T1"),  # type: ignore[operator]
            ("fresh", True if internal else self._fresh(source.date), "T1"),
            ("authored", True if internal else bool(source.author), "T1"),
            ("reputable", self._reliability.is_reliable(source.domain), "T1"),
            ("corroborated", self._corroborated(claim, sources), "T1"),
            ("supports", self._grounding.classify(source.text, claim).supported, "T2"),
            ("disinterested", self._disinterested(source), "T1"),
        ]
        atoms = [self._log(source.id, name, ok, tier) for name, ok, tier in checks]
        score = sum(1 for _, ok, _ in checks if ok) / len(checks)
        return score, atoms

    def _fresh(self, date: str | None) -> bool:
        return date is not None and date <= self._cfg.source_quality.as_of_date

    def _corroborated(self, claim: str, sources: list[ContextChunk]) -> bool:
        keys = {
            (s.domain or s.author or s.id)
            for s in sources
            if self._grounding.classify(s.text, claim).supported
        }
        return len(keys) >= self._cfg.source_quality.corroboration_min

    def _disinterested(self, source: ContextChunk) -> bool:
        rate = self._cfg.source_quality.disinterest_sample_rate
        bucket = int(hashlib.sha256(source.id.encode()).hexdigest(), 16) % 100
        if bucket < rate * 100:  # sampled -> judge [T3]
            return self._judge.judge(
                subject=f"source:{source.id}", role="sq_disinterest_judge",
                question="[[affirm]] Is this source disinterested (not self-serving)?",
                context=source.text, tier="T3",
            ).verdict
        return True  # not sampled -> assumed disinterested

    def _log(self, source_id: str, name: str, verdict: bool, tier: Tier) -> AtomRecord:
        return self._logger.record(
            subject=f"source:{source_id}", role=f"sq_{name}", question=f"source {name}?",
            tier=tier, verdict=verdict, grader_id="source_quality.property",
            model=_CODE[0], model_version=_CODE[1],
        )
