"""§3 — the seven source-quality property checks + mean score.

Per source: reachable [T1], dated&fresh [T1], authored [T1], reputable-domain
[T1], corroborated≥N [T1 count over S], supports-claim [imported from S],
disinterested [T3 sampled]. **Supports and corroboration are read off the §1
support set ``S`` — no new NLI here** (that is the support-set reform, Evidence
§3). ``source_quality = mean(property booleans)``; each property is an AtomRecord.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, ContextChunk, Tier
from rq_eval.dimensions.groundedness.export import GroundednessExport
from rq_eval.dimensions.source_quality.coi import CoiRule
from rq_eval.dimensions.source_quality.reliability_list import ReliabilityList
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
        grounded_export: GroundednessExport,
        judge: JudgeGrader,
        reliability: ReliabilityList,
        coi: CoiRule,
        resolver_resolve: object,  # ResolverProvider.resolve bound method
    ) -> None:
        """Inject config, logger, the §1 support set, judge, reliability, COI, resolver."""
        self._cfg = cfg
        self._logger = logger
        self._grounded = grounded_export
        self._judge = judge
        self._reliability = reliability
        self._coi = coi
        self._resolve = resolver_resolve

    def score(
        self, source: ContextChunk, claim: str, sources: list[ContextChunk], *, claim_id: str = ""
    ) -> tuple[float, list[AtomRecord]]:
        """Return (mean(properties), property atoms) for ``source`` vs ``claim``.

        ``supports`` and ``corroborated`` come from the imported support set ``S``
        (per claim when ``claim_id`` is given, else answer-wide) — no NLI here.
        """
        internal = source.url is None and source.domain is None
        disinterested, di_tier = self._disinterested(source, claim)
        supports = (
            self._grounded.claim_supported(claim_id)
            if claim_id
            else self._grounded.answer_supported()
        )
        checks: list[tuple[str, bool, Tier]] = [
            ("reachable", True if internal else bool(self._resolve(source.url)), "T1"),  # type: ignore[operator]
            ("fresh", True if internal else self._fresh(source.date), "T1"),
            ("authored", True if internal else bool(source.author), "T1"),
            ("reputable", self._reliability.is_reliable(source.domain), "T1"),
            ("corroborated", self._corroborated(claim_id), "T1"),
            ("supports", supports, "T1"),
            ("disinterested", disinterested, di_tier),
        ]
        atoms = [self._log(source.id, name, ok, tier) for name, ok, tier in checks]
        score = sum(1 for _, ok, _ in checks if ok) / len(checks)
        return score, atoms

    def _fresh(self, date: str | None) -> bool:
        return date is not None and date <= self._cfg.source_quality.as_of_date

    def _corroborated(self, claim_id: str) -> bool:
        """[T1] Distinct supporting documents in S ≥ corroboration_min (no NLI)."""
        docs = (
            self._grounded.claim_support_docs(claim_id)
            if claim_id
            else self._grounded.answer_support_docs()
        )
        return len(docs) >= self._cfg.source_quality.corroboration_min

    def _disinterested(self, source: ContextChunk, claim: str) -> tuple[bool, Tier]:
        """[T1] COI rule where decisive; only ambiguous sources sample the judge."""
        verdict, _reason = self._coi.decide(source, claim)
        if verdict is not None:  # rule is decisive -> pure T1
            return verdict, "T1"
        rate = self._cfg.source_quality.disinterest_sample_rate
        bucket = int(hashlib.sha256(source.id.encode()).hexdigest(), 16) % 100
        if bucket < rate * 100:  # ambiguous remainder, sampled -> reference-grounded judge [T3]
            v = self._judge.judge(
                subject=f"source:{source.id}", role="disinterest_residual",
                question="Is this source disinterested (not self-serving)?",
                context=source.text, reference=claim, tier="T3",
            ).verdict
            return v, "T3"
        return True, "T1"  # ambiguous, not sampled -> assume disinterested

    def _log(self, source_id: str, name: str, verdict: bool, tier: Tier) -> AtomRecord:
        return self._logger.record(
            subject=f"source:{source_id}", role=f"sq_{name}", question=f"source {name}?",
            tier=tier, verdict=verdict, grader_id="source_quality.property",
            model=_CODE[0], model_version=_CODE[1],
        )
