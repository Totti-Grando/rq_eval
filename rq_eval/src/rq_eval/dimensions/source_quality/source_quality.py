"""§3 source_quality — the bridge to world factuality (build order E5).

Per source, the seven property checks → ``mean(properties)``; the dimension
reports the mean source_quality across the answer's sources. The
``SourceQualityProvider`` (same scorer) is what accuracy imports as
``source-adequate?``, replacing the old stub.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, DimensionResult, EvalInput
from rq_eval.dimensions.base import Dimension
from rq_eval.dimensions.source_quality.reliability_list import ReliabilityList
from rq_eval.dimensions.source_quality.scorer import SourceQualityScorer
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.graders.judge_grader import JudgeGrader
from rq_eval.providers.model_stamp import ModelStamp
from rq_eval.scoring.bands import BandMapper
from rq_eval.scoring.formulas import default_registry

if TYPE_CHECKING:
    from rq_eval.config import Config
    from rq_eval.providers.factory import Providers

_FORMULA = "mean"


class SourceQualityDimension(Dimension):
    """§3 — per-source property mean, aggregated across the answer's sources."""

    name = "source_quality"

    def __init__(self, providers: Providers, cfg: Config, logger: AtomLogger) -> None:
        """Assemble the property scorer from providers + config."""
        self._logger = logger
        stamp = ModelStamp(cfg)
        seed = cfg.seeds.judge
        grounding = GroundingGrader(
            providers.grounding, logger, stamp.grounding(), "source_quality.supports", seed
        )
        judge = JudgeGrader(
            providers.judge, logger, stamp.judge(), "source_quality.disinterest", seed
        )
        self._scorer = SourceQualityScorer(
            cfg, logger, grounding, judge, ReliabilityList(cfg), providers.resolver.resolve
        )
        self._threshold = cfg.source_quality.adequacy_threshold
        self._registry = default_registry()
        self._bands = BandMapper(cfg.thresholds.bands.G, cfg.thresholds.bands.A)

    def evaluate(self, eval_input: EvalInput) -> DimensionResult:
        """Score each source (vs the answer) and average; one 'adequate' atom/source."""
        sources = eval_input.context
        atoms: list[AtomRecord] = []
        for source in sources:
            score, _props = self._scorer.score(source, eval_input.answer, sources)
            atoms.append(
                self._logger.record(
                    subject=f"source:{source.id}", role="source_adequate",
                    question="source adequate (score >= threshold)?", tier="T1",
                    verdict=score >= self._threshold,
                    evidence=f"score={score:.4f}", grader_id="source_quality.adequate",
                    model="code", model_version="rq_eval",
                )
            )
        score = self._registry.compute(_FORMULA, atoms)
        adequate_n = sum(1 for a in atoms if a.verdict)
        return DimensionResult(
            dimension=self.name, score=score, band=self._bands.band(score),
            ci_low=0.0, ci_high=1.0, n=len(sources),
            inputs_hash=hashlib.sha256(eval_input.answer.encode()).hexdigest()[:16],
            atom_ids=[a.id for a in atoms], formula_id=_FORMULA, abstained=(len(sources) == 0),
            extra={"adequate_sources": float(adequate_n), "total_sources": float(len(sources))},
        )
