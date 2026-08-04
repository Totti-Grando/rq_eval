"""§2 completeness — two-tier nugget recall (build order B8).

Tier-1 requirement scaffold (oracle) → Tier-2 unit drafting → admissibility gate
→ dedupe → per-unit assignment → two-level scoring. Headline score is strict
vital recall (replayable via `mean` over vital support atoms); requirement
coverage + weighted recall are reported alongside. Wilson CI + min-n abstention.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import DimensionResult, EvalInput
from rq_eval.dimensions.base import Dimension
from rq_eval.dimensions.completeness.admissibility_gate import UnitAdmissibilityGate
from rq_eval.dimensions.completeness.deduper import UnitDeduper
from rq_eval.dimensions.completeness.requirement_templates import RequirementTemplates
from rq_eval.dimensions.completeness.two_level_scoring import TwoLevelScoring
from rq_eval.dimensions.completeness.unit import Unit
from rq_eval.dimensions.completeness.unit_assigner import UnitAssigner
from rq_eval.dimensions.completeness.unit_drafter import UnitDrafter
from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.graders.judge_grader import JudgeGrader
from rq_eval.graders.t1 import T1Tools
from rq_eval.providers.model_stamp import ModelStamp
from rq_eval.scoring.aggregation import MinNAbstention
from rq_eval.scoring.bands import BandMapper
from rq_eval.scoring.formulas import default_registry
from rq_eval.scoring.wilson import WilsonInterval

if TYPE_CHECKING:
    from rq_eval.config import Config
    from rq_eval.providers.factory import Providers

_FORMULA = "mean"  # strict vital recall = mean over vital support atoms


class CompletenessDimension(Dimension):
    """§2 — two-tier nugget recall over a fixed requirement scaffold."""

    name = "completeness"

    def __init__(self, providers: Providers, cfg: Config, logger: AtomLogger) -> None:
        """Assemble the Tier-1/Tier-2 pipeline from injected providers + config."""
        self._cfg = cfg
        self._logger = logger
        stamp = ModelStamp(cfg)
        seed = cfg.seeds.judge
        self._templates = RequirementTemplates(cfg)
        self._drafter = UnitDrafter(providers.generator, providers.nlp, cfg.seeds.dedupe)
        self._gate = UnitAdmissibilityGate(
            T1Tools(), providers.nlp,
            GroundingGrader(
                providers.grounding, logger, stamp.grounding(), "completeness.decidable_nli", seed
            ),
            JudgeGrader(
                providers.judge, logger, stamp.judge(), "completeness.decidability_residual", seed
            ),
            logger,
            double_nli=cfg.completeness.double_nli,
        )
        self._deduper = UnitDeduper(providers.embedding, cfg.completeness.dedupe_tau)
        vital_w = 2.0 if cfg.completeness.vital_weighting else 1.0
        self._assigner = UnitAssigner(
            GroundingGrader(
                providers.grounding, logger, stamp.grounding(), "completeness.assign", seed
            ),
            vital_weight=vital_w, okay_weight=1.0,
        )
        self._scorer = TwoLevelScoring()
        self._registry = default_registry()
        self._bands = BandMapper(cfg.thresholds.bands.G, cfg.thresholds.bands.A)

    def evaluate(self, eval_input: EvalInput) -> DimensionResult:
        """Build the frozen unit set, assign, and score strict vital recall."""
        requirements = self._templates.requirements_for(eval_input.question)
        sources = " ".join(c.text for c in eval_input.context)
        candidates: list[Unit] = []
        for req in requirements:
            candidates.extend(self._drafter.draft(req, sources))
        frozen = self._deduper.dedupe(
            self._gate.admit(candidates, answer=eval_input.answer, sources=sources)
        )
        self._log_frozen_set(frozen, sources)

        support = self._assigner.assign(frozen, eval_input.answer)
        vital_atoms = [a for u, a in zip(frozen, support, strict=True) if u.vital]
        score = self._registry.compute(_FORMULA, vital_atoms)

        vital_total = len(vital_atoms)
        vital_supported = sum(1 for a in vital_atoms if a.verdict)
        low, high = WilsonInterval().interval(vital_supported, vital_total)
        abstained = MinNAbstention().should_abstain(vital_total, self._cfg.completeness.min_n)
        return DimensionResult(
            dimension=self.name, score=score, band=self._bands.band(score),
            ci_low=low, ci_high=high, n=vital_total,
            inputs_hash=self._corpus_hash(sources, eval_input.answer),
            atom_ids=[a.id for a in vital_atoms], formula_id=_FORMULA, abstained=abstained,
            extra={
                "requirement_coverage": self._scorer.requirement_coverage(
                    frozen, support, requirements
                ),
                "weighted_recall": self._scorer.weighted_recall(
                    frozen, support, requirements, self._cfg.completeness.vital_weighting
                ),
                "vital_units": float(vital_total),
                "total_units": float(len(frozen)),
            },
        )

    def _log_frozen_set(self, units: list[Unit], sources: str) -> None:
        """Record the frozen set's provenance (version + corpus hash)."""
        self._logger.record(
            subject="frozen_set", role="frozen_set", question="frozen unit set", tier="T1",
            verdict=True,
            evidence=(
                f"nuggetizer={self._cfg.pins.nuggetizer_version} "
                f"template={self._templates.version} "
                f"corpus_hash={self._corpus_hash(sources, '')} n_units={len(units)}"
            ),
            grader_id="completeness.frozen_set", model="code", model_version="rq_eval",
        )

    @staticmethod
    def _corpus_hash(sources: str, answer: str) -> str:
        return hashlib.sha256(f"{sources}||{answer}".encode()).hexdigest()[:16]
