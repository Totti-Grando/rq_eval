"""§2 hallucination — unsupported rate + fabrication gate (build order E4).

Two distinct failures: the **unsupported rate** (`1 − groundedness`, read from
§1's triplet verdicts, with Neutral vs Contradiction reported separately) and the
deterministic **fabrication gate** (a fabricated citation fails the run). The
unsupported rate is the score; fabrication is a hard gate.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import Claim, DimensionResult, EvalInput
from rq_eval.dimensions.base import Dimension
from rq_eval.dimensions.groundedness.export import GroundednessExport
from rq_eval.dimensions.hallucination.fabrication_gate import FabricationGate
from rq_eval.scoring.bands import BandMapper
from rq_eval.scoring.formulas import default_registry

if TYPE_CHECKING:
    from rq_eval.config import Config
    from rq_eval.providers.factory import Providers

_FORMULA = "unsupported_rate"


class HallucinationDimension(Dimension):
    """§2 — unsupported-claim rate (N/C split) + the T1 fabrication gate."""

    name = "hallucination"

    def __init__(
        self,
        providers: Providers,
        cfg: Config,
        logger: AtomLogger,
        claims: list[Claim],
        grounded_export: GroundednessExport,
    ) -> None:
        """Inject the groundedness export (triplet verdicts) + resolver gate."""
        self._logger = logger
        self._claims = claims
        self._export = grounded_export
        self._gate = FabricationGate(providers.resolver, logger)
        self._registry = default_registry()
        self._bands = BandMapper(cfg.thresholds.bands.G, cfg.thresholds.bands.A)

    def evaluate(self, eval_input: EvalInput) -> DimensionResult:
        """Score the unsupported rate; run the fabrication gate over citations."""
        triplet_ids = self._export.triplet_atom_ids()
        counts = self._export.label_counts()
        total = sum(counts.values())
        # score = unsupported rate = 1 - |E|/|total|, replayed from triplet atoms
        score = 1.0 - (counts.get("E", 0) / total) if total else 0.0

        retrieved_ids = {c.id for c in eval_input.context}
        citations = [c.citation for c in self._claims if c.citation] + list(eval_input.citations)
        fab = self._gate.check(citations, retrieved_ids)

        neutral_rate = counts.get("N", 0) / total if total else 0.0
        contradiction_rate = counts.get("C", 0) / total if total else 0.0
        # band on "goodness" (1 - unsupported); a fabricated citation forces R
        band = "R" if fab.gate_failed else self._bands.band(1.0 - score)
        return DimensionResult(
            dimension=self.name, score=score, band=band,
            ci_low=0.0, ci_high=1.0, n=total,
            inputs_hash=hashlib.sha256(eval_input.answer.encode()).hexdigest()[:16],
            atom_ids=triplet_ids, formula_id=_FORMULA, abstained=(total == 0),
            extra={
                "unsupported_rate": score,
                "neutral_rate": neutral_rate,
                "contradiction_rate": contradiction_rate,
                "gate_failed": 1.0 if fab.gate_failed else 0.0,
            },
        )
