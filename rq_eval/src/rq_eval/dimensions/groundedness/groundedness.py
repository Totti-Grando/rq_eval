"""§1 groundedness — source faithfulness via a per-chunk support pass (Evidence §1).

Similarity pre-filter keeps the top-``groundedness_k`` chunks per triplet [T1];
``GroundingProvider.entails(chunk, triplet)`` runs **once per kept chunk** [T2];
code collects the **support set** ``S = {chunk : E}`` (grouped by document) and
scores ``groundedness = |triplets with S≠∅| / |total|``. ``S`` is logged and
exported — source_quality (corroboration/supports) and source_attribution
(``C∩S``) derive from it with no further NLI.

*Scope statement:* groundedness is **direct, per-claim source presence** — the
axiom-*builder* for the claim graph (§RQ 0.3), **not** the answer's headline
factuality number. A validly-*derived* claim correctly scores "not directly
grounded" here; its transitive truth is accuracy's DAG resolution (§RQ 1).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord, ContextChunk, DimensionResult, EvalInput, Triplet
from rq_eval.dimensions.base import Dimension
from rq_eval.dimensions.groundedness.export import GroundednessExport
from rq_eval.dimensions.groundedness.prefilter import SimilarityPreFilter
from rq_eval.providers.model_stamp import ModelStamp
from rq_eval.scoring.bands import BandMapper
from rq_eval.scoring.formulas import default_registry
from rq_eval.scoring.wilson import WilsonInterval

if TYPE_CHECKING:
    from rq_eval.config import Config
    from rq_eval.providers.factory import Providers

_FORMULA = "mean"  # |triplets with S≠∅| / |total triplets|


def _doc_key(chunk: ContextChunk) -> str:
    """Source-document identity for corroboration (distinct-doc counting)."""
    return chunk.domain or chunk.url or chunk.id


class GroundednessDimension(Dimension):
    """§1 — per-triplet three-way entailment; exports per-claim grounded."""

    name = "groundedness"

    def __init__(
        self,
        providers: Providers,
        cfg: Config,
        logger: AtomLogger,
        triplets: list[Triplet],
        export: GroundednessExport,
    ) -> None:
        """Assemble pre-filter + grounding grader; store triplets + export."""
        self._logger = logger
        self._triplets = triplets
        self._export = export
        self._grounding_provider = providers.grounding
        self._k = cfg.groundedness.groundedness_k
        stamp = ModelStamp(cfg)
        self._prefilter = SimilarityPreFilter(providers.embedding)
        self._stamp = stamp.grounding()
        self._claim_stamp = stamp.grounding()
        self._seed = cfg.seeds.judge
        self._registry = default_registry()
        self._bands = BandMapper(cfg.thresholds.bands.G, cfg.thresholds.bands.A)

    def evaluate(self, eval_input: EvalInput) -> DimensionResult:
        """Per-chunk support pass: build S per triplet; score |supported|/|total|."""
        chunks = eval_input.context
        triplet_atoms: list[AtomRecord] = []
        by_claim: dict[str, list[tuple[bool, float]]] = {}
        for triplet in self._triplets:
            atom, is_supported, best_conf = self._assess_triplet(triplet, chunks)
            triplet_atoms.append(atom)
            by_claim.setdefault(triplet.claim_id, []).append((is_supported, best_conf))

        self._export_per_claim(by_claim)
        score = self._registry.compute(_FORMULA, triplet_atoms)
        supported = sum(1 for a in triplet_atoms if a.verdict)
        n = len(triplet_atoms)
        low, high = WilsonInterval().interval(supported, n)
        return DimensionResult(
            dimension=self.name, score=score, band=self._bands.band(score),
            ci_low=low, ci_high=high, n=n,
            inputs_hash=hashlib.sha256(
                " ".join(c.text for c in chunks).encode()
            ).hexdigest()[:16],
            atom_ids=[a.id for a in triplet_atoms], formula_id=_FORMULA, abstained=(n == 0),
            extra={"supported": float(supported), "total": float(n)},
        )

    def _assess_triplet(
        self, triplet: Triplet, chunks: list[ContextChunk]
    ) -> tuple[AtomRecord, bool, float]:
        """Entail the triplet against each kept chunk; build + log its support set S."""
        kept = self._prefilter.select_k(triplet.text, chunks, self._k)
        support: set[str] = set()
        docs: set[str] = set()
        best_conf = 0.0
        saw_contradiction = False
        for chunk in kept:
            res = self._grounding_provider.entails(chunk.text, triplet.text)
            best_conf = max(best_conf, res.raw_score)
            if res.label == "E":
                support.add(chunk.id)
                docs.add(_doc_key(chunk))
            elif res.label == "C":
                saw_contradiction = True
        supported = bool(support)
        # aggregate triplet label: E if any chunk entails, else C if any contradicts, else N
        label = "E" if supported else ("C" if saw_contradiction else "N")
        atom = self._logger.record(
            subject=triplet.id, role="triplet_grounded",
            question="supported by any retrieved chunk?", tier="T2", verdict=supported,
            evidence=f"label={label} score={best_conf:.4f} S={sorted(support)}",
            grader_id="groundedness.triplet", model=self._stamp[0], model_version=self._stamp[1],
            seed=self._seed,
        )
        self._export.add_triplet(atom.id, label, triplet.claim_id, support, docs)
        return atom, supported, best_conf

    def _export_per_claim(self, by_claim: dict[str, list[tuple[bool, float]]]) -> None:
        """Log + publish the per-claim grounded atom (all its triplets E)."""
        for claim_id, items in by_claim.items():
            grounded = all(ok for ok, _ in items)
            atom = self._logger.record(
                subject=claim_id, role="grounded", question="all triplets entailed?", tier="T2",
                verdict=grounded, evidence=f"n_triplets={len(items)}",
                grader_id="groundedness.claim", model=self._claim_stamp[0],
                model_version=self._claim_stamp[1], seed=self._seed,
            )
            self._export.set(claim_id, atom, [conf for _, conf in items])
