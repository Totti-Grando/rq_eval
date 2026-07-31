"""§0 (Evidence & Truthfulness) — claims → claim-triplets [T3-gen].

RefChecker-style subject-predicate-object decomposition: each cached Claim is
split into triplets checked separately by the groundedness/attribution verifiers
(triplet-level checking beats sentence-level by 4–9 pts). Pinned by
``pins.triplet_extractor_version``; stability measured over re-runs. The mock
generator's ``[[triplets]]`` tag is the deterministic parse-based splitter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rq_eval.contracts import Claim, Triplet
from rq_eval.providers.base import GeneratorProvider

if TYPE_CHECKING:
    from rq_eval.config import Config

# Pinned prompt (mock: [[triplets]] parse-splitter on the payload; live: strip
# tag + unwrap {{ }} -> a real RefChecker-style extraction instruction).
_PROMPT = (
    "[[triplets]] Decompose into subject-predicate-object triplets, "
    "one per line as 'subject | predicate | object'. {{ {claim} }}"
)


class ClaimTripletExtractor:
    """[T3-gen] Decomposes claims into provenance-carrying triplets."""

    def __init__(self, generator: GeneratorProvider, cfg: Config) -> None:
        """Inject the generator; pin the extractor version + seed from config."""
        self._generator = generator
        self._version = cfg.pins.triplet_extractor_version
        self._seed = cfg.seeds.judge

    @property
    def version(self) -> str:
        """The pinned triplet-extractor version."""
        return self._version

    def extract(self, claim: Claim) -> list[Triplet]:
        """Return ≥1 triplet for ``claim``, each carrying its provenance."""
        result = self._generator.generate(
            _PROMPT.replace("{claim}", claim.text), seed=self._seed
        )
        triplets = [self._parse(claim, item) for item in result.items if item.strip()]
        if not triplets:  # always ≥1 triplet per claim
            triplets = [
                Triplet.create(
                    claim_id=claim.id, subject=claim.text, predicate="", obj="",
                    citation=claim.citation, source_pointer=claim.source_sentence,
                )
            ]
        return triplets

    def extract_all(self, claims: list[Claim]) -> list[Triplet]:
        """Flatten :meth:`extract` over many claims."""
        return [t for claim in claims for t in self.extract(claim)]

    def _parse(self, claim: Claim, item: str) -> Triplet:
        parts = [p.strip() for p in item.split("|")]
        subject = parts[0] if parts else item.strip()
        predicate = parts[1] if len(parts) > 1 else ""
        obj = parts[2] if len(parts) > 2 else ""
        return Triplet.create(
            claim_id=claim.id, subject=subject, predicate=predicate, obj=obj,
            citation=claim.citation, source_pointer=claim.source_sentence,
        )


class TripletStabilityHarness:
    """Re-runs triplet extraction and reports triplet-id set agreement ∈ [0, 1]."""

    def __init__(self, extractor: ClaimTripletExtractor) -> None:
        """Inject the extractor to re-run."""
        self._extractor = extractor

    def measure(self, claims: list[Claim], runs: int) -> float:
        """Agreement = |∩ triplet-id sets| / |∪| over ``runs`` passes (1.0 if empty)."""
        sets = [
            {t.id for t in self._extractor.extract_all(claims)} for _ in range(max(1, runs))
        ]
        union = set().union(*sets) if sets else set()
        if not union:
            return 1.0
        intersection = set(sets[0]).intersection(*sets[1:]) if len(sets) > 1 else sets[0]
        return len(intersection) / len(union)
