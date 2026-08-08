"""Shared groundedness export (§1 → §3/§4/accuracy hand-off).

Groundedness runs one **per-chunk support pass** and publishes the artifact the
whole Evidence category derives from: per triplet, the **support set** ``S`` (the
chunk-ids that entail it); aggregated per claim (support chunk-ids + distinct
source documents) and answer-wide. Accuracy imports the per-claim grounded atom;
source_quality reads ``supports``/``corroboration`` off ``S`` (no new NLI);
source_attribution intersects the cited set ``C`` with ``S``. Confidences feed §5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rq_eval.contracts import AtomRecord


@dataclass(frozen=True, slots=True)
class _Entry:
    atom: AtomRecord
    confidences: list[float]


@dataclass
class _Support:
    """A claim's aggregated support set (union over its triplets)."""

    chunks: set[str] = field(default_factory=set)  # supporting chunk-ids
    docs: set[str] = field(default_factory=set)  # distinct source-document keys


class GroundednessExport:
    """Support-set ``S`` + per-claim grounded atom, written by §1, read by §3/§4/accuracy."""

    def __init__(self) -> None:
        """Start empty; groundedness populates one entry per claim + support sets."""
        self._data: dict[str, _Entry] = {}
        self._triplet_atom_ids: list[str] = []
        self._label_counts: dict[str, int] = {"E": 0, "N": 0, "C": 0}
        self._support: dict[str, _Support] = {}  # claim_id -> aggregated S
        self._answer = _Support()  # answer-wide S (union over all triplets)

    def add_triplet(
        self,
        atom_id: str,
        label: str,
        claim_id: str = "",
        support_chunks: set[str] | None = None,
        support_docs: set[str] | None = None,
    ) -> None:
        """Record a per-triplet atom id + aggregate label, folding ``S`` into the claim + answer."""
        self._triplet_atom_ids.append(atom_id)
        self._label_counts[label] = self._label_counts.get(label, 0) + 1
        chunks = support_chunks or set()
        docs = support_docs or set()
        if claim_id:
            agg = self._support.setdefault(claim_id, _Support())
            agg.chunks |= chunks
            agg.docs |= docs
        self._answer.chunks |= chunks
        self._answer.docs |= docs

    def claim_supported(self, claim_id: str) -> bool:
        """True iff the claim's support set is non-empty (``S ≠ ∅``)."""
        s = self._support.get(claim_id)
        return bool(s and s.chunks)

    def claim_support_docs(self, claim_id: str) -> set[str]:
        """Distinct source documents supporting the claim (corroboration count)."""
        s = self._support.get(claim_id)
        return set(s.docs) if s else set()

    def claim_support_chunks(self, claim_id: str) -> set[str]:
        """Supporting chunk-ids for the claim (the set attribution intersects with ``C``)."""
        s = self._support.get(claim_id)
        return set(s.chunks) if s else set()

    def answer_supported(self) -> bool:
        """True iff any triplet in the answer is supported (answer-wide ``S ≠ ∅``)."""
        return bool(self._answer.chunks)

    def answer_support_docs(self) -> set[str]:
        """Distinct source documents supporting the answer (answer-wide corroboration)."""
        return set(self._answer.docs)

    def triplet_atom_ids(self) -> list[str]:
        """All per-triplet grounding atom ids (hallucination's score atoms)."""
        return list(self._triplet_atom_ids)

    def label_counts(self) -> dict[str, int]:
        """E/N/C counts across all triplets (Neutral vs Contradiction split)."""
        return dict(self._label_counts)

    def set(self, claim_id: str, atom: AtomRecord, confidences: list[float]) -> None:
        """Publish the grounded atom + triplet confidences for ``claim_id``."""
        self._data[claim_id] = _Entry(atom=atom, confidences=confidences)

    def has(self, claim_id: str) -> bool:
        """True iff a grounded atom was published for ``claim_id``."""
        return claim_id in self._data

    def atom(self, claim_id: str) -> AtomRecord:
        """Return the grounded atom groundedness logged for ``claim_id``."""
        return self._data[claim_id].atom

    def grounded(self, claim_id: str) -> bool:
        """Return the published grounded verdict for ``claim_id``."""
        return self._data[claim_id].atom.verdict

    def confidences(self, claim_id: str) -> list[float]:
        """Return the per-triplet confidences for ``claim_id``."""
        return self._data[claim_id].confidences

    def all_confidences(self) -> dict[str, list[float]]:
        """Return every claim's triplet confidences (for the conformal layer)."""
        return {cid: e.confidences for cid, e in self._data.items()}
