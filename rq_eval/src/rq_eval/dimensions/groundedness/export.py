"""Shared groundedness export (§1 → §1-accuracy hand-off).

Groundedness computes the per-claim ``grounded?`` boolean (all the claim's
triplets labeled Entailment) **once** and publishes the exact atom + the
per-triplet confidences here. Accuracy imports the atom (never recomputes);
the confidences feed the conformal layer (§5).
"""

from __future__ import annotations

from dataclasses import dataclass

from rq_eval.contracts import AtomRecord


@dataclass(frozen=True, slots=True)
class _Entry:
    atom: AtomRecord
    confidences: list[float]


class GroundednessExport:
    """Per-claim grounded atom + triplet confidences, written by §1, read by §1-accuracy."""

    def __init__(self) -> None:
        """Start empty; groundedness populates one entry per claim."""
        self._data: dict[str, _Entry] = {}

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
