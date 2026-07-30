"""Shared responsiveness export (§3 → §1 hand-off).

Relevance computes the per-claim responsive boolean **once** and publishes the
exact :class:`AtomRecord` here. Accuracy imports that same atom (verdict + id)
and never recomputes — so accuracy's ``responsive`` term is provably the atom
relevance logged.
"""

from __future__ import annotations

from rq_eval.contracts import AtomRecord


class ResponsivenessExport:
    """Per-claim responsive atom, written by §3, read by §1."""

    def __init__(self) -> None:
        """Start empty; relevance populates one entry per claim."""
        self._data: dict[str, AtomRecord] = {}

    def set(self, claim_id: str, atom: AtomRecord) -> None:
        """Publish the responsive atom for ``claim_id``."""
        self._data[claim_id] = atom

    def has(self, claim_id: str) -> bool:
        """True iff a responsive atom was published for ``claim_id``."""
        return claim_id in self._data

    def atom(self, claim_id: str) -> AtomRecord:
        """Return the responsive atom relevance logged for ``claim_id``."""
        return self._data[claim_id]

    def responsive(self, claim_id: str) -> bool:
        """Return the published responsive verdict for ``claim_id``."""
        return self._data[claim_id].verdict

    def atom_id(self, claim_id: str) -> str:
        """Return the id of the responsive atom for ``claim_id``."""
        return self._data[claim_id].id
