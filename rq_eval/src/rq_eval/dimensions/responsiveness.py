"""Shared responsiveness export (§3 → §1 hand-off).

Relevance computes the per-claim responsive boolean **once** and publishes it
here (verdict + the exact atom id). Accuracy imports it and never recomputes —
so accuracy's ``responsive`` term is provably the same atom relevance logged.
"""

from __future__ import annotations


class ResponsivenessExport:
    """Per-claim responsive verdict + atom id, written by §3, read by §1."""

    def __init__(self) -> None:
        """Start empty; relevance populates one entry per claim."""
        self._data: dict[str, tuple[bool, str]] = {}

    def set(self, claim_id: str, responsive: bool, atom_id: str) -> None:
        """Publish the responsive verdict + its atom id for ``claim_id``."""
        self._data[claim_id] = (responsive, atom_id)

    def has(self, claim_id: str) -> bool:
        """True iff a responsive verdict was published for ``claim_id``."""
        return claim_id in self._data

    def responsive(self, claim_id: str) -> bool:
        """Return the published responsive verdict for ``claim_id``."""
        return self._data[claim_id][0]

    def atom_id(self, claim_id: str) -> str:
        """Return the id of the responsive atom relevance logged for ``claim_id``."""
        return self._data[claim_id][1]
