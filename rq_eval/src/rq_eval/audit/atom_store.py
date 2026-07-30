"""Append-only atom store interface (§0.5.2).

The audit log is append-only: atoms are written once and never mutated, so the
record of *what decided each verdict and why* is tamper-evident (the replay
verifier detects any post-hoc change).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rq_eval.contracts import AtomRecord


class AtomStore(ABC):
    """Append-only persistence for :class:`AtomRecord`s."""

    @abstractmethod
    def append(self, atom: AtomRecord) -> None:
        """Persist one atom (append-only; never overwrites existing atoms)."""

    @abstractmethod
    def all(self) -> list[AtomRecord]:
        """Return every stored atom, in insertion order."""

    def get(self, atom_id: str) -> AtomRecord | None:
        """Return the atom with ``atom_id`` (last write wins), or None."""
        found: AtomRecord | None = None
        for atom in self.all():
            if atom.id == atom_id:
                found = atom
        return found

    def by_ids(self, atom_ids: list[str]) -> list[AtomRecord]:
        """Return atoms for ``atom_ids`` (skips ids not present)."""
        index = {atom.id: atom for atom in self.all()}
        return [index[i] for i in atom_ids if i in index]
