"""JSONL append-only atom store (§0.5.2)."""

from __future__ import annotations

from pathlib import Path

from rq_eval.audit.atom_store import AtomStore
from rq_eval.contracts import AtomRecord


class JsonlAtomStore(AtomStore):
    """One JSON object per line; append-only, human-inspectable."""

    def __init__(self, path: Path) -> None:
        """Store the log path, creating its parent directory."""
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, atom: AtomRecord) -> None:
        """Append the atom as a single JSON line."""
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(atom.model_dump_json() + "\n")

    def all(self) -> list[AtomRecord]:
        """Read and parse every atom line (empty if the file is absent)."""
        if not self._path.exists():
            return []
        out: list[AtomRecord] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(AtomRecord.model_validate_json(line))
        return out
