"""SQLite append-only atom store (§0.5.2)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rq_eval.audit.atom_store import AtomStore
from rq_eval.contracts import AtomRecord


class SqliteAtomStore(AtomStore):
    """Atoms in a single ``atoms(seq, id, data)`` table; insertion-ordered."""

    def __init__(self, path: Path) -> None:
        """Open/create the database and ensure the schema exists."""
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS atoms ("
                "seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT, data TEXT)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def append(self, atom: AtomRecord) -> None:
        """Insert one atom row (append-only)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO atoms (id, data) VALUES (?, ?)",
                (atom.id, atom.model_dump_json()),
            )

    def all(self) -> list[AtomRecord]:
        """Return every atom ordered by insertion sequence."""
        with self._connect() as conn:
            rows = conn.execute("SELECT data FROM atoms ORDER BY seq").fetchall()
        return [AtomRecord.model_validate_json(row[0]) for row in rows]
