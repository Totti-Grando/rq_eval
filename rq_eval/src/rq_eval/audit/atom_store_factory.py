"""Atom-store construction from config (§0.5.2).

The single place that reads ``paths.atom_log_backend`` + ``paths.atom_log`` and
returns the configured append-only store. Mirrors ProviderFactory's pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rq_eval.audit.atom_store import AtomStore

if TYPE_CHECKING:
    from rq_eval.config import Config


class AtomStoreFactory:
    """Builds the config-selected :class:`AtomStore` (jsonl or sqlite)."""

    def __init__(self, cfg: Config) -> None:
        """Store config; construction is deferred to :meth:`build`."""
        self._cfg = cfg

    def build(self, path: Path | None = None) -> AtomStore:
        """Return the configured store, defaulting to ``paths.atom_log``."""
        target = path if path is not None else self._cfg.resolve(self._cfg.paths.atom_log)
        backend = self._cfg.paths.atom_log_backend
        if backend == "sqlite":
            from rq_eval.audit.sqlite_atom_store import SqliteAtomStore

            return SqliteAtomStore(target)
        from rq_eval.audit.jsonl_atom_store import JsonlAtomStore

        return JsonlAtomStore(target)
