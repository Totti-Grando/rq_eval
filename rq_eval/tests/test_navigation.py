"""Navigability guard (addendum §2/§3) — the map cannot rot.

Checks, over the current tree:
  1. every ``src/rq_eval`` subfolder containing ``.py`` files has a README.md;
  2. every such subfolder is listed in ARCHITECTURE.md's section->folder table;
  3. every ``src/`` path the table names actually exists.

Grows automatically as folders land; stays green as long as each phase writes
its README and updates the table (addendum requirement).
"""

from __future__ import annotations

import re
from pathlib import Path

from rq_eval.config import project_root

ROOT = project_root()
SRC_PKG = ROOT / "src" / "rq_eval"


def _code_subfolders() -> list[Path]:
    """Subfolders strictly under src/rq_eval that contain at least one .py."""
    out: list[Path] = []
    for d in SRC_PKG.rglob("*"):
        if d.is_dir() and d.name != "__pycache__" and any(d.glob("*.py")):
            out.append(d)
    return out


def _table_src_paths() -> set[str]:
    text = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    # backtick-wrapped paths beginning with src/
    return set(re.findall(r"`(src/[^`]+)`", text))


def test_every_code_subfolder_has_readme() -> None:
    missing = [
        str(d.relative_to(ROOT)) for d in _code_subfolders() if not (d / "README.md").exists()
    ]
    assert not missing, f"src subfolders missing README.md: {missing}"


def test_every_code_subfolder_is_in_architecture_table() -> None:
    listed = _table_src_paths()
    absent = [
        p
        for d in _code_subfolders()
        if (p := str(d.relative_to(ROOT)).replace("\\", "/")) not in listed
    ]
    assert not absent, f"src subfolders missing from ARCHITECTURE.md table: {absent}"


def test_table_src_paths_exist() -> None:
    missing = [p for p in _table_src_paths() if not (ROOT / p).exists()]
    assert not missing, f"ARCHITECTURE.md table names paths that don't exist: {missing}"
