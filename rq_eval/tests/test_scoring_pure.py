"""B5 — scoring/ imports no model/provider/grader code (booleans-only fence).

The scoring layer must be a pure-math island: it may import stdlib, numpy/scipy,
and contracts, but never providers or graders (which touch models).
"""

from __future__ import annotations

import ast

from rq_eval.config import project_root

SCORING = project_root() / "src" / "rq_eval" / "scoring"
FORBIDDEN_PREFIXES = ("rq_eval.providers", "rq_eval.graders", "rq_eval.pipeline", "rq_eval.audit")


def _imports(path: str) -> list[str]:
    tree = ast.parse(path)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_scoring_has_no_model_imports() -> None:
    offenders: list[str] = []
    for py in SCORING.rglob("*.py"):
        for mod in _imports(py.read_text(encoding="utf-8")):
            if mod.startswith(FORBIDDEN_PREFIXES):
                offenders.append(f"{py.name}: imports {mod}")
    assert not offenders, "scoring/ must not import model code:\n" + "\n".join(offenders)
