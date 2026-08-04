"""R1 — judge split: ScoringJudge (booleans) vs ExplanationJudge (read-only).

Structural quarantine: the old `JudgeProvider` is gone, and nothing on a scoring
path (scoring/, graders/, dimensions/, the formula registry) references
`ExplanationJudge`. It may only appear in providers/, the runner, and report.
"""

from __future__ import annotations

from rq_eval.config import project_root

SRC = project_root() / "src" / "rq_eval"
# where ExplanationJudge is legitimately allowed to appear:
_ALLOWED = {"providers", "runner.py", "report.py"}


def _rel(path: object) -> str:
    from pathlib import Path

    return str(Path(str(path)).relative_to(SRC)).replace("\\", "/")


def test_judge_provider_name_is_gone() -> None:
    offenders = [
        _rel(p)
        for p in SRC.rglob("*.py")
        if "JudgeProvider" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"legacy JudgeProvider still referenced in: {offenders}"


def test_explanation_judge_absent_from_scoring_paths() -> None:
    offenders: list[str] = []
    for p in SRC.rglob("*.py"):
        rel = _rel(p)
        if rel.split("/")[0] in _ALLOWED or rel in _ALLOWED:
            continue
        if "ExplanationJudge" in p.read_text(encoding="utf-8"):
            offenders.append(rel)
    assert not offenders, f"ExplanationJudge leaked onto a scoring path: {offenders}"


def test_no_formula_imports_explanation_judge() -> None:
    scoring = SRC / "scoring"
    offenders = [
        _rel(p)
        for p in scoring.rglob("*.py")
        if "ExplanationJudge" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"scoring/ must never reference ExplanationJudge: {offenders}"
