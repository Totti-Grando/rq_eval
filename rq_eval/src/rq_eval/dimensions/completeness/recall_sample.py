"""§2 — human recall-sample miss-rate (completeness's honest error bar).

A periodic human recall sample backstops the unknown-unknown: a vital fact no
pipeline surfaced. Each labeled row names a should-contain ``fact`` for questions
matching ``question_match``. The miss-rate — the fraction of sampled facts the
frozen unit set failed to surface — is **reported on the result**, not hidden.
Deterministic + offline: "surfaced" is a lexical key-term coverage check.
"""

from __future__ import annotations

import json
from pathlib import Path

from rq_eval.graders.t1 import T1Tools

_SURFACE_MIN = 0.5  # a fact is "surfaced" if a unit covers >= this share of its terms


class RecallSample:
    """Loads the labeled should-contain sample and computes a miss-rate."""

    def __init__(self, path: Path | None) -> None:
        """Load ``{question_match, fact}`` rows from a JSONL file (empty if None)."""
        self._t1 = T1Tools()
        self._rows: list[dict[str, str]] = []
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self._rows.append(json.loads(line))

    def miss_rate(self, question: str, unit_texts: list[str]) -> float | None:
        """Fraction of this question's sampled facts not surfaced by any unit.

        Returns ``None`` when no sampled fact applies to the question (no error
        bar to report), so callers only stamp a real measurement.
        """
        q = question.lower()
        rows = [r for r in self._rows if r["question_match"].lower() in q]
        if not rows:
            return None
        misses = sum(
            1
            for r in rows
            if not any(self._t1.key_term_overlap(r["fact"], u) >= _SURFACE_MIN for u in unit_texts)
        )
        return misses / len(rows)
