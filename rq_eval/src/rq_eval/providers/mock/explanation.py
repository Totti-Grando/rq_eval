"""Mock explanation judge — deterministic templated summary (R1).

Read-only: turns finished results + atoms into prose. Never emits a verdict,
writes no atom, and is never read by a formula (design §0.5).
"""

from __future__ import annotations

from rq_eval.contracts import AtomRecord, DimensionResult
from rq_eval.providers.base import ExplanationJudge


class MockExplanationJudge(ExplanationJudge):
    """Deterministic templated run summary for offline use."""

    def summarize(self, results: dict[str, DimensionResult], atoms: list[AtomRecord]) -> str:
        """Render a fixed-format one-line-per-dimension summary."""
        lines = [
            f"{name}: {r.score:.3f} [{r.band}]{' (abstained)' if r.abstained else ''}"
            for name, r in sorted(results.items())
        ]
        return f"Summary of {len(results)} dimensions over {len(atoms)} atoms:\n" + "\n".join(lines)
