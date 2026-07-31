"""Human-readable run report (build order B10).

Renders the four dimension results — scores, bands, CIs, abstentions — plus
pipeline stability and atom counts by tier, from an :class:`EvaluationResult`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rq_eval.runner import EvaluationResult

_ORDER = ("accuracy", "completeness", "relevance", "task_success")


class ReportRenderer:
    """Formats an evaluation run as a readable text block."""

    def render(self, result: EvaluationResult) -> str:
        """Return the multi-line report string."""
        lines = ["Response Quality evaluation"]
        stability = "n/a" if result.stability is None else f"{result.stability:.2f}"
        lines.append(f"  claims: {len(result.claims)}   pipeline stability: {stability}")
        lines.append("  " + "-" * 58)
        for dim in _ORDER:
            r = result.results[dim]
            flag = " ABSTAINED" if r.abstained else ""
            lines.append(
                f"  [{r.band}] {dim:<13} {r.score:.3f}  "
                f"CI[{r.ci_low:.2f},{r.ci_high:.2f}] n={r.n}{flag}"
            )
            if r.extra:
                extra = ", ".join(f"{k}={v:.3f}" for k, v in sorted(r.extra.items()))
                lines.append(f"        {extra}")
        lines.append("  " + "-" * 58)
        lines.append("  atoms by tier: " + self._tier_counts(result))
        return "\n".join(lines)

    @staticmethod
    def _tier_counts(result: EvaluationResult) -> str:
        counts: dict[str, int] = {}
        for atom in result.atoms:
            counts[atom.tier] = counts.get(atom.tier, 0) + 1
        return ", ".join(f"{tier}={counts[tier]}" for tier in sorted(counts)) or "none"
