"""Wilson 95% score interval for a proportion (build order B5).

Used to bound recall/precision estimates (§2 completeness, and any dimension
reporting a proportion). Pure closed form — no model, no I/O.
"""

from __future__ import annotations

import math


class WilsonInterval:
    """Two-sided Wilson score interval for a binomial proportion."""

    def __init__(self, z: float = 1.96) -> None:
        """Store the z-score (default 1.96 == 95% two-sided)."""
        self._z = z

    def interval(self, successes: int, n: int) -> tuple[float, float]:
        """Return (low, high) for ``successes``/``n``, clamped to [0, 1].

        center = (p̂ + z²/2n) / (1 + z²/n)
        half   = (z / (1 + z²/n)) · sqrt( p̂(1-p̂)/n + z²/4n² )
        n == 0 yields the uninformative interval (0.0, 1.0).
        """
        if n <= 0:
            return (0.0, 1.0)
        z = self._z
        phat = successes / n
        denom = 1.0 + z * z / n
        center = (phat + z * z / (2 * n)) / denom
        half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
        return (max(0.0, center - half), min(1.0, center + half))
