"""Band mapper — score → G/A/R (build order B5).

Bands are policy-set in config (default G ≥ 0.90 / A ≥ 0.75 / R < 0.75).
"""

from __future__ import annotations


class BandMapper:
    """Maps a score in [0, 1] to a band letter using config thresholds."""

    def __init__(self, g: float, a: float) -> None:
        """Store the Green and Amber thresholds (g >= a expected)."""
        self._g = g
        self._a = a

    def band(self, score: float) -> str:
        """Return "G" if score ≥ g, "A" if score ≥ a, else "R"."""
        if score >= self._g:
            return "G"
        if score >= self._a:
            return "A"
        return "R"
