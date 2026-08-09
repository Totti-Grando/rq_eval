"""§3 relevance — the support-edge type (edges come from the shared graph).

Relevance does **not** build its own edges: the one shared ``ClaimGraph`` (§0.3)
develops the support edges once (marker-propose → topical-narrow → entailment
confirm, in ``pipeline/edge_detection.py``), and relevance *reads* them as
reachability. This module now only defines the lightweight ``Edge`` view that the
anchor/tree machinery consumes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Edge:
    """A confirmed support edge: ``src`` (premise) entails ``dst`` (conclusion)."""

    src: str  # premise claim id
    dst: str  # conclusion claim id
    raw_score: float
    marker: bool  # a discourse marker proposed this pair (candidate prior only)
