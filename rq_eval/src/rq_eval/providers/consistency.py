"""Forward-declared cross-category interface — the Reasoning `ConsistencyProvider`.

Relevance's orphan-resolution pass (§3) produces two outputs owned by a dimension
in the *Reasoning* category, which is designed separately. Until it exists these
are typed **stubs with safe defaults** (the same pattern as accuracy's imported
`SourceQualityProvider`/`AttributionProvider`), so relevance is independently
buildable and swaps cleanly when Reasoning lands — with no change to relevance:

* ``edge_sound`` — relevance confirms an edge *exists* (A entails B); whether a
  *stated* inference is **valid** is Reasoning's job. Default ``True`` (+ a
  flag-for-review), so relevance never silently penalizes reasoning it isn't
  built to judge.
* ``route_contradiction`` — accepts a stranded orphan that **contradicts** an
  anchor for scoring by logical_consistency + completeness. Default returns a
  receipt marking it routed; no-op downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RouteReceipt:
    """Result of routing a stranded contradiction to the Reasoning category.

    ``routed`` records that the item was accepted for downstream scoring;
    ``reason`` is the human-readable note the caller logs as an atom. The stub
    performs no downstream work — the receipt is the whole effect.
    """

    routed: bool
    reason: str


class ConsistencyProvider(ABC):
    """[Reasoning, forward-declared] Edge soundness + contradiction routing."""

    @abstractmethod
    def edge_sound(self, premise: str, conclusion: str) -> bool:
        """Is the *stated* premise→conclusion inference valid? (not just present)."""

    @abstractmethod
    def route_contradiction(self, claim: str, anchor: str) -> RouteReceipt:
        """Route a stranded orphan that contradicts ``anchor`` for scoring."""


class StubConsistencyProvider(ConsistencyProvider):
    """Default stub: edges are assumed sound; contradictions are routed, no-op."""

    def edge_sound(self, premise: str, conclusion: str) -> bool:
        """Assume sound (flag-for-review) — never penalize on unbuilt reasoning."""
        return True

    def route_contradiction(self, claim: str, anchor: str) -> RouteReceipt:
        """Accept the routed contradiction; the receipt is the only effect."""
        return RouteReceipt(routed=True, reason="stub: routed to Reasoning (no-op downstream)")
