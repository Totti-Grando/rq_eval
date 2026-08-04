"""§3 relevance — entailment-backed support edges [T1 prior + T2 confirm].

An edge ``src → dst`` (``src`` is a *premise* supporting the *conclusion* ``dst``)
holds iff ``src`` entails/supports ``dst`` above ``edge_tau`` — the argumentative
relation-classification task, whose operational signal is textual entailment.
Discourse markers ("because", "tied to") are recorded as a cheap ``[T1]``
candidate prior, but they never make an edge on their own: the entailment
threshold is the gate, which is what keeps the support tree contained.
"""

from __future__ import annotations

from dataclasses import dataclass

from rq_eval.contracts import Claim
from rq_eval.graders.t1 import T1Tools
from rq_eval.providers.base import GroundingProvider


@dataclass(frozen=True, slots=True)
class Edge:
    """A confirmed support edge: ``src`` (premise) entails ``dst`` (conclusion)."""

    src: str  # premise claim id
    dst: str  # conclusion claim id
    raw_score: float
    marker: bool  # a discourse marker proposed this pair (candidate prior only)


class EdgeBuilder:
    """[T1 prior + T2 confirm] Build the confirmed support graph over claims."""

    def __init__(self, grounding: GroundingProvider, t1: T1Tools, edge_tau: float) -> None:
        """Inject the NLI provider, the T1 toolbox, and the edge threshold."""
        self._grounding = grounding
        self._t1 = t1
        self._edge_tau = edge_tau

    def build(self, claims: list[Claim]) -> list[Edge]:
        """Return confirmed edges: ``entails(src, dst).raw_score ≥ edge_tau``.

        Every ordered pair is a candidate; entailment confirms it. A stated
        "because" link whose premise does not actually entail the conclusion
        fails the threshold and never becomes an edge.
        """
        edges: list[Edge] = []
        for src in claims:
            for dst in claims:
                if src.id == dst.id:
                    continue
                res = self._grounding.entails(src.text, dst.text)
                if res.raw_score >= self._edge_tau and res.label != "C":
                    edges.append(
                        Edge(
                            src=src.id, dst=dst.id, raw_score=res.raw_score,
                            marker=self._t1.has_discourse_marker(dst.text),
                        )
                    )
        return edges
