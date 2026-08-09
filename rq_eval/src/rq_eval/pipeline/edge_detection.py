"""§0.3 — support-edge detection over the shared claim graph [T1 propose + T2 confirm].

The hard part is "which claims did this one build on." The literature's answer,
adopted here: **backward premise identification from each conclusion**, not forward
subset enumeration. For each target claim, candidate parents are restricted to
**earlier claims only** — which enforces acyclicity by construction (edges always
point earlier → later). For each target the pipeline runs three signals:

* **discourse markers** `[T1]` propose that the claim has premises;
* **topical restriction** `[T1]` limits candidate parents to claims sharing terms
  (a lexical proxy for the coref + embedding clustering the live path uses);
* **entailment confirmation** `[T2]` — ``entails(⋀parents, claim) ≥ edge_tau``
  confirms the edge, reduced to a **minimal-complete premise set** by greedy
  removal (drop any parent whose removal keeps the entailment).

**Numeric convergence** `[T1]` is near-deterministic: if a claim's figure is a
sum/difference of two earlier figures, those are its parents precisely. Edges are
earlier → later, so the graph is a DAG; a ``networkx`` cycle-cut is a safety net.
"""

from __future__ import annotations

import networkx as nx

from rq_eval.contracts import Claim
from rq_eval.graders.t1 import T1Tools
from rq_eval.pipeline.claim_graph import ClaimGraph
from rq_eval.providers.base import GroundingProvider


class EdgeDetector:
    """[T1 propose + T2 confirm] Adds confirmed support edges to the shared graph."""

    def __init__(
        self,
        grounding: GroundingProvider,
        t1: T1Tools,
        edge_tau: float,
        topical_min: float,
        numeric_tolerance: float,
    ) -> None:
        """Inject the NLI verifier + T1 tools; store the edge / topical / numeric thresholds."""
        self._grounding = grounding
        self._t1 = t1
        self._edge_tau = edge_tau
        self._topical_min = topical_min
        self._numeric_tolerance = numeric_tolerance

    def detect(self, claims: list[Claim], graph: ClaimGraph) -> None:
        """Populate ``graph`` with confirmed premise → conclusion support edges."""
        for i, target in enumerate(claims):
            earlier = claims[:i]  # candidates restricted to earlier nodes -> acyclic
            for parent in self._premises(target, earlier):
                graph.add_edge(parent.id, target.id, "supports", confirmed_by="edge_detection")
        self._cut_cycles(graph)

    def _premises(self, target: Claim, candidates: list[Claim]) -> list[Claim]:
        numeric = self._numeric_parents(target, candidates)
        if numeric:
            return numeric
        topical = [
            c for c in candidates
            if self._t1.key_term_overlap(target.text, c.text) >= self._topical_min
        ]
        if not topical or not self._confirms(topical, target):
            return []
        return self._minimal(topical, target)

    def _confirms(self, parents: list[Claim], target: Claim) -> bool:
        premise = " ".join(p.text for p in parents)
        res = self._grounding.entails(premise, target.text)
        return res.raw_score >= self._edge_tau and res.label != "C"

    def _minimal(self, parents: list[Claim], target: Claim) -> list[Claim]:
        """Greedy reduction to a minimal-complete premise set (ReasoningFlow)."""
        minimal = list(parents)
        for p in list(minimal):
            trial = [q for q in minimal if q.id != p.id]
            if trial and self._confirms(trial, target):
                minimal = trial
        return minimal

    def _numeric_parents(self, target: Claim, candidates: list[Claim]) -> list[Claim]:
        """Parents identified by arithmetic provenance (profit = revenue − costs)."""
        n_t = self._t1.extract_number(target.text)
        if n_t is None:
            return []
        numbered = [
            (c, n) for c in candidates if (n := self._t1.extract_number(c.text)) is not None
        ]
        for a_idx, (ca, na) in enumerate(numbered):
            for cb, nb in numbered[a_idx + 1 :]:
                if any(self._approx(n_t, v) for v in (na + nb, na - nb, nb - na)):
                    return [ca, cb]
        return []

    def _approx(self, x: float, y: float) -> bool:
        return abs(x - y) <= self._numeric_tolerance * max(abs(x), abs(y), 1.0)

    @staticmethod
    def _cut_cycles(graph: ClaimGraph) -> None:
        """Safety net: cut any cycle (none should exist given earlier-only edges)."""
        g = graph.graph
        while not nx.is_directed_acyclic_graph(g):
            cycle = nx.find_cycle(g)
            g.remove_edge(*cycle[0][:2])
