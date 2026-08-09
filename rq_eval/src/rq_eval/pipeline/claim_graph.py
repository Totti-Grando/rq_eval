"""§0.3 — the single shared claim graph (build once, read by many projections).

Nodes are claims; directed edges are typed dependencies. This is **shared
infrastructure, not a scorer**: atomic checks (groundedness §1) ignore it; only
structural metrics read *projections* of it — accuracy reads support edges as
**derivation** (does a chain reach a true axiom?), relevance reads the *same*
edges as **reachability** (does a chain reach a question anchor?). Built once by
``ClaimGraphBuilder``; no dimension rebuilds it.

Three claim types (typed at build, mostly ``[T1]``): **independent** (a complete,
self-groundable proposition — the default); **inference-dependent** (a well-formed
claim with an empty support set ``S`` — a byproduct of §1, no separate detector);
**indexical-dependent** (an *incomplete* proposition with free deictic slots,
flagged by ``T1Tools``) — bound to a sibling filler before scoring, or flagged
``context-incomplete`` and routed out of grounding.

Backed by a ``networkx.DiGraph``; edges are added in the edge-detection phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx

from rq_eval.contracts import Claim
from rq_eval.dimensions.groundedness.export import GroundednessExport
from rq_eval.graders.t1 import T1Tools
from rq_eval.providers.base import NlpProvider

ClaimType = Literal["independent", "inference", "indexical"]
EdgeType = Literal["supports", "derives", "binds", "contradicts"]


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One claim node: its type, the (possibly bound) text, and completeness flag."""

    claim: Claim
    ctype: ClaimType
    bound_text: str  # indexical claims are bound to their filler; else == claim.text
    context_incomplete: bool  # indexical but unbindable -> routed out of grounding


class ClaimGraph:
    """A directed claim graph over a ``networkx.DiGraph`` (nodes keyed by claim id)."""

    def __init__(self) -> None:
        """Start empty; the builder adds nodes and (later) typed edges."""
        self._g: nx.DiGraph[str] = nx.DiGraph()

    def add_node(self, node: GraphNode) -> None:
        """Register a claim node (keyed by claim id)."""
        self._g.add_node(node.claim.id, data=node)

    def add_edge(self, src: str, dst: str, etype: EdgeType, confirmed_by: str = "") -> None:
        """Add a typed dependency edge ``src → dst`` (premise → conclusion)."""
        self._g.add_edge(src, dst, etype=etype, confirmed_by=confirmed_by)

    def node(self, claim_id: str) -> GraphNode:
        """Return the :class:`GraphNode` for ``claim_id``."""
        data: GraphNode = self._g.nodes[claim_id]["data"]
        return data

    def nodes(self) -> list[GraphNode]:
        """All claim nodes, in insertion order."""
        return [self._g.nodes[n]["data"] for n in self._g.nodes]

    def edges(self) -> list[tuple[str, str, EdgeType]]:
        """All typed edges as ``(src, dst, etype)`` triples."""
        return [(u, v, d["etype"]) for u, v, d in self._g.edges(data=True)]

    def parents(self, claim_id: str) -> list[str]:
        """Premise nodes with a confirmed edge into ``claim_id``."""
        return list(self._g.predecessors(claim_id))

    def type_counts(self) -> dict[str, int]:
        """Count of nodes by claim type (a diagnostic)."""
        counts: dict[str, int] = {}
        for node in self.nodes():
            counts[node.ctype] = counts.get(node.ctype, 0) + 1
        return counts

    @property
    def graph(self) -> nx.DiGraph[str]:
        """The underlying ``networkx`` graph (for the viz projection, read-only use)."""
        return self._g


class ClaimGraphBuilder:
    """[T1 + §1 byproduct] Types every claim and binds indexicals; adds no edges."""

    def __init__(self, t1: T1Tools, nlp: NlpProvider, grounded: GroundednessExport) -> None:
        """Inject the T1 toolbox, the NLP provider (coref), and the §1 support set."""
        self._t1 = t1
        self._nlp = nlp
        self._grounded = grounded

    def build(self, claims: list[Claim]) -> ClaimGraph:
        """Return the typed node graph (edges are added by edge detection, §0.3/G4)."""
        graph = ClaimGraph()
        for claim in claims:
            graph.add_node(self._node(claim, claims))
        return graph

    def _node(self, claim: Claim, siblings: list[Claim]) -> GraphNode:
        if self._t1.is_indexical(claim.text):
            bound, incomplete = self._bind(claim, siblings)
            return GraphNode(claim, "indexical", bound, incomplete)
        # inference-dependent: well-formed but nothing directly supports it (empty S)
        if self._grounded.has(claim.id) and not self._grounded.claim_supported(claim.id):
            return GraphNode(claim, "inference", claim.text, context_incomplete=False)
        return GraphNode(claim, "independent", claim.text, context_incomplete=False)

    def _bind(self, claim: Claim, siblings: list[Claim]) -> tuple[str, bool]:
        """Fill the free deictic slot from the nearest sibling supplying a filler.

        Returns ``(bound_text, context_incomplete)``. A binding is accepted only if
        the completed claim contains the filler; if no sibling supplies one the
        claim is flagged ``context-incomplete`` (reported, not guessed).
        """
        for sib in siblings:
            if sib.id == claim.id:
                continue
            filler = self._t1.find_filler(sib.text)
            if filler:
                bound = f"{claim.text} [{filler}]"
                if filler in bound:  # verify the filler is carried into the bound claim
                    return bound, False
        return claim.text, True
