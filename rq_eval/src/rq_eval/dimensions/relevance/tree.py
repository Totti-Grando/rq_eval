"""§3 relevance — the support tree (iterative reachability from anchors) [code].

From the anchors, cycle through the remaining claims attaching any with a
confirmed support-path to something already in the tree, to a fixpoint (this
handles premise→premise→conclusion *chains*). A claim ``c`` attaches to a tree
member ``m`` when a confirmed edge ``c → m`` exists (``c`` is a premise for ``m``).
Depth from the anchor is the **relevance grade** (direct anchor = 0, its premise
= 1, a distant premise = 2+); depth is bounded by ``max_hops`` so a long weak
chain cannot launder an orphan into relevance, and relevance weight decays with
depth (``depth_decay ** depth``).
"""

from __future__ import annotations

from collections import defaultdict

from rq_eval.dimensions.relevance.edges import Edge


class SupportTree:
    """[code] Reachability from anchors over confirmed premise→conclusion edges."""

    def __init__(self, max_hops: int, depth_decay: float) -> None:
        """Store the hop bound and the per-hop relevance decay factor."""
        self._max_hops = max_hops
        self._depth_decay = depth_decay

    def build(self, anchor_ids: set[str], edges: list[Edge]) -> dict[str, int]:
        """Return ``{claim_id: depth}`` for every claim reachable from an anchor.

        Anchors are depth 0. A claim attaches at ``depth(m) + 1`` when it is a
        premise (edge ``claim → m``) of a tree member ``m`` and it is not already
        placed; growth stops at ``max_hops``.
        """
        premises_of: dict[str, set[str]] = defaultdict(set)
        for e in edges:
            if e.src != e.dst:
                premises_of[e.dst].add(e.src)

        depth: dict[str, int] = {a: 0 for a in anchor_ids}
        frontier = set(anchor_ids)
        hop = 0
        while frontier and hop < self._max_hops:
            hop += 1
            nxt: set[str] = set()
            for member in frontier:
                for premise in premises_of.get(member, ()):
                    if premise not in depth:
                        depth[premise] = hop
                        nxt.add(premise)
            frontier = nxt
        return depth

    def relevance_weight(self, depth: int) -> float:
        """Depth-decayed relevance grade: ``depth_decay ** depth`` (anchor = 1.0)."""
        return self._depth_decay ** depth
