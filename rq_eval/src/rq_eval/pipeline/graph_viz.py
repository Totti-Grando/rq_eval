"""§0.3 — a first-class diagnostic render of the resolved claim graph.

Because the resolved graph is already logged (typed nodes + edges + per-node
verdicts), a visualization is a **view over it, not new computation**: axioms as
roots (green), derived nodes green/red by chain survival with the broken step
highlighted, contradiction edges red, orphans floating unconnected. Emitted as an
optional artifact alongside the scores. The canonical artifact is a deterministic
JSON node-link dump (pure code, offline-safe); a ``networkx`` + matplotlib
force-directed PNG is written too **when a drawing backend is importable**.
"""

from __future__ import annotations

import json
from pathlib import Path

from rq_eval.contracts import AtomRecord
from rq_eval.pipeline.claim_graph import ClaimGraph

_COLOR = {"axiom": "green", "derived": "green", "failed": "red"}


class GraphVisualizer:
    """Renders a resolved :class:`ClaimGraph` to a diagnostic artifact (view only)."""

    def status_from_atoms(self, graph: ClaimGraph, atoms: list[AtomRecord]) -> dict[str, str]:
        """Per-node status from accuracy's logged verdicts: axiom / derived / failed."""
        derived = {a.subject for a in atoms if a.role == "derived" and a.verdict}
        axiom_ok = {a.subject for a in atoms if a.role == "axiom" and a.verdict}
        status: dict[str, str] = {}
        for node in graph.nodes():
            cid = node.claim.id
            if cid in axiom_ok:
                status[cid] = "axiom"
            elif cid in derived:
                status[cid] = "derived"
            else:
                status[cid] = "failed"
        return status

    def render(self, graph: ClaimGraph, status: dict[str, str], out_path: Path) -> Path:
        """Write the JSON node-link artifact (+ a PNG if a backend is present); return the path."""
        data = self._node_link(graph, status)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._maybe_png(graph, status, out_path.with_suffix(".png"))
        return out_path

    def _node_link(self, graph: ClaimGraph, status: dict[str, str]) -> dict[str, object]:
        nodes = [
            {
                "id": n.claim.id,
                "type": n.ctype,
                "status": status.get(n.claim.id, "failed"),
                "color": _COLOR.get(status.get(n.claim.id, "failed"), "red"),
                "context_incomplete": n.context_incomplete,
            }
            for n in graph.nodes()
        ]
        links = [
            {"source": s, "target": d, "etype": e,
             "color": "red" if e == "contradicts" else "gray"}
            for s, d, e in graph.edges()
        ]
        return {"nodes": nodes, "links": links}

    def _maybe_png(self, graph: ClaimGraph, status: dict[str, str], png_path: Path) -> None:
        """[lazy] Force-directed PNG when matplotlib is importable; silent no-op otherwise."""
        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
            import networkx as nx  # noqa: PLC0415
        except ImportError:
            return  # headless / offline: the JSON artifact is the render
        g = graph.graph
        colors = [_COLOR.get(status.get(n, "failed"), "red") for n in g.nodes]
        edge_colors = [
            "red" if d.get("etype") == "contradicts" else "gray"
            for _, _, d in g.edges(data=True)
        ]
        nx.draw(
            g, pos=nx.spring_layout(g), node_color=colors, edge_color=edge_colors,
            with_labels=True,
        )
        plt.savefig(png_path)
        plt.close()
