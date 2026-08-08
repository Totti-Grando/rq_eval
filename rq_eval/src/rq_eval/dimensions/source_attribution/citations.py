"""§4 step 1 — cited-set resolution (explicit regex + implicit scope) [T1].

Each claim's **cited set** ``C`` is the sources the author pointed at. Explicit
pointers (``[chunk-1]``, inline ids) are parsed by regex. Implicit citations —
where a claim inherits a source introduced earlier ("according to the 10-K, …")
until a new source appears — are recovered by **scope propagation** that *proposes*
a source, then **confirms it against the support set** ``S`` (accepted only if the
scoped source actually entails the claim). Structure proposes, entailment disposes.
"""

from __future__ import annotations

import re

_CITE = re.compile(r"\[([^\]]+)\]")


def resolve_explicit(text: str, retrieved: set[str]) -> set[str]:
    """Explicit cited ids parsed from ``text`` that are in the retrieved set."""
    return {m for m in _CITE.findall(text) if m in retrieved}


class ScopePropagator:
    """Stateful implicit-citation scope over claims in answer order.

    The last explicitly-cited source governs following uncited claims until a new
    explicit cite appears; a propagated (implicit) source is accepted only when it
    is in the claim's support set ``S`` (confirm-in-``S``).
    """

    def __init__(self, retrieved: set[str]) -> None:
        """Track the retrieved id set; scope starts empty."""
        self._retrieved = retrieved
        self._scope: set[str] = set()

    def cited_for(self, text: str, support_chunks: set[str]) -> tuple[set[str], str]:
        """Return ``(C, tag)`` for a claim: explicit ids, else scope confirmed in ``S``.

        ``tag`` is ``explicit`` when the claim carries its own pointer,
        ``implicit-confirmed`` when an inherited source is confirmed by ``S``,
        else ``none``.
        """
        explicit = resolve_explicit(text, self._retrieved)
        if explicit:
            self._scope = explicit
            return explicit, "explicit"
        implicit = self._scope & support_chunks  # confirm the scoped source in S
        return (implicit, "implicit-confirmed") if implicit else (set(), "none")
