"""§2 step 2 — fabricated-citation / reference-existence gate [T1].

Fully deterministic and it GATES. A citation exists iff:
* a plain chunk-id is a member of the retrieved set (set-membership), or
* a URL / DOI resolves via the :class:`ResolverProvider`.
Any fabricated citation → gate FAIL. Orthogonal to whether the source *supports*
the claim (that is groundedness/attribution).
"""

from __future__ import annotations

from dataclasses import dataclass

from rq_eval.audit.atom_logger import AtomLogger
from rq_eval.contracts import AtomRecord
from rq_eval.providers.base import ResolverProvider


@dataclass(frozen=True, slots=True)
class FabricationResult:
    """Gate outcome: whether it failed + the per-citation existence atoms."""

    gate_failed: bool
    atoms: list[AtomRecord]


class FabricationGate:
    """[T1] Checks every citation exists; any fabrication fails the gate."""

    def __init__(self, resolver: ResolverProvider, logger: AtomLogger) -> None:
        """Inject the reference resolver and the atom logger."""
        self._resolver = resolver
        self._logger = logger

    def check(self, citations: list[str], retrieved_ids: set[str]) -> FabricationResult:
        """Return the gate result over the distinct ``citations``."""
        atoms: list[AtomRecord] = []
        failed = False
        for citation in sorted(set(citations)):
            exists = self._exists(citation, retrieved_ids)
            failed = failed or not exists
            atoms.append(
                self._logger.record(
                    subject=f"citation:{citation}", role="fabrication",
                    question="citation exists?", tier="T1", verdict=exists,
                    evidence=f"citation={citation!r}", grader_id="hallucination.fabrication",
                    model="code", model_version="rq_eval",
                )
            )
        return FabricationResult(gate_failed=failed, atoms=atoms)

    def _exists(self, citation: str, retrieved_ids: set[str]) -> bool:
        ref = citation.strip()
        if ref.startswith(("http://", "https://", "10.", "doi:")):
            return self._resolver.resolve(ref)
        return ref in retrieved_ids  # a chunk-id must be in the retrieved set
