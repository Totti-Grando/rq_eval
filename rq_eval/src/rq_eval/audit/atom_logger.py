"""AtomLogger — create + append an atom in one call (§0.5.2).

Graders and dimensions use this to record every yes/no check. It stamps the
timestamp from the injected clock and delegates id derivation to
:meth:`AtomRecord.create`, then appends to the store and returns the atom so the
caller can collect its id for the :class:`DimensionResult`.
"""

from __future__ import annotations

from rq_eval.audit.atom_store import AtomStore
from rq_eval.audit.clock import Clock, SystemClock
from rq_eval.contracts import AtomRecord, Tier


class AtomLogger:
    """Records boolean checks to an append-only store."""

    def __init__(self, store: AtomStore, clock: Clock | None = None) -> None:
        """Inject the store and (optionally) a clock; defaults to SystemClock."""
        self._store = store
        self._clock = clock or SystemClock()

    def record(
        self,
        *,
        subject: str,
        role: str,
        question: str,
        tier: Tier,
        verdict: bool,
        weight: float = 1.0,
        evidence: str = "",
        grader_id: str = "",
        model: str = "",
        model_version: str = "",
        seed: int | None = None,
    ) -> AtomRecord:
        """Create the atom (timestamp from the clock), append it, return it."""
        atom = AtomRecord.create(
            subject=subject,
            role=role,
            question=question,
            tier=tier,
            verdict=verdict,
            weight=weight,
            evidence=evidence,
            grader_id=grader_id,
            model=model,
            model_version=model_version,
            seed=seed,
            timestamp=self._clock.now(),
        )
        self._store.append(atom)
        return atom
