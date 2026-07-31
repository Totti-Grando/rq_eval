"""Verifier interface + context + router (§4 v2 routing table).

Each required outcome is routed to the cheapest verifier that fits its tag. A
verifier decides one boolean (achieved?) and returns the logged
:class:`AtomRecord` (tier reflects the verifier: T1/T2/T3). The judge fires only
on ``adequacy`` outcomes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rq_eval.contracts import AtomRecord
from rq_eval.dimensions.task_success.task_templates import Outcome


@dataclass(frozen=True, slots=True)
class VerifyContext:
    """Inputs a verifier may consult for one evaluation."""

    question: str
    answer: str
    context_text: str


class Verifier(ABC):
    """Decides one outcome and returns its logged atom (role='outcome')."""

    @abstractmethod
    def verify(self, outcome: Outcome, ctx: VerifyContext) -> AtomRecord:
        """Judge whether ``outcome`` is achieved; log + return the atom."""


class VerifierRouter:
    """Maps an outcome's verifier tag to its :class:`Verifier`."""

    def __init__(self, verifiers: dict[str, Verifier]) -> None:
        """Store the tag -> verifier mapping."""
        self._verifiers = verifiers

    def route(self, outcome: Outcome, ctx: VerifyContext) -> AtomRecord:
        """Dispatch ``outcome`` to its tagged verifier (KeyError if unknown)."""
        if outcome.verifier not in self._verifiers:
            raise KeyError(f"no verifier for tag: {outcome.verifier}")
        return self._verifiers[outcome.verifier].verify(outcome, ctx)
