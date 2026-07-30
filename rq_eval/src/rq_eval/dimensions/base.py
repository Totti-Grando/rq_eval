"""Dimension base class (addendum §1).

Every dimension is a class implementing this interface. Providers, graders, the
atom logger, the cached claims, and any shared cross-dimension state are injected
via ``__init__`` (dependency injection) — a dimension never constructs them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rq_eval.contracts import DimensionResult, EvalInput


class Dimension(ABC):
    """Scores one Response-Quality dimension from an :class:`EvalInput`."""

    name: str

    @abstractmethod
    def evaluate(self, eval_input: EvalInput) -> DimensionResult:
        """Compute the dimension's :class:`DimensionResult` (logs atoms)."""
