"""Fixture suite with known qualitative outcomes (build order B10).

Small Q/A/context cases used by ``python -m rq_eval.runner`` and the runner
tests, including a planted off-ask answer, a missing-facet answer, and an
explanation-instead-of-fix answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rq_eval.contracts import ContextChunk, EvalInput, Profile

_DRIVERS_Q = "What were the key drivers of the revenue decline?"
_DRIVERS_CTX = (
    "Cost drivers behind the change included higher input costs. "
    "Pricing actions taken were modest discounts. "
    "One-time or non-recurring items included a legal settlement. "
    "Foreign-exchange effects reduced revenue."
)


@dataclass(frozen=True, slots=True)
class FixtureCase:
    """One evaluation case with a human note on the expected behavior."""

    name: str
    note: str
    question: str
    answer: str
    context: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    profile: Profile = "nexa"

    def to_input(self) -> EvalInput:
        """Build the typed :class:`EvalInput` for this case."""
        chunks = [ContextChunk(id=f"chunk-{i + 1}", text=t) for i, t in enumerate(self.context)]
        return EvalInput(
            question=self.question, answer=self.answer, context=chunks,
            citations=self.citations, profile=self.profile,
        )


class FixtureSuite:
    """The planted fixture cases."""

    def cases(self) -> list[FixtureCase]:
        """Return all fixture cases."""
        return [
            FixtureCase(
                name="aligned",
                note="grounded, on-ask, all facets covered -> strong across the board",
                question=_DRIVERS_Q,
                answer=_DRIVERS_CTX,
                context=[_DRIVERS_CTX],
            ),
            FixtureCase(
                name="off_ask",
                note="on nothing the question asked -> relevance capped",
                question=_DRIVERS_Q,
                answer="Bananas are a good source of potassium and grow in tropical climates.",
                context=[_DRIVERS_CTX],
            ),
            FixtureCase(
                name="missing_facet",
                note="covers only cost drivers -> requirement coverage < 1",
                question=_DRIVERS_Q,
                answer="Cost drivers behind the change included higher input costs.",
                context=[_DRIVERS_CTX],
            ),
            FixtureCase(
                name="explanation_instead_of_fix",
                note="fix task answered with prose -> task_success low",
                question="Fix this broken function.",
                answer="The function is slow because the loop is quadratic in the input size.",
                context=["The function loops over the list twice."],
            ),
        ]
