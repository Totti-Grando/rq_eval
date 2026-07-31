"""Fixture suite with known qualitative outcomes (build order B10 + E9).

Response Quality cases (off-ask, missing-facet, explanation-instead-of-fix) plus
Evidence & Truthfulness cases: a fabricated-citation answer (gates), a
right-fact/wrong-citation answer (attribution fails, groundedness passes), a
bad-source answer (source_quality fails, groundedness passes), and a
contradiction-vs-neutral pair.
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
    context: list[ContextChunk] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    profile: Profile = "nexa"

    def to_input(self) -> EvalInput:
        """Build the typed :class:`EvalInput` for this case."""
        return EvalInput(
            question=self.question, answer=self.answer, context=self.context,
            citations=self.citations, profile=self.profile,
        )


class FixtureSuite:
    """The planted fixture cases (Response Quality + Evidence & Truthfulness)."""

    def cases(self) -> list[FixtureCase]:
        """Return all fixture cases."""
        return [*self._response_quality(), *self._evidence_truthfulness()]

    def _response_quality(self) -> list[FixtureCase]:
        drivers_ctx = [ContextChunk(id="chunk-1", text=_DRIVERS_CTX)]
        return [
            FixtureCase(
                name="aligned",
                note="grounded, on-ask, all facets covered -> strong across the board",
                question=_DRIVERS_Q, answer=_DRIVERS_CTX, context=drivers_ctx,
            ),
            FixtureCase(
                name="off_ask",
                note="on nothing the question asked -> relevance capped",
                question=_DRIVERS_Q,
                answer="Bananas are a good source of potassium and grow in tropical climates.",
                context=drivers_ctx,
            ),
            FixtureCase(
                name="missing_facet",
                note="covers only cost drivers -> requirement coverage < 1",
                question=_DRIVERS_Q,
                answer="Cost drivers behind the change included higher input costs.",
                context=drivers_ctx,
            ),
            FixtureCase(
                name="explanation_instead_of_fix",
                note="fix task answered with prose -> task_success low",
                question="Fix this broken function.",
                answer="The function is slow because the loop is quadratic in the input size.",
                context=[ContextChunk(id="chunk-1", text="The function loops over the list twice")],
            ),
        ]

    def _evidence_truthfulness(self) -> list[FixtureCase]:
        return [
            FixtureCase(
                name="fabricated_citation",
                note="cites a source that does not exist -> hallucination gate FAIL",
                question="Who won the final?",
                answer="Real Madrid won the final [fabricated-99].",
                context=[ContextChunk(id="chunk-1", text="Real Madrid won the final in 2024.")],
            ),
            FixtureCase(
                name="wrong_citation",
                note="true claim cited to the wrong chunk -> attribution fails, grounds",
                question="Who won the final?",
                answer="Real Madrid won the final [chunk-2].",
                context=[
                    ContextChunk(id="chunk-1", text="Real Madrid won the final in 2024."),
                    ContextChunk(id="chunk-2", text="Bananas are rich in potassium."),
                ],
            ),
            FixtureCase(
                name="bad_source",
                note="supported by a deny-listed domain -> source_quality low, groundedness passes",
                question="What happened to the market?",
                answer="The market crashed sharply in March [chunk-1].",
                context=[ContextChunk(
                    id="chunk-1", text="The market crashed sharply in March.",
                    url="https://fakenews.example/x", domain="fakenews.example",
                )],
            ),
            FixtureCase(
                name="contradiction",
                note="answer contradicts the source -> hallucination contradiction_rate > 0",
                question="Was the sky blue?",
                answer="The sky is not blue today.",
                context=[ContextChunk(id="chunk-1", text="The sky is blue and clear today.")],
            ),
        ]
