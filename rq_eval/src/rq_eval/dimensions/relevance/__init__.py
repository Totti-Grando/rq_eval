"""§3 relevance dimension — on-topic + responsiveness."""

from rq_eval.dimensions.relevance.claim_responsiveness import ClaimResponsiveness
from rq_eval.dimensions.relevance.method_a import MethodAReverseQuestions
from rq_eval.dimensions.relevance.method_b import MethodBGuardrail
from rq_eval.dimensions.relevance.relevance import RelevanceDimension

__all__ = [
    "ClaimResponsiveness",
    "MethodAReverseQuestions",
    "MethodBGuardrail",
    "RelevanceDimension",
]
