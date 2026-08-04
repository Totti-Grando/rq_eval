"""Providers — external dependencies behind mockable interfaces.

Public surface: the interfaces + result types (``base``) and the single
construction point (``factory``). Import providers only through the factory.
"""

from rq_eval.providers.base import (
    CorefResult,
    EmbeddingProvider,
    EntailmentLabel,
    EntailmentResult,
    ExplanationJudge,
    GenerationResult,
    GeneratorProvider,
    GroundingProvider,
    JudgeVerdict,
    NlpProvider,
    RelevanceProvider,
    ResolverProvider,
    ScoringJudge,
    Vector,
)
from rq_eval.providers.factory import ProviderFactory, Providers

__all__ = [
    "CorefResult",
    "EmbeddingProvider",
    "EntailmentLabel",
    "EntailmentResult",
    "ExplanationJudge",
    "GenerationResult",
    "GeneratorProvider",
    "GroundingProvider",
    "JudgeVerdict",
    "NlpProvider",
    "ProviderFactory",
    "Providers",
    "RelevanceProvider",
    "ResolverProvider",
    "ScoringJudge",
    "Vector",
]
