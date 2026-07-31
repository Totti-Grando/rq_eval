"""Providers — external dependencies behind mockable interfaces.

Public surface: the interfaces + result types (``base``) and the single
construction point (``factory``). Import providers only through the factory.
"""

from rq_eval.providers.base import (
    CorefResult,
    EmbeddingProvider,
    EntailmentLabel,
    EntailmentResult,
    GenerationResult,
    GeneratorProvider,
    GroundingProvider,
    JudgeProvider,
    JudgeVerdict,
    NlpProvider,
    RelevanceProvider,
    ResolverProvider,
    Vector,
)
from rq_eval.providers.factory import ProviderFactory, Providers

__all__ = [
    "CorefResult",
    "EmbeddingProvider",
    "EntailmentLabel",
    "EntailmentResult",
    "GenerationResult",
    "GeneratorProvider",
    "GroundingProvider",
    "JudgeProvider",
    "JudgeVerdict",
    "NlpProvider",
    "ProviderFactory",
    "Providers",
    "RelevanceProvider",
    "ResolverProvider",
    "Vector",
]
