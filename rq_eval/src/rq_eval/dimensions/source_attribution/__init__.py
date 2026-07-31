"""§4 source_attribution dimension — ALCE citation recall/precision."""

from rq_eval.dimensions.source_attribution.alce import AlceScorer
from rq_eval.dimensions.source_attribution.export import AttributionExport
from rq_eval.dimensions.source_attribution.labels import AttributionLabeler
from rq_eval.dimensions.source_attribution.provider import AttributionProviderImpl
from rq_eval.dimensions.source_attribution.source_attribution import SourceAttributionDimension

__all__ = [
    "AlceScorer",
    "AttributionExport",
    "AttributionLabeler",
    "AttributionProviderImpl",
    "SourceAttributionDimension",
]
