"""Dimensions — one class per Response-Quality dimension (§1–§4)."""

from rq_eval.dimensions.base import Dimension
from rq_eval.dimensions.responsiveness import ResponsivenessExport

__all__ = ["Dimension", "ResponsivenessExport"]
