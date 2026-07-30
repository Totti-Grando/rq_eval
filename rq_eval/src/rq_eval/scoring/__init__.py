"""Scoring — pure formulas over atoms (no model imports).

The replay-critical composition formulas + registry live here from B3; the
statistical library (Wilson CI, band mapping, off-ask cap, min-n abstention)
is added in B5. Everything in this package is a pure function of its inputs.
"""

from rq_eval.scoring.aggregation import MinNAbstention, OffAskCap
from rq_eval.scoring.bands import BandMapper
from rq_eval.scoring.formulas import (
    ConjunctionWeightedMeanFormula,
    MeanFormula,
    WeightedMeanFormula,
    default_registry,
)
from rq_eval.scoring.registry import Formula, FormulaRegistry
from rq_eval.scoring.wilson import WilsonInterval

__all__ = [
    "BandMapper",
    "ConjunctionWeightedMeanFormula",
    "Formula",
    "FormulaRegistry",
    "MeanFormula",
    "MinNAbstention",
    "OffAskCap",
    "WeightedMeanFormula",
    "WilsonInterval",
    "default_registry",
]
