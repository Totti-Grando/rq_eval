"""§2 hallucination dimension — unsupported rate + fabrication gate."""

from rq_eval.dimensions.hallucination.fabrication_gate import FabricationGate, FabricationResult
from rq_eval.dimensions.hallucination.hallucination import HallucinationDimension

__all__ = ["FabricationGate", "FabricationResult", "HallucinationDimension"]
