"""Graders — tier adapters that turn provider calls into audited booleans.

T1 tools are pure; T2/T3 adapters apply config thresholds in code and log an
atom per check. Dimensions compose these; scoring never imports them.
"""

from rq_eval.graders.grounding_grader import GroundingGrader
from rq_eval.graders.judge_grader import JudgeGrader
from rq_eval.graders.relevance_grader import RelevanceGrader
from rq_eval.graders.t1 import T1Tools

__all__ = ["GroundingGrader", "JudgeGrader", "RelevanceGrader", "T1Tools"]
