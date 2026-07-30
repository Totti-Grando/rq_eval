"""rq_eval — Response Quality evaluation program.

Four dimensions (accuracy, completeness, relevance, task_success) built on a
shared claim-extraction pipeline, a contracts/audit layer, and a strict
discipline: AI emits booleans only; code computes every number.

See ``ARCHITECTURE.md`` and ``response-quality-design.md`` (one level up).
"""

from rq_eval.config import Config, get_config, load_config

__all__ = ["Config", "get_config", "load_config"]
__version__ = "0.1.0"
