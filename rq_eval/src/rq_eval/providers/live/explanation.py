"""Live explanation judge — Bedrock Claude, read-only summary (R1).

Turns finished results + atoms into a user-facing summary via Converse. Emits
no verdict and writes no atom; never read by any formula (design §0.5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rq_eval.contracts import AtomRecord, DimensionResult
from rq_eval.providers.base import ExplanationJudge
from rq_eval.providers.live.bedrock_session import BedrockSession

if TYPE_CHECKING:
    from rq_eval.config import Config

_SYSTEM = (
    "You are a read-only reporter. Summarize these already-computed evaluation "
    "scores for a human. Do not re-judge, re-score, or change any number."
)


class BedrockExplanationJudge(ExplanationJudge):
    """Read-only run summary backed by a fixed Bedrock Claude model."""

    def __init__(self, cfg: Config, session: BedrockSession) -> None:
        """Store config + shared Bedrock session (no network yet)."""
        self._cfg = cfg
        self._session = session

    def summarize(self, results: dict[str, DimensionResult], atoms: list[AtomRecord]) -> str:
        """Render a prose summary of the finished scores (no scoring)."""
        rows = "\n".join(
            f"{name}: score={r.score:.3f} band={r.band} n={r.n}"
            for name, r in sorted(results.items())
        )
        resp = self._session.runtime().converse(
            modelId=self._cfg.models.judge_id,
            system=[{"text": _SYSTEM}],
            messages=[{"role": "user", "content": [{"text": f"SCORES:\n{rows}"}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 400},
        )
        return str(resp["output"]["message"]["content"][0]["text"]).strip()
