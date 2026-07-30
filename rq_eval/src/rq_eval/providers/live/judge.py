"""Live judge — Bedrock Claude, boolean-only (build order B2, live path).

Uses the Converse API with a strict YES/NO instruction and parses the reply to
a boolean. No numeric output is ever requested or parsed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rq_eval.providers.base import JudgeProvider, JudgeVerdict
from rq_eval.providers.live.bedrock_session import BedrockSession
from rq_eval.providers.live.prompt_prep import PromptPrep

if TYPE_CHECKING:
    from rq_eval.config import Config

_SYSTEM = (
    "You are a strict binary judge. Answer the question about the provided "
    "context with exactly 'YES' or 'NO' on the first line, then a brief reason."
)


class BedrockJudgeProvider(JudgeProvider):
    """Boolean judge backed by a fixed Bedrock Claude model."""

    def __init__(self, cfg: Config, session: BedrockSession) -> None:
        """Store config + shared Bedrock session (no network yet)."""
        self._cfg = cfg
        self._session = session

    def binary(self, question: str, context: str) -> JudgeVerdict:
        """Ask the model a yes/no question; parse YES/NO to a boolean verdict."""
        clean_q = PromptPrep.clean(question)
        resp = self._session.runtime().converse(
            modelId=self._cfg.models.judge_id,
            system=[{"text": _SYSTEM}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": f"QUESTION:\n{clean_q}\n\nCONTEXT:\n{context}"}],
                }
            ],
            inferenceConfig={"temperature": 0.0, "maxTokens": 200},
        )
        text = resp["output"]["message"]["content"][0]["text"].strip()
        verdict = text.upper().lstrip().startswith("YES")
        return JudgeVerdict(verdict=verdict, reason=text[:500])
