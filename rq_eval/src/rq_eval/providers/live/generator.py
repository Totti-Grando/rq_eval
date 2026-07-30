"""Live generator — Bedrock Claude ``[T3-gen]`` (build order B2, live path).

Returns text for pinned generative steps (claim extraction, unit drafting,
objective/outcome inference). Deterministic-leaning: temperature 0 and the
caller's seed threaded into the request for reproducibility. Never emits a
number used as a score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rq_eval.providers.base import GenerationResult, GeneratorProvider
from rq_eval.providers.live.bedrock_session import BedrockSession

if TYPE_CHECKING:
    from rq_eval.config import Config


class BedrockGeneratorProvider(GeneratorProvider):
    """Text generator backed by a fixed Bedrock Claude model."""

    def __init__(self, cfg: Config, session: BedrockSession) -> None:
        """Store config + shared Bedrock session (no network yet)."""
        self._cfg = cfg
        self._session = session

    def generate(self, prompt: str, *, seed: int, n: int = 1) -> GenerationResult:
        """Generate text; if ``n`` > 1 return that many newline-split items."""
        instruction = prompt
        if n > 1:
            instruction = f"{prompt}\n\nProduce exactly {n} items, one per line."
        resp = self._session.runtime().converse(
            modelId=self._cfg.models.judge_id,
            system=[{"text": f"Deterministic generation. seed={seed}."}],
            messages=[{"role": "user", "content": [{"text": instruction}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 1024},
        )
        text = resp["output"]["message"]["content"][0]["text"].strip()
        items = [ln.strip() for ln in text.splitlines() if ln.strip()] if n > 1 else [text]
        return GenerationResult(text=text, items=items)
