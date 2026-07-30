"""Live embeddings — Amazon Titan Text Embeddings (build order B2, live path)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rq_eval.providers.base import EmbeddingProvider, Vector
from rq_eval.providers.live.bedrock_session import BedrockSession

if TYPE_CHECKING:
    from rq_eval.config import Config


class TitanEmbeddingProvider(EmbeddingProvider):
    """Embeddings backed by a fixed Titan model via InvokeModel."""

    def __init__(self, cfg: Config, session: BedrockSession) -> None:
        """Store config + shared Bedrock session (no network yet)."""
        self._cfg = cfg
        self._session = session

    def embed(self, texts: list[str]) -> list[Vector]:
        """Invoke Titan once per text; return one embedding vector each."""
        runtime = self._session.runtime()
        out: list[Vector] = []
        for text in texts:
            resp = runtime.invoke_model(
                modelId=self._cfg.models.embed_id,
                body=json.dumps({"inputText": text}),
            )
            payload = json.loads(resp["body"].read())
            out.append([float(x) for x in payload["embedding"]])
        return out
