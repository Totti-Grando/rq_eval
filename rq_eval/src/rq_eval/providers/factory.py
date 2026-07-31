"""ProviderFactory — the single construction point for all providers (B2).

Nothing else in the codebase instantiates a provider. The factory reads config
(``providers.mode`` and ``models.nli``) and returns a :class:`Providers` bundle
of mock or live implementations. Swapping mock↔live is therefore a config-only
change — no call site moves.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rq_eval.providers.base import (
    EmbeddingProvider,
    GeneratorProvider,
    GroundingProvider,
    JudgeProvider,
    NlpProvider,
    RelevanceProvider,
)

if TYPE_CHECKING:
    from rq_eval.config import Config
    from rq_eval.providers.live.bedrock_session import BedrockSession

CheckFn = Callable[[str, Callable[[], object]], bool]


@dataclass(frozen=True, slots=True)
class Providers:
    """Typed bundle of every provider a dimension may need (injected via DI)."""

    judge: JudgeProvider
    generator: GeneratorProvider
    embedding: EmbeddingProvider
    grounding: GroundingProvider
    relevance: RelevanceProvider
    nlp: NlpProvider


class ProviderFactory:
    """Builds the config-selected provider set (mock or live)."""

    def __init__(self, cfg: Config) -> None:
        """Store config; construction is deferred to :meth:`build`."""
        self._cfg = cfg

    def build(self) -> Providers:
        """Return the :class:`Providers` bundle for the configured mode."""
        if self._cfg.providers.mode == "mock":
            return self._build_mock()
        return self._build_live()

    # -- mock -------------------------------------------------------------- #
    def _build_mock(self) -> Providers:
        from rq_eval.providers.mock.embedding import MockEmbeddingProvider
        from rq_eval.providers.mock.generator import MockGeneratorProvider
        from rq_eval.providers.mock.grounding import MockGroundingProvider
        from rq_eval.providers.mock.judge import MockJudgeProvider
        from rq_eval.providers.mock.nlp import MockNlpProvider
        from rq_eval.providers.mock.relevance import MockRelevanceProvider

        s = self._cfg.seeds
        t = self._cfg.thresholds
        return Providers(
            judge=MockJudgeProvider(seed=s.judge),
            generator=MockGeneratorProvider(seed=s.judge),
            embedding=MockEmbeddingProvider(seed=s.embedding),
            grounding=MockGroundingProvider(
                seed=s.judge, entail_tau=t.entail_tau, contra_tau=t.contra_tau
            ),
            relevance=MockRelevanceProvider(seed=s.judge),
            nlp=MockNlpProvider(seed=s.judge),
        )

    # -- live -------------------------------------------------------------- #
    def _build_live(self) -> Providers:
        from rq_eval.providers.live.bedrock_session import BedrockSession
        from rq_eval.providers.live.embedding import TitanEmbeddingProvider
        from rq_eval.providers.live.generator import BedrockGeneratorProvider
        from rq_eval.providers.live.judge import BedrockJudgeProvider
        from rq_eval.providers.live.nlp import SpacyNlpProvider
        from rq_eval.providers.live.relevance_guardrail import GuardrailRelevanceProvider

        session = BedrockSession(self._cfg)
        return Providers(
            judge=BedrockJudgeProvider(self._cfg, session),
            generator=BedrockGeneratorProvider(self._cfg, session),
            embedding=TitanEmbeddingProvider(self._cfg, session),
            grounding=self._build_grounding(session),
            relevance=GuardrailRelevanceProvider(self._cfg, session),
            nlp=SpacyNlpProvider(self._cfg),
        )

    def _build_grounding(self, session: BedrockSession) -> GroundingProvider:
        """Select the grounding backend by ``models.nli``."""
        nli = self._cfg.models.nli
        if nli == "fairseq":
            from rq_eval.providers.live.grounding_fairseq import FairseqGroundingProvider

            return FairseqGroundingProvider(self._cfg)
        if nli == "mock":
            from rq_eval.providers.mock.grounding import MockGroundingProvider

            return MockGroundingProvider(
                seed=self._cfg.seeds.judge,
                entail_tau=self._cfg.thresholds.entail_tau,
                contra_tau=self._cfg.thresholds.contra_tau,
            )
        from rq_eval.providers.live.grounding_guardrail import GuardrailGroundingProvider

        return GuardrailGroundingProvider(self._cfg, session)

    # -- smoke ------------------------------------------------------------- #
    def smoke_probes(self, check: CheckFn) -> list[bool]:
        """Probe each provider via ``check`` (used by smoke_test.py)."""
        p = self.build()
        return [
            check("judge", lambda: p.judge.binary("[[affirm]] ok?", "context")),
            check("generator", lambda: p.generator.generate("[[echo]] hello", seed=1)),
            check("embedding", lambda: p.embedding.embed(["alpha beta", "gamma"])),
            check("grounding", lambda: p.grounding.entails("the sky is blue", "sky is blue")),
            check("relevance", lambda: p.relevance.score("why blue?", "the sky is blue")),
            check("nlp", lambda: p.nlp.segment("One. Two.")),
        ]
