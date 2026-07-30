"""Provider interfaces + result types (build order B2).

Every external dependency sits behind one of these abstract interfaces with a
``mock`` and a ``live`` sibling implementation, so the whole program builds,
runs, and tests offline. Construction is only ever via
:class:`~rq_eval.providers.factory.ProviderFactory`.

Booleans-only discipline is enforced structurally here:

* :class:`JudgeProvider` exposes exactly one method — ``binary`` — returning a
  ``verdict: bool``. There is no numeric scoring endpoint on the judge.
* :class:`GeneratorProvider` is a *separate* interface for the design's
  ``[T3-gen]`` steps (claim extraction, decontextualization, unit drafting,
  objective/outcome inference). It returns **text**, never a number — generated
  references are pinned and then judged by booleans, so this does not create a
  scoring endpoint.
* :class:`GroundingProvider` / :class:`RelevanceProvider` return raw floats
  only; the float→boolean thresholding happens in *our* grader/dimension code
  from config (they never read a threshold), keeping all decisions in-code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

Vector = list[float]


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """A single boolean judgment plus its rationale (audited as an atom)."""

    verdict: bool
    reason: str


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Text produced by a ``[T3-gen]`` step; ``items`` is an optional split.

    Never carries a number used as a score — generation yields reference text
    (claims, units, objectives) that is subsequently pinned and judged.
    """

    text: str
    items: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GroundingResult:
    """Raw grounding/entailment score in [0, 1]; thresholded in our code."""

    raw_score: float


@dataclass(frozen=True, slots=True)
class CorefResult:
    """Decontextualized text with references resolved (spaCy/coreferee, [T2])."""

    resolved_text: str


class JudgeProvider(ABC):
    """[T3] The judge — the ONLY method is a boolean verdict (no scoring)."""

    @abstractmethod
    def binary(self, question: str, context: str) -> JudgeVerdict:
        """Answer a yes/no ``question`` about ``context``. Inputs→ bool+reason."""


class GeneratorProvider(ABC):
    """[T3-gen] Pinned generative steps; returns text, never numbers."""

    @abstractmethod
    def generate(self, prompt: str, *, seed: int, n: int = 1) -> GenerationResult:
        """Generate reference text for ``prompt``. Deterministic given ``seed``."""


class EmbeddingProvider(ABC):
    """[T2] Text → fixed-dimension vectors (Titan live; hashed mock)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[Vector]:
        """Embed each text. Inputs→ one vector per text (equal dimension)."""


class GroundingProvider(ABC):
    """[T2] Entailment/grounding check → raw score (thresholded in our code)."""

    @abstractmethod
    def check(self, source: str, claim: str) -> GroundingResult:
        """Score whether ``source`` entails ``claim``. Inputs→ raw_score∈[0,1]."""


class RelevanceProvider(ABC):
    """[T2] Query↔response relevance → raw score (thresholded in our code)."""

    @abstractmethod
    def score(self, query: str, response: str) -> float:
        """Score response relevance to query. Inputs→ raw score∈[0,1]."""


class NlpProvider(ABC):
    """[T1/T2] Deterministic NLP — sentence segmentation + coref resolution.

    Wrapped behind this interface (with a regex/identity mock) so §0's pipeline
    runs offline without spaCy/coreferee installed; the live impl uses spaCy
    (``en_core_web_lg``) + coreferee exactly as the design specifies.
    """

    @abstractmethod
    def segment(self, text: str) -> list[str]:
        """[T1] Split text into sentences. Deterministic."""

    @abstractmethod
    def resolve_coref(self, text: str, context: str = "") -> CorefResult:
        """[T2] Resolve pronouns/referents, carrying ``context`` forward."""
