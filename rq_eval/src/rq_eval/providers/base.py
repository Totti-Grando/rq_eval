"""Provider interfaces + result types (build order B2).

Every external dependency sits behind one of these abstract interfaces with a
``mock`` and a ``live`` sibling implementation, so the whole program builds,
runs, and tests offline. Construction is only ever via
:class:`~rq_eval.providers.factory.ProviderFactory`.

Booleans-only discipline is enforced structurally here:

* :class:`ScoringJudge` exposes exactly one method — ``binary`` — returning a
  ``verdict: bool``. There is no numeric scoring endpoint on the judge. The
  read-only :class:`ExplanationJudge` is separate and never feeds a score.
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
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from rq_eval.contracts import AtomRecord, ContextChunk, DimensionResult

Vector = list[float]
EntailmentLabel = Literal["E", "N", "C"]  # Entailment / Neutral / Contradiction


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
class EntailmentResult:
    """Three-way NLI verdict (design §1/§6): E/N/C label + raw score.

    ``supported`` is the boolean downstream code reads (label == "E"); the raw
    score is kept for the conformal layer. Neutral = source silent; Contradiction
    = source says the opposite (the severe sub-case, split out by hallucination).
    """

    label: EntailmentLabel
    raw_score: float

    @property
    def supported(self) -> bool:
        """True iff the hypothesis is entailed by the premise (label == 'E')."""
        return self.label == "E"


@dataclass(frozen=True, slots=True)
class CorefResult:
    """Decontextualized text with references resolved (spaCy/coreferee, [T2])."""

    resolved_text: str


@dataclass(frozen=True, slots=True)
class AttributionResult:
    """§4/§6 — whether a claim is attributed to its cited chunk + confidence.

    ``attributed`` = Attributable (cited chunk entails the claim) ∧ confidence ≥
    the precision-favoring threshold; ``confidence`` feeds the conformal layer.
    """

    attributed: bool
    confidence: float
    label: str


class ScoringJudge(ABC):
    """[T3] Score-affecting judge — the ONLY method is a boolean verdict.

    Confined (design §0.5) to the irreducible residuals: accuracy-unsourced,
    task_success-adequacy, relevance-abstention, admissibility-decidability, and
    source_quality-disinterest. Booleans-only (no numeric output); an optional
    ``reference`` is passed where one exists so the verdict isn't a no-reference
    guess (judges over-credit without one).
    """

    @abstractmethod
    def binary(self, question: str, context: str, reference: str | None = None) -> JudgeVerdict:
        """Answer a yes/no ``question`` about ``context`` (+ optional reference)."""


class ExplanationJudge(ABC):
    """[read-only] User-facing run summary — never an input to any score.

    Receives finished ``DimensionResult``s + ``AtomRecord``s and returns prose.
    Structurally quarantined: no ``verdict``, writes no ``AtomRecord``, and no
    ``formula_id`` may reference it (enforced by tests). "Explain, never override."
    """

    @abstractmethod
    def summarize(
        self, results: dict[str, DimensionResult], atoms: list[AtomRecord]
    ) -> str:
        """Return a human-readable summary of a finished run (no scoring)."""


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
    """[T2] Three-way entailment (E/N/C); one verifier, three premises (§6).

    Shared by groundedness (premise = context), attribution (premise = the
    *cited* chunk), and source_quality's supports-check. Thresholding of the raw
    score into the label happens in the live impl from config bands; downstream
    code reads ``label``/``supported``, never re-thresholds.
    """

    @abstractmethod
    def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        """Classify ``hypothesis`` vs ``premise`` as E / N / C (+ raw score)."""


class RelevanceProvider(ABC):
    """[T2] Query↔response relevance → raw score (thresholded in our code)."""

    @abstractmethod
    def score(self, query: str, response: str) -> float:
        """Score response relevance to query. Inputs→ raw score∈[0,1]."""


class ResolverProvider(ABC):
    """[T1-ish] Reference-existence check for the fabrication gate (§2).

    Returns whether a citation reference *exists* (a URL resolves, a DOI is in a
    registry). Orthogonal to whether it *supports* the claim. Set-membership of a
    chunk-id in the retrieved set is done in our code, not here.
    """

    @abstractmethod
    def resolve(self, reference: str) -> bool:
        """Return True iff the reference exists / resolves."""


class SourceQualityProvider(ABC):
    """§3/§6 — is a source trustworthy enough to count for a claim.

    ``adequate`` = source_quality score ≥ config threshold; accuracy imports this
    as its ``source-adequate?`` atom. Takes the full source set too (corroboration
    needs cross-source independence). Emits property AtomRecords.
    """

    @abstractmethod
    def adequate(self, source: ContextChunk, claim: str, sources: list[ContextChunk]) -> bool:
        """Return whether ``source`` is adequate for ``claim`` (score ≥ threshold)."""


class AttributionProvider(ABC):
    """§4/§6 — is a claim attributed to the source that actually supports it.

    accuracy imports this as its ``attributed?`` atom. Returns the boolean +
    confidence (the confidence is what the conformal layer wraps).
    """

    @abstractmethod
    def attributed(self, claim: str, cited_chunk: str) -> AttributionResult:
        """Three-way cited-chunk↔claim verdict → attributed bool + confidence."""


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
    def parse_clauses(self, sentence: str) -> list[str]:
        """[T1] Decompose a sentence into content-unit clauses (§0.2).

        Deterministic dependency-parse decomposition (ClausIE/PredPatt-style
        over spaCy live; a rule/clause splitter in the mock) — *not* generation.
        A sentence with no separable clause returns itself as the single unit.
        """

    @abstractmethod
    def resolve_coref(self, text: str, context: str = "") -> CorefResult:
        """[T2] Resolve pronouns/referents, carrying ``context`` forward."""
