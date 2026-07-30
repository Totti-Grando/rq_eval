"""Contracts — the shared typed data shapes (§0.5).

One data shape per record makes every dimension interchangeable, replayable, and
auditable. No raw dicts cross module boundaries (addendum §1).

* :class:`Claim`            — cached output of the §0 pipeline.
* :class:`AtomRecord`       — the audit primitive: one per yes/no check.
* :class:`DimensionResult`  — a dimension's score + band + CI + provenance.
* :class:`EvalInput` / :class:`ContextChunk` — one evaluation's inputs.

The **replay guarantee** (§0.5.4) rests on :class:`AtomRecord`: each atom carries
its ``verdict``, ``weight``, and ``subject`` so any score recomputes from the
logged atoms + a formula id without re-invoking a model (see ``audit/replay.py``).
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Tier = Literal["T1", "T2", "T3", "code"]
Profile = Literal["nexa", "ravenpack"]

_Model = ConfigDict(extra="forbid")


class ContextChunk(BaseModel):
    """A retrieved source chunk, addressable by ``id`` for citation/attribution."""

    model_config = _Model
    id: str
    text: str


class EvalInput(BaseModel):
    """Everything one evaluation needs (addendum: typed, no raw dicts)."""

    model_config = _Model
    question: str
    answer: str
    context: list[ContextChunk] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)  # cited chunk ids
    profile: Profile = "nexa"


class Claim(BaseModel):
    """§0 output — an atomic, decontextualized, verifiable claim."""

    model_config = _Model
    id: str
    text: str
    source_sentence: str
    verifiable: bool
    decontextualized: bool
    extractor_version: str
    citation: str | None = None  # cited chunk id, if any


class AtomRecord(BaseModel):
    """§0.5.2 — the audit primitive; one immutable record per yes/no check.

    ``id`` is a content hash of the identity fields (everything except the
    wall-clock ``timestamp``), so identical checks on distinct subjects get
    distinct ids while replays are stable.
    """

    model_config = _Model
    id: str
    subject: str  # claim/unit/outcome id this atom concerns ("answer" for answer-level)
    role: str  # which check, e.g. "grounded", "responsive", "unit_support"
    question: str
    tier: Tier
    verdict: bool
    weight: float = 1.0
    evidence: str = ""  # span / source-id / "score=..." — what decided it
    grader_id: str = ""
    model: str = ""
    model_version: str = ""
    seed: int | None = None
    timestamp: str = ""

    @classmethod
    def create(
        cls,
        *,
        subject: str,
        role: str,
        question: str,
        tier: Tier,
        verdict: bool,
        weight: float = 1.0,
        evidence: str = "",
        grader_id: str = "",
        model: str = "",
        model_version: str = "",
        seed: int | None = None,
        timestamp: str = "",
    ) -> AtomRecord:
        """Build an atom, deriving its content-hash ``id`` (excludes timestamp)."""
        identity = "|".join(
            str(x)
            for x in (
                subject,
                role,
                question,
                tier,
                verdict,
                weight,
                evidence,
                grader_id,
                model,
                model_version,
                seed,
            )
        )
        atom_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
        return cls(
            id=atom_id,
            subject=subject,
            role=role,
            question=question,
            tier=tier,
            verdict=verdict,
            weight=weight,
            evidence=evidence,
            grader_id=grader_id,
            model=model,
            model_version=model_version,
            seed=seed,
            timestamp=timestamp,
        )


class DimensionResult(BaseModel):
    """§0.5.3 — a dimension's score with band, CI, and full provenance."""

    model_config = _Model
    dimension: str
    score: float = Field(ge=0.0, le=1.0)
    band: str
    ci_low: float = Field(ge=0.0, le=1.0)
    ci_high: float = Field(ge=0.0, le=1.0)
    n: int = Field(ge=0)
    inputs_hash: str
    atom_ids: list[str]
    formula_id: str
    abstained: bool = False
    extra: dict[str, float] = Field(default_factory=dict)  # e.g. requirement_coverage
