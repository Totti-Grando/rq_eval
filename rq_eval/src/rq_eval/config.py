"""Configuration — the ONLY module that reads ``config.yaml`` / ``.env``.

Design ref: build order B1 (single-spot config) and §0.5 reproducibility fence.

Every environment-specific or tunable value in the program is defined in
``config.yaml`` and reaches the rest of the code exclusively through the typed
:class:`Config` object returned by :func:`load_config`. No other module opens
``config.yaml`` or reads model ids / regions / thresholds / seeds from the
environment (enforced by ``tests/test_config_single_source.py``).

All fields are required: a missing key raises ``pydantic.ValidationError`` at
load time rather than silently defaulting.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Typed config schema — mirrors config.yaml one-to-one.                        #
# --------------------------------------------------------------------------- #

_Strict = ConfigDict(extra="forbid", frozen=True)


class ProvidersConfig(BaseModel):
    """Which provider implementations to construct (mock == fully offline)."""

    model_config = _Strict
    mode: Literal["mock", "live"]


class AwsConfig(BaseModel):
    """AWS region/profile for the live provider stack."""

    model_config = _Strict
    region: str
    profile: str


class ModelsConfig(BaseModel):
    """Bedrock model ids + NLI backend selector."""

    model_config = _Strict
    judge_id: str
    embed_id: str
    guardrail_id: str
    guardrail_version: str
    nli: Literal["mock", "bedrock", "fairseq"]


class BandsConfig(BaseModel):
    """Score cutoffs: ``score >= G`` -> "G", ``>= A`` -> "A", else "R"."""

    model_config = _Strict
    G: float = Field(ge=0.0, le=1.0)
    A: float = Field(ge=0.0, le=1.0)


class ThresholdsConfig(BaseModel):
    """Float-to-boolean cutoffs; thresholding always happens in our code."""

    model_config = _Strict
    relevance_tau: float = Field(ge=0.0, le=1.0)
    grounding_tau: float = Field(ge=0.0, le=1.0)
    attribution_tau: float = Field(ge=0.0, le=1.0)
    entail_tau: float = Field(ge=0.0, le=1.0)
    contra_tau: float = Field(ge=0.0, le=1.0)
    bands: BandsConfig


class CompletenessConfig(BaseModel):
    """§2 knobs — abstention floor, vitality weighting, dedupe cutoff."""

    model_config = _Strict
    min_n: int = Field(ge=0)
    vital_weighting: bool
    dedupe_tau: float = Field(ge=0.0, le=1.0)


class AccuracyConfig(BaseModel):
    """§1 knobs — importance weighting, numeric tolerance, residual policy."""

    model_config = _Strict
    importance_weighting: bool
    numeric_tolerance: float = Field(ge=0.0)
    residual_policy: Literal["nexa", "ravenpack"]


class TaskSuccessConfig(BaseModel):
    """§4 knobs — whether 'executable' outcomes run for real (else heuristic)."""

    model_config = _Strict
    execution_sandbox: bool


class HallucinationConfig(BaseModel):
    """§2 knobs — fabrication-gate reference resolver + DOI toggle."""

    model_config = _Strict
    resolver: Literal["mock", "live"]
    doi_registry_enabled: bool


class RelevanceConfig(BaseModel):
    """§3 knobs — method selection and Method-A reverse-question count."""

    model_config = _Strict
    method: Literal["A", "B", "both"]
    reverse_questions_n: int = Field(ge=1)
    off_ask_cap: float = Field(ge=0.0, le=1.0)


class PipelineConfig(BaseModel):
    """§0 knobs — claim-extraction stability harness."""

    model_config = _Strict
    stability_runs: int = Field(ge=1)


class PinsConfig(BaseModel):
    """Frozen reference versions (reproducibility fence, §0.5.5)."""

    model_config = _Strict
    extractor_version: str
    triplet_extractor_version: str
    nuggetizer_version: str
    template_version: str


class SeedsConfig(BaseModel):
    """Fixed seeds for every sampling step (§0.5.5)."""

    model_config = _Strict
    judge: int
    embedding: int
    reverse_questions: int
    dedupe: int
    stability: int


class PathsConfig(BaseModel):
    """Filesystem locations; resolved relative to the project root."""

    model_config = _Strict
    atom_log: str
    atom_log_backend: Literal["jsonl", "sqlite"]
    requirement_templates: str
    task_templates: str
    prompts: str
    runs_dir: str


class Config(BaseModel):
    """Root config object — the typed view of ``config.yaml``."""

    model_config = _Strict
    providers: ProvidersConfig
    aws: AwsConfig
    models: ModelsConfig
    thresholds: ThresholdsConfig
    completeness: CompletenessConfig
    accuracy: AccuracyConfig
    task_success: TaskSuccessConfig
    hallucination: HallucinationConfig
    relevance: RelevanceConfig
    pipeline: PipelineConfig
    pins: PinsConfig
    seeds: SeedsConfig
    paths: PathsConfig

    # Set by load_config so path helpers can resolve against the project root.
    root: Path = Field(exclude=True)

    def resolve(self, relative: str) -> Path:
        """Resolve a config-relative path against the project root."""
        return (self.root / relative).resolve()


# --------------------------------------------------------------------------- #
# Loading — the single entry point.                                           #
# --------------------------------------------------------------------------- #


def project_root() -> Path:
    """Return the rq_eval project root (the folder holding ``config.yaml``)."""
    return Path(__file__).resolve().parents[2]


def _load_dotenv(root: Path) -> None:
    """Load ``.env`` KEY=VALUE lines into ``os.environ`` (does not overwrite).

    Dependency-free so the offline core needs no extra package. Secrets (AWS
    profile/keys) are consumed by boto3 in live mode via the environment.
    """
    env_path = root / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load + validate ``config.yaml`` into a typed :class:`Config`.

    [code] Inputs: optional explicit path (defaults to the project-root
    ``config.yaml``). Output: a frozen :class:`Config`. Raises
    ``pydantic.ValidationError`` on any missing/extra/ill-typed key.
    """
    root = project_root()
    _load_dotenv(root)
    cfg_path = Path(path) if path is not None else root / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config.yaml must be a mapping, got {type(data).__name__}")
    return Config(root=root, **data)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached accessor for the default project config."""
    return load_config()


def load_yaml(path: str | os.PathLike[str]) -> object:
    """Load an arbitrary YAML data file (e.g. requirement/task templates).

    Centralized here so ``config.py`` remains the ONLY module importing yaml
    (enforced by tests/test_config_single_source.py). Callers pass an already
    config-resolved path.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML data file not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))
