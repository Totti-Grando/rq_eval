"""Versioned prompt library for the §0 pipeline (build order B4, step 5 pin).

Prompts are stored as JSON (not YAML — only ``config.py`` reads YAML) under
``paths.prompts/<extractor_version>.json`` and pinned by
``pins.extractor_version``. Each prompt carries a mock-only ``[[tag]]`` /
``{{ payload }}`` marker that live providers strip.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rq_eval.config import Config


class PromptLibrary:
    """Loads + serves the pinned claim-extraction prompts."""

    def __init__(self, cfg: Config) -> None:
        """Load the JSON prompt file for the configured extractor version."""
        path = cfg.resolve(cfg.paths.prompts) / f"{cfg.pins.extractor_version}.json"
        if not path.exists():
            raise FileNotFoundError(f"prompt file not found: {path}")
        raw: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
        self._data: dict[str, str] = {k: str(v) for k, v in raw.items()}
        self._version: str = self._data["version"]

    @property
    def version(self) -> str:
        """The pinned prompt-set version (stamped onto generated references)."""
        return self._version

    def verifiable(self) -> str:
        """[T3] Prompt: is this span provable true/false (VeriScore filter)?"""
        return self._data["verifiable"]

    def disambiguate(self) -> str:
        """[T3] Prompt: is this ambiguous and unresolvable (flag, don't guess)?"""
        return self._data["disambiguate"]

    def extract(self, sentence: str) -> str:
        """[T3-gen] Prompt: extract atomic propositions from ``sentence``."""
        return self._data["extract"].replace("{sentence}", sentence)

    def decontextualized(self) -> str:
        """[T3] Prompt: is the claim self-contained after coref resolution?"""
        return self._data["decontextualized"]
