"""Mock generator — deterministic ``[T3-gen]`` text (build order B2).

Reproducible text for the pinned generative steps. An optional leading
``[[tag]]`` selects the rule:

* ``[[echo]]``       -> returns the payload unchanged
* ``[[sentences]]``  -> splits the payload into sentence ``items``
* ``[[repeat]]``     -> ``items`` = ``n`` copies of the payload (used by Method-A
                        reverse-questions: questions derived from the answer)
* (no tag)           -> echo

The "payload" is the text inside ``{{ ... }}`` if present (so a live-friendly
natural-language instruction can wrap the content), else the whole body. Live
providers strip the tag and unwrap ``{{ }}`` before calling the model.

Emits text only, never a number.
"""

from __future__ import annotations

import re

from rq_eval.providers.base import GenerationResult, GeneratorProvider

_TAG = re.compile(r"^\s*\[\[([^\]]+)\]\]\s*(.*)$", re.DOTALL)
_PAYLOAD = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)
_SENT = re.compile(r"(?<=[.!?])\s+")


class MockGeneratorProvider(GeneratorProvider):
    """Deterministic, tag-dispatched text generator for offline runs."""

    def __init__(self, seed: int) -> None:
        """Store the base seed (kept for signature parity with live)."""
        self._seed = seed

    def generate(self, prompt: str, *, seed: int, n: int = 1) -> GenerationResult:
        """Return deterministic text/items per the optional ``[[tag]]`` rule."""
        tag, body = self._parse(prompt)
        payload = _PAYLOAD.search(body)
        body = (payload.group(1) if payload else body).strip()
        if tag == "sentences":
            items = [s.strip() for s in _SENT.split(body) if s.strip()]
            return GenerationResult(text=body, items=items)
        if tag == "repeat":
            return GenerationResult(text=body, items=[body] * max(1, n))
        return GenerationResult(text=body, items=[body] if body else [])

    @staticmethod
    def _parse(prompt: str) -> tuple[str | None, str]:
        m = _TAG.match(prompt)
        return (m.group(1).strip(), m.group(2)) if m else (None, prompt)
