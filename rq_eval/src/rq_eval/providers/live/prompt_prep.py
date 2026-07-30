"""Strip mock-only control markers before sending prompts to a live model.

The mock providers use a leading ``[[tag]]`` (and, for the generator, a
``{{ payload }}`` marker) as a deterministic control channel. Those markers are
meaningless to a real model, so live providers run prompts through
:class:`PromptPrep` first: the tag is removed and ``{{ x }}`` is unwrapped to
``x``, leaving the natural-language instruction intact.
"""

from __future__ import annotations

import re

_TAG = re.compile(r"^\s*\[\[[^\]]+\]\]\s*")
_PAYLOAD = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)


class PromptPrep:
    """Removes mock control markers from a prompt for live model calls."""

    @staticmethod
    def clean(text: str) -> str:
        """Drop a leading ``[[tag]]`` and unwrap any ``{{ payload }}``."""
        without_tag = _TAG.sub("", text, count=1)
        return _PAYLOAD.sub(lambda m: m.group(1), without_tag).strip()
