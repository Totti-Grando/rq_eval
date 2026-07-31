"""The Tier-2 unit (nugget) data shape (§2)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Unit:
    """An atomic, checkable Tier-2 nugget under one requirement."""

    id: str
    text: str
    requirement_id: str
    vital: bool
    origin: str  # "top_down" | "bottom_up"

    @classmethod
    def create(cls, text: str, requirement_id: str, vital: bool, origin: str) -> Unit:
        """Build a unit with a content-hash id (stable across runs)."""
        uid = "unit:" + hashlib.sha256(f"{requirement_id}|{text}".encode()).hexdigest()[:12]
        return cls(id=uid, text=text, requirement_id=requirement_id, vital=vital, origin=origin)
