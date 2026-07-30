"""Clock abstraction for atom timestamps (§0.5.2).

Timestamps are metadata only — they never enter a score or an atom's content
hash — so replay is unaffected by them. The abstraction lets tests inject a
fixed time for reproducible atom logs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime


class Clock(ABC):
    """Source of ISO-8601 timestamps for atom records."""

    @abstractmethod
    def now(self) -> str:
        """Return the current time as an ISO-8601 string."""


class SystemClock(Clock):
    """Wall-clock UTC time."""

    def now(self) -> str:
        """Return the current UTC time in ISO-8601 form."""
        return datetime.now(UTC).isoformat()


class FixedClock(Clock):
    """A constant timestamp — deterministic atom logs for tests."""

    def __init__(self, value: str = "1970-01-01T00:00:00+00:00") -> None:
        """Store the constant timestamp to return from :meth:`now`."""
        self._value = value

    def now(self) -> str:
        """Return the fixed timestamp."""
        return self._value
