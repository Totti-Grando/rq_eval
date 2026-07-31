"""Calibration-set store — pinned human-labeled examples (§5 / build order E7).

A versioned, per-stratum-partitionable store of ``{claim, context, label,
stratum}`` used by the conformal layer (§5/E8). Loaded from JSONL under
``conformal.calibration_path``; pinned by ``pins.calibration_version``. A small
synthetic fixture ships for offline testing; real human labels are added on the
target machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rq_eval.contracts import CalibrationExample

if TYPE_CHECKING:
    from rq_eval.config import Config


class CalibrationStore:
    """Loads + validates + partitions the pinned calibration examples."""

    def __init__(self, cfg: Config) -> None:
        """Load + validate the calibration JSONL for the configured path."""
        path = cfg.resolve(cfg.conformal.calibration_path)
        if not path.exists():
            raise FileNotFoundError(f"calibration set not found: {path}")
        self._version = cfg.pins.calibration_version
        self._examples: list[CalibrationExample] = [
            CalibrationExample.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @property
    def version(self) -> str:
        """The pinned calibration-set version."""
        return self._version

    def examples(self) -> list[CalibrationExample]:
        """Return all calibration examples."""
        return list(self._examples)

    def by_stratum(self) -> dict[str, list[CalibrationExample]]:
        """Partition the examples by their ``stratum`` key."""
        out: dict[str, list[CalibrationExample]] = {}
        for ex in self._examples:
            out.setdefault(ex.stratum, []).append(ex)
        return out
