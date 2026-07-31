"""E7 — calibration-set store (§5), offline fixture."""

from __future__ import annotations

from rq_eval.audit.calibration import CalibrationStore
from rq_eval.config import load_config


def test_loads_and_versions() -> None:
    store = CalibrationStore(load_config())
    assert store.version == "calibration-v1"
    examples = store.examples()
    assert len(examples) >= 8
    assert all(isinstance(e.label, bool) for e in examples)


def test_per_stratum_partition() -> None:
    strata = CalibrationStore(load_config()).by_stratum()
    assert set(strata) == {"finance", "sports"}
    assert all(len(v) >= 3 for v in strata.values())
    # every example lands in exactly its stratum
    for name, exs in strata.items():
        assert all(e.stratum == name for e in exs)
