"""B1 — config loads, validates, and fails loudly on missing keys."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rq_eval.config import Config, load_config, project_root


def test_default_config_loads() -> None:
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.providers.mode in {"mock", "live"}
    assert cfg.thresholds.bands.G >= cfg.thresholds.bands.A
    # default project config ships offline-first
    assert cfg.providers.mode == "mock"


def test_resolve_is_project_relative() -> None:
    cfg = load_config()
    assert cfg.resolve(cfg.paths.runs_dir) == (project_root() / cfg.paths.runs_dir).resolve()


def test_missing_key_fails_loudly(tmp_path: Path) -> None:
    full = yaml.safe_load((project_root() / "config.yaml").read_text(encoding="utf-8"))
    del full["thresholds"]["relevance_tau"]  # drop one required key
    bad = tmp_path / "config.yaml"
    bad.write_text(yaml.safe_dump(full), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_extra_key_forbidden(tmp_path: Path) -> None:
    full = yaml.safe_load((project_root() / "config.yaml").read_text(encoding="utf-8"))
    full["surprise"] = 1
    bad = tmp_path / "config.yaml"
    bad.write_text(yaml.safe_dump(full), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_band_bounds_enforced(tmp_path: Path) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text(
        textwrap.dedent(
            (project_root() / "config.yaml")
            .read_text(encoding="utf-8")
            .replace("G: 0.90", "G: 1.5")
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(bad)
