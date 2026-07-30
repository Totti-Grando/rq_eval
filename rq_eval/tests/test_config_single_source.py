"""B1 — config.py is the ONLY place that reads config / holds magic values.

Enforces the build-order rule: no model id, region, endpoint, threshold, seed,
band, or path is hard-coded outside config.yaml, and no module other than
config.py parses config or reads AWS secrets from the environment.
"""

from __future__ import annotations

from pathlib import Path

from rq_eval.config import project_root

SRC = project_root() / "src" / "rq_eval"

# Substrings that must never appear outside config.py.
FORBIDDEN_LITERALS = (
    "anthropic.claude",   # model ids
    "amazon.titan",
    "us-east-1",          # regions
    "us-west-2",
)
FORBIDDEN_CONFIG_READS = (
    "yaml.safe_load",
    "yaml.load",
    'open("config',
    "open('config",
    "config.yaml",
)
FORBIDDEN_ENV_READS = (
    "AWS_ACCESS_KEY",
    "AWS_SECRET",
    "AWS_PROFILE",
)


def _py_files_excluding_config() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if p.name != "config.py"]


def test_only_config_py_reads_config_and_magic_values() -> None:
    offenders: list[str] = []
    for path in _py_files_excluding_config():
        text = path.read_text(encoding="utf-8")
        for needle in (*FORBIDDEN_LITERALS, *FORBIDDEN_CONFIG_READS, *FORBIDDEN_ENV_READS):
            if needle in text:
                offenders.append(f"{path.relative_to(project_root())}: contains '{needle}'")
    assert not offenders, "magic values / config reads leaked outside config.py:\n" + "\n".join(
        offenders
    )


def test_only_config_py_imports_yaml() -> None:
    offenders = [
        str(p.relative_to(project_root()))
        for p in _py_files_excluding_config()
        if "import yaml" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"only config.py may import yaml; found in: {offenders}"
