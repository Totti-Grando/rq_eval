"""Provider smoke test — verify each dependency BEFORE any evaluation.

Design ref: build order B2 acceptance + migration step 6.

Run ``python smoke_test.py``. It loads config and probes each provider by name
(judge, embeddings, grounding, relevance, NLP) plus optional NLI, printing
PASS/FAIL per provider. In mock mode everything runs offline and should PASS;
on the target machine in live mode it exercises Bedrock/Titan/Guardrail and the
spaCy/coreferee/NLI backends. Do not run evaluations until every check passes.

Provider probes are filled in by B2; until then this verifies config + the
offline core toolchain.
"""

from __future__ import annotations

import importlib
import sys

from rq_eval.config import load_config


def _check(name: str, fn: object) -> bool:
    try:
        fn()  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001 - smoke test reports, never raises
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        return False
    print(f"  PASS  {name}")
    return True


def _check_core() -> None:
    for mod in ("pydantic", "yaml", "numpy", "scipy"):
        importlib.import_module(mod)


def main() -> int:
    """Load config, probe every provider, return process exit code."""
    cfg = load_config()
    print(f"rq_eval smoke test — providers.mode = {cfg.providers.mode}")
    print(f"  region={cfg.aws.region} nli={cfg.models.nli} relevance={cfg.relevance.method}")
    print("-" * 60)

    results: list[bool] = [_check("offline-core (pydantic/yaml/numpy/scipy)", _check_core)]

    try:
        from rq_eval.providers.factory import ProviderFactory  # noqa: PLC0415
    except ImportError:
        print("  ....  providers not built yet (B2) — skipping provider probes")
    else:
        factory = ProviderFactory(cfg)
        results.extend(factory.smoke_probes(_check))

    print("-" * 60)
    ok = all(results)
    print("ALL PASS" if ok else "SOME CHECKS FAILED — fix before evaluating")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
