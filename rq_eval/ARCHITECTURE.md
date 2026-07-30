# rq_eval — architecture

Response Quality evaluation. **AI extracts and judges yes/no; code computes
every number.** Every boolean is logged as an atom; every score replays from the
atoms + a formula id with no model call.

## Layer diagram (dependencies point downward)

```
                        runner.py            (orchestration + report)
                            |
        +-------------------+-------------------+
        |                   |                   |
   dimensions/        dimensions/         dimensions/ ...     (§1–§4: one class each)
   accuracy           completeness        relevance
        |                   |                   |
        +----------+--------+---------+---------+
                   |                  |
               graders/           scoring/      (T1/T2/T3 adapters | pure formulas)
                   |                  |
              pipeline/          (no model imports in scoring/)
                   |
              providers/  <-- ProviderFactory (config-selected mock|live)
                   |
        contracts.py  +  audit/     (typed records | append-only log + replay)
                   |
               config.py            (THE single config reader)
```

## Data flow of one evaluation (end-to-end)

1. `runner.evaluate(EvalInput)` builds providers via `ProviderFactory(config)`.
2. `pipeline/` (§0) extracts cached, decontextualized, verifiable `Claim`s.
3. `dimensions/relevance` (§3) runs first and **exports per-claim responsive
   atoms**; `dimensions/accuracy` (§1) imports them (never recomputes).
4. Each dimension emits booleans → `AtomRecord`s appended to `audit/` and
   feeds `scoring/` (pure functions) to compute a `DimensionResult`.
5. `runner` renders a report; the `audit/` replay verifier recomputes every
   score from the log with no model call and asserts equality.

## Design-doc section → folder map

This table is verified against the actual tree by
`tests/test_navigation.py` (the map cannot rot): every `src/` path named here
must exist, and every code subfolder must be listed here and carry a `README.md`.
Rows are added as each phase lands its folder (per the addendum).

| Design ref | Path | Phase |
|---|---|---|
| build order B1 — single-spot config | `src/rq_eval/config.py` | B1 ✔ |
| build order B2 — provider interfaces + factory | `src/rq_eval/providers` | B2 ✔ |
| build order B2 — deterministic mocks | `src/rq_eval/providers/mock` | B2 ✔ |
| build order B2 — live Bedrock/spaCy/fairseq | `src/rq_eval/providers/live` | B2 ✔ |
| §0.5 contracts (records) | `src/rq_eval/contracts.py` | B3 ✔ |
| §0.5 audit (atom log + replay) | `src/rq_eval/audit` | B3 ✔ |
| formulas + registry (replay-critical; B5 adds CI/bands) | `src/rq_eval/scoring` | B3 ✔ |
| §0 shared claim-extraction pipeline | `src/rq_eval/pipeline` | B4 ✔ |
| tier adapters (T1 tools, T2/T3 graders) | `src/rq_eval/graders` | B5 ✔ |

_Planned (added in their phase): dimension base `dimensions/`,
§1 `dimensions/accuracy/`, §2 `dimensions/completeness/`,
§3 `dimensions/relevance/`, §4 `dimensions/task_success/`._

## Build status

Phase A: **B1 complete** (skeleton, one-spot config, requirements, install/smoke
scaffolding). B2 (providers), B3 (contracts+audit) next. This document and the
table above grow with each phase.
