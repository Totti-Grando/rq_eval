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

### The two judge roles (judge-minimizing reform)

Two provider interfaces enforce "explain, never override" structurally:

- **`ScoringJudge`** — booleans only, confined to the five irreducible,
  reference-grounded residuals: accuracy **unsourced**, task_success
  **adequacy**, relevance **abstention**, admissibility **decidability**, and
  source_quality **disinterest**. `tests/test_determinism_ledger.py` locks this
  set so no new judge creeps onto the scoring path. Everything else is now fixed:
  on-ask = NLI+lexical (DIVER-QA), admissibility = double-NLI, disinterest = a T1
  COI rule.
- **`ExplanationJudge`** — runs *after* all scores are final, read-only over the
  `DimensionResult`s + `AtomRecord`s, returns the user-facing summary. It has no
  `verdict`, writes no atom, and no `formula_id` references it (test-enforced), so
  the explanation layer sits strictly downstream of scoring.

### Design-sync update (U1–U7): parse-first extraction, relevance tree, completeness modes

- **Deterministic extraction (§0.2).** `pipeline/` decomposes claims by parsing,
  not generation: `NlpProvider.parse_clauses` (mock clause splitter / live spaCy
  ClausIE-style) + a lexical `T1Tools.is_verifiable` filter + a structural
  self-contained check. No judge, no GeneratorProvider on the primary path;
  abstractive-implied spans are flagged; an optional pinned `[T2]` surface-realizer
  is off by default (`extraction.realizer_enabled`). Evidence triplets are
  parse-first the same way — generator only for the nested/abstractive residual.
- **Relevance is an anchor-and-support tree (§3).** `dimensions/relevance/`:
  `EdgeBuilder` (entailment-confirmed premise→conclusion edges) → `AnchorSelector`
  (on-ask seed + graph centrality + conformal recall band) → `SupportTree`
  (depth-graded reachability, `max_hops`, `depth_decay`) → `OrphanResolver`
  (off-topic / stranded-veracity / background; contradictions routed). The score
  (`relevance_tree_capped_mean`) is fixed NLI + code; the `responsive` export for
  accuracy is unchanged.
- **Forward-declared `ConsistencyProvider`.** A Reasoning-category interface
  (`providers/consistency.py`) that relevance routes edge-soundness and stranded
  contradictions to; a default stub with safe defaults, swapped cleanly later.
- **Completeness reference modes (§2).** `ReferenceModeSelector` picks the Tier-1
  reference by `completeness.reference_mode` — `generated` (default), `archetype`
  (`question_archetypes.yaml`), `templated` — stamps `assurance_mode` on the
  result, and reports a human recall-sample `recall_miss_rate` as its error bar.

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
| dimension base + shared responsiveness | `src/rq_eval/dimensions` | B6 ✔ |
| §3 relevance | `src/rq_eval/dimensions/relevance` | B6 ✔ |
| §1 accuracy | `src/rq_eval/dimensions/accuracy` | B7 ✔ |
| §2 completeness | `src/rq_eval/dimensions/completeness` | B8 ✔ |
| §4 task_success (verifier-routed) | `src/rq_eval/dimensions/task_success` | B9 ✔ |
| §4 outcome verifiers (routing table) | `src/rq_eval/dimensions/task_success/verifiers` | B9 ✔ |
| §0.3 edge-detection recall harness | `src/rq_eval/validation` | G4 ✔ |
| Phase D — runner + report + fixtures | `src/rq_eval/runner.py` | B10 ✔ |
| E&T §1 groundedness | `src/rq_eval/dimensions/groundedness` | E3 ✔ |
| E&T §2 hallucination | `src/rq_eval/dimensions/hallucination` | E4 ✔ |
| E&T §3 source_quality | `src/rq_eval/dimensions/source_quality` | E5 ✔ |
| E&T §4 source_attribution | `src/rq_eval/dimensions/source_attribution` | E6 ✔ |

## Build status

**Complete: B1–B10 (Response Quality) + E1–E9 (Evidence & Truthfulness).**
Response Quality: config, providers, contracts+audit, §0 pipeline,
graders+scoring, the four dimensions, runner. Evidence & Truthfulness: three-way
grounding contract, claim-triplets, groundedness, hallucination (fabrication
gate), source_quality + source_attribution (accuracy's real imports — no stubs),
calibration store, conformal factuality, and the eight-dimension runner. Full
suite runs offline in `providers.mode: mock`; every score replays from the atom
log; `mypy --strict` and `ruff` clean.
