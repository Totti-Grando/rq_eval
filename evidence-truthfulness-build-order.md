# Claude Code work order — Evidence & Truthfulness build (extends rq_eval)

**Goal:** implement the Evidence & Truthfulness category — claim-triplets, **groundedness, hallucination, source_quality, source_attribution**, and the **conformal factuality** layer — **strictly following `@evidence-truthfulness-design.md`**, inside the **existing `rq_eval` repo** built by the Response Quality order. Where this order and the design doc differ, the design doc wins; flag conflicts, don't improvise.

**This build has wired dependencies into the existing code — treat them as first-class work, not cleanup:**
1. `AccuracyDimension` currently uses a **stubbed `SourceQualityProvider`** (config default true) → this build replaces it with the real implementation.
2. `AccuracyDimension.attributed?` currently reuses plain grounding vs the cited chunk → this build gives it the real **AttributionProvider** (three-way + conformal confidence).
3. `GroundingProvider.check()` returns `{grounded: bool, raw_score}` → this build **upgrades the interface to three-way** `{label ∈ {E, N, C}, raw_score}` (design §1), which touches groundedness, attribution, source_quality's supports-check, completeness's unit assignment, and the mocks. This is a **contract version bump**: update every call site + tests in the same phase; nothing may keep consuming the old boolean shape.

**Environment (unchanged from the first order):** no AWS on the dev machine — everything behind the provider interfaces with **mock implementations**, all settings in the **single `config.yaml`**, live Bedrock/fairseq on the target machine, smoke test extended for any new live call. Style addendum (`@build-order-addendum-style-docs.md`) applies to every phase: OOP, one class per file, per-folder README.md with the calculations written out, `mypy --strict`, ARCHITECTURE.md updated.

## How to run this

- **Read first:** `@evidence-truthfulness-design.md` in full (§0 triplets, §1–§4 numbered steps, §5 conformal, §6 contracts) plus `@response-quality-design.md` §0.5 (AtomRecord/replay are unchanged and mandatory) and the existing `providers/` + `dimensions/accuracy/` code.
- **Plan mode.** Propose a phased plan for E1–E9 and wait for approval. One phase at a time; every phase ships tests; the full suite (including all existing Response Quality tests) stays green offline in mock mode.
- **Non-negotiables:** AI emits booleans/labels only — all arithmetic in code; every verdict an AtomRecord; every score replayable from the log without a model call.

---

## Phase A — contract upgrade + shared unit

### E1. Three-way GroundingProvider (contract version bump)
Upgrade `GroundingProvider.check(premise, hypothesis)` → `entails(premise, hypothesis) -> {label ∈ {E, N, C}, raw_score}` per design §6. Update: live Bedrock implementation (map grounding score + threshold bands to E/N/C per config), live fairseq-MNLI (native three-way), the **mock** (seeded deterministic labels), and **every existing call site** — groundedness usage inside accuracy, completeness's unit assignment (E = supported), and their tests. Keep a thin `supported: bool = (label == E)` convenience so downstream code reads cleanly.
**Accept:** no consumer of the old boolean shape remains (grep-proven); all existing tests green with the new contract; label thresholds live in config; tested.

### E2. Claim-triplet decomposition (design §0)
`pipeline/triplets.py`: decompose each cached Claim into subject-predicate-object **claim-triplets** (RefChecker-style) `[T3-gen]`, via a versioned prompt through the JudgeProvider; each triplet carries its claim id, citation, and source pointer. Pin + extend the stability harness to triplet agreement across re-runs. Mock: deterministic parse-based splitter so offline tests exercise multi-triplet claims.
**Accept:** Claims yield ≥1 triplet with provenance; extractor pinned + stability measured; runs offline; tested.

## Phase B — the four dimensions

### E3. groundedness (design §1)
`dimensions/groundedness/`: Titan-similarity pre-filter selects nearest context spans `[T1]` (mock: keyword overlap) → three-way entailment per triplet via E1 `[T2]` → `score = |E-labeled| / |total|` `[code]`. Export the per-claim `grounded?` boolean (all its triplets E) and the per-triplet confidences (for E8).
**Accept:** score replays from atoms; per-claim export consumed by accuracy (test: flipping a triplet label changes accuracy); pre-filter is provably not the score; tested.

### E4. hallucination (design §2)
`dimensions/hallucination/`: unsupported rate `= 1 − groundedness` with **Neutral vs Contradiction reported separately** `[code]`; **fabrication gate** `[T1]` — cited id ∈ retrieved set (set-membership), URL resolves (`urllib`, config-gated for offline: mock resolver), DOI/metadata validation (config-gated registry), any fabricated citation → **gate FAIL** through the existing gate path.
**Accept:** fabricated-citation fixture gates the run; N/C split visible in the DimensionResult; existence checks run offline via mocks; tested.

### E5. source_quality (design §3) — replaces accuracy's stub
`dimensions/source_quality/`: per source the eight checks — reachable `[T1]`, dated/fresh vs as-of window `[T1]`, authored `[T1]`, domain on the **config-versioned reliability allow/deny-list** `[T1]` (a human-maintained YAML, structural oracle), corroborated ≥2 independent (distinct domains/authors) `[T1 count]`, supports-this-claim via E1 `[T2]`, disinterested `[T3-samp]` (judge, sampled per config rate) → `source_quality = mean(property booleans)`; **`SourceQualityProvider.adequate(source, claim) -> bool` = score ≥ config threshold**. **Wire into accuracy:** replace the stub; Nexa profile keeps `source_adequate_default: true` via config.
**Accept:** the stub is gone; accuracy's `source-adequate` atom now traces to real property AtomRecords; profile default still honored; reliability list is config, not code; tested.

### E6. source_attribution (design §4) — completes accuracy's attributed atom
`dimensions/source_attribution/`: per claim-with-citation, three-way verdict of the **cited** chunk vs the claim via E1 (Attributable = E; config option for the four-way CAQA labels) `[T2]`; **ALCE citation recall + citation precision** computed in code from per-citation verdicts; **precision-favoring threshold** (config); claims with **no citation are excluded** from attribution scoring (they route to accuracy's unsourced residual — assert no double-count); `AttributionProvider.attributed(claim, cited_chunk) -> {bool, confidence}` feeding E8. **Wire into accuracy** as the real `attributed?` atom.
**Accept:** right-fact/wrong-citation fixture fails attribution while passing groundedness; recall+precision both on the DimensionResult; no-citation claims excluded here and counted once in the residual; accuracy consumes the provider; tested.

## Phase C — the statistical layer

### E7. Calibration-set store
`audit/calibration.py`: a pinned, versioned store of human-labeled calibration examples `{claim, context, label, stratum}` (YAML/JSONL under `paths.*`), with per-stratum partitions (stratum keys from config: question-type / source-type). Ship a small synthetic calibration fixture so the machinery is testable offline; real labels are added on the target machine.
**Accept:** load/validate/version works; per-stratum partitioning tested; fixture present.

### E8. Conformal factuality (design §5)
`scoring/conformal.py`: split-conformal in plain `numpy` (or `MAPIE` if configured): nonconformity scores from E3/E6 confidences → `τ̂ = Quantile_{⌈(1−α)(n+1)⌉/n}` on the calibration set → retain/flag per claim, recording the guarantee band `[1−α, 1−α+1/(n+1)]` on the DimensionResult. **Per-stratum calibration** per config (the marginal-vs-conditional caveat); `α`, set-size minimums, and refresh cadence are config keys. Deterministic given calibration set + α — property-tested.
**Accept:** with a fixed calibration fixture the threshold and retained set replay exactly; the guarantee band is stamped on results; per-stratum thresholds differ when strata differ; α is config; tested.

## Phase D — assembly

### E9. Runner, fixtures, docs, smoke test
Extend `evaluate()` to return the four new DimensionResults alongside Response Quality's; add fixtures: a fabricated-citation answer (gates), a right-fact/wrong-citation answer (attribution fails, groundedness passes), a bad-source answer (source_quality fails, groundedness passes), a contradiction-vs-neutral pair; extend the run report (N/C split, ALCE precision/recall, conformal band); extend `smoke_test.py` for any new live calls (reachability, DOI registry if enabled); per-folder READMEs with the formulas (`supported/|total|`, `mean(properties)`, ALCE recall/precision, the conformal quantile + band); ARCHITECTURE.md updated.
**Accept:** full end-to-end offline run with all eight dimensions; each fixture produces its expected outcome; replay verifier passes over the whole run; READMEs + ARCHITECTURE current.

---

## Definition of done

- Strict fidelity to `@evidence-truthfulness-design.md`; deviations flagged.
- **The three wired dependencies are closed:** accuracy's `source-adequate` and `attributed` atoms trace to the real providers (no stubs), and the three-way GroundingProvider contract is universal (no boolean-shape consumers left).
- Fabrication is a **T1 gate** through the existing gate path; the judge appears only in triplet extraction `[T3-gen]` and the sampled disinterest check.
- Conformal guarantee band computed, stamped, per-stratum-capable, fully deterministic given the pinned calibration set.
- One-spot config for everything new (labels/thresholds/α/strata/reliability-list path/registry toggles); offline-green in mock mode; smoke test covers new live calls; pinned requirements updated.
- Style addendum honored every phase; all existing + new tests green.
