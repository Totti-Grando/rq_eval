# Claude Code work order — Response Quality evaluation program (build to spec)

**Goal:** implement the Response Quality category — shared claim pipeline, contracts/audit layer, and the four dimensions (accuracy, completeness, relevance, task_success) — **strictly following `@response-quality-design.md`**. Where this order and the design doc differ, **the design doc wins**; flag the conflict, don't improvise.

**Environment constraint (critical):** this machine has **no AWS connectivity**. The target machine (where this will run for real) has Bedrock access and can perform installs. Therefore:
- **All environment-specific settings live in exactly one place** — a single `config.yaml` at repo root (plus `.env` for secrets). No model ID, region, endpoint, threshold, seed, or path may be hard-coded anywhere else. Moving machines must mean editing one file, running one script.
- **Every external dependency sits behind a provider interface with a `mock` implementation**, so the entire program builds, runs, and tests **offline** here. On the target machine, switching `providers.mode: mock → live` in config is the only change needed.
- Ship an **install script + pinned requirements** the target machine runs once, and a **smoke test** that verifies each live provider (Bedrock Claude, Titan embeddings, ApplyGuardrail, spaCy/coreferee models, optional fairseq-NLI) before any evaluation is attempted.

## How to run this

- **Read first:** `@response-quality-design.md` in full — §0 pipeline, §0.5 contracts (the atom record and replay guarantee are non-negotiable), §1–§4 numbered build steps, and the development summary table.
- **Plan mode.** Propose a phased plan for B1–B10 and wait for approval. One phase at a time; each phase ships tests; keep all tests green.
- **Non-negotiables from the design:** AI emits **booleans only** — any code path where a model output is parsed into a number, or where a model does arithmetic, is a defect. All scores are computed by the tool layer from atom booleans. Every atom is logged per §0.5. Scores must satisfy the replay guarantee.

---

## Phase A — skeleton, config, providers (the portability layer)

### B1. Repo skeleton + single-spot config
```
rq_eval/
  config.yaml            # THE one place: providers, model ids, region, thresholds,
                         # seeds, pins, toggles, paths, bands, min-n
  .env.example           # AWS_PROFILE / keys — template only, never committed
  install.sh             # target-machine setup (pip install -r, spacy/coreferee
                         # model downloads, optional fairseq)
  requirements.txt       # pinned versions
  smoke_test.py          # verifies each live provider; prints pass/fail per provider
  src/rq_eval/
    config.py            # loads+validates config.yaml (pydantic); the ONLY config reader
    contracts.py         # Claim, AtomRecord, DimensionResult (§0.5)
    providers/           # base interfaces + live/ + mock/
    pipeline/            # §0 claim extraction
    graders/             # T1 tools, T2 adapters, T3 judge wrapper
    dimensions/          # accuracy.py, completeness.py, relevance.py, task_success.py
    scoring/             # formulas, Wilson CI, bands, replay
    audit/               # atom log store, replay verifier
  tests/                 # unit + replay + fixture tests, ALL runnable offline
```
Config keys (minimum): `providers.mode (mock|live)`, `aws.{region, profile}`, `models.{judge_id, embed_id, guardrail_id+version, nli (bedrock|fairseq|mock)}`, `thresholds.{relevance_tau, grounding_tau, bands{G,A,R}}`, `completeness.{min_n, vital_weighting}`, `accuracy.{importance_weighting, numeric_tolerance, residual_policy}`, `relevance.{method (A|B|both)}`, `pins.{extractor_version, nuggetizer_version, template_version}`, `seeds.*`, `paths.*`.
**Accept:** `config.py` is the only module reading config; grep proves no literal model IDs/regions/thresholds elsewhere; loading with a missing key fails loudly; tested.

### B2. Provider interfaces + mock + live implementations
Interfaces: `JudgeProvider.binary(question, context) -> {verdict: bool, reason: str}` (the ONLY judge method — no free scoring endpoint exists, enforcing booleans-only at the interface); `EmbeddingProvider.embed(texts) -> vectors`; `GroundingProvider.check(source, claim) -> {grounded: bool, raw_score: float}` (thresholding happens in OUR code from config, then the boolean is what's used); `RelevanceProvider.score(query, response) -> float` (thresholded in code).
- **mock/**: deterministic fakes — seeded-hash verdicts, fixed-dim pseudo-embeddings, keyword-overlap grounding — good enough to exercise every code path offline and to make tests deterministic.
- **live/**: `boto3` Bedrock Claude (judge), Titan (embeddings), ApplyGuardrail contextual-grounding (grounding + relevance); optional `fairseq` torch.hub RoBERTa-MNLI behind the same `GroundingProvider` interface, selected by `models.nli`.
**Accept:** the full test suite passes with `mode: mock` and no network; `smoke_test.py` exercises each live provider and reports per-provider pass/fail; swapping mock→live changes zero code outside config; tested.

### B3. Contracts + audit store (§0.5, verbatim)
Implement `Claim`, `AtomRecord`, `DimensionResult` exactly as §0.5 defines (fields included). Append-only atom log (SQLite or JSONL, config-selected path). Implement the **replay verifier**: recompute any `DimensionResult.score` from its logged `atom_ids` + `formula_id` **without any model call** and assert equality.
**Accept:** every grader call produces an AtomRecord with grader_id, model+version, seed, evidence; replay reproduces every score bit-for-bit; a tampered atom makes replay fail; tested.

## Phase B — shared pipeline + graders

### B4. §0 claim-extraction pipeline
The five numbered steps: spaCy segmentation `[T1]` → verifiable-span selection (judge, boolean per span) → Claimify-style select/disambiguate/extract (judge prompts from a versioned prompt file; ambiguous → flagged, not guessed) → decontextualization (coreferee + judge, carrying context forward) → **pin & stability**: extractor prompt+model versioned in config; a stability harness re-runs extraction N times (config) and reports claim-set agreement.
**Accept:** produces cached `Claim` objects with all fields; unverifiable spans excluded and routed; stability metric computed and logged; runs fully under mock; tested with fixture answers.

### B5. T1 grader toolbox + scoring library
`graders/t1.py`: numeric exact-match with config tolerance; citation-id set-membership; parse/conjunction-split atomicity check; length/counts. `scoring/`: Wilson 95% CI, weighted means, off-ask cap, band mapper (config bands), abstention (min-n) — pure functions, property-tested.
**Accept:** all pure-deterministic; property tests pass; no model imports anywhere in `scoring/`.

## Phase C — the four dimensions (each strictly per its numbered steps)

### B6. relevance (§3) — build FIRST (accuracy imports it)
Method A (RAGAS-style reverse-questions via judge + Titan cosine, seeded, diagnostic) and Method B (grounding-relevance score thresholded in code) behind `relevance.method`; answer-level + **claim-level responsive atoms** (exported for accuracy); off-ask cap; abstention handling (a proper decline on an unanswerable question scores relevant); T3-res residual only.
**Accept:** per-claim responsive booleans exported and consumed downstream; cap and abstention behaviors tested; both methods runnable (A under mock via fake embeddings); tested.

### B7. accuracy (§1)
Four booleans per claim — grounded (GroundingProvider), source-adequate (stub interface until source_quality exists; config default = true for Nexa-profile), attributed (directional grounding vs the *cited* chunk), responsive (imported from B6, never recomputed) — conjunction + weighted mean in code; unsourced→residual judge path; inferred→flagged for the shared inference-validity check (interface stub); importance weights read from completeness units when present; numeric/temporal edge rules from B5.
**Accept:** `score = Σ correct·w / Σ w` replays from atoms; responsive is provably the imported atom (test: changing B6 output changes accuracy); Nexa-profile collapse (source-adequacy≈1) via config; tested.

### B8. completeness (§2, two-tier)
Tier-1 requirement templates loaded from versioned YAML (fixed scaffold; per-question-type; human-editable — this file IS the structural oracle); Tier-2 unit drafting (judge, top-down + bottom-up); **unit admissibility gate** (atomic `[T1 parse]`, self-contained `[coreferee]`, entailment-decidable `[one-time judge admission]`) → frozen, versioned unit set; dedupe (embeddings + clustering); vital/okay at both tiers; per-unit binary assignment (GroundingProvider, answer=premise/unit=hypothesis, partial=unsupported); **two-level scoring** (per-requirement recall, normalized + requirement coverage) + strict-vital-recall core; Wilson CI + min-n abstention.
**Accept:** a unit failing any admissibility check never enters the frozen set; scores replay; requirement-coverage catches a wholly-missing facet in a fixture; frozen sets carry version+corpus_hash; tested.

### B9. task_success (§4)
Objective inference (judge); task-type classification against the config-versioned template taxonomy (fix/explain/compare/produce/summarize/recommend/extract); outcome decomposition; per-outcome boolean; `achieved/required` in code; multi-goal weighting and impossible-task ("can't be done because X" = success) per spec.
**Accept:** templates load from versioned config; per-outcome atoms logged; edge behaviors (partial, implicit goal fixture, impossible task) tested.

## Phase D — assembly

### B10. Runner + report + fixtures
`evaluate(question, answer, context, citations, profile) -> {4 DimensionResults + atom log}`; a small fixture suite of Q/A/context cases with known expected behaviors (including a planted off-ask answer, a missing-facet answer, an explanation-instead-of-fix answer); a human-readable run report (scores, bands, CIs, abstentions, atom counts by tier); end-to-end offline run in mock mode.
**Accept:** full end-to-end run offline; fixtures produce the expected qualitative outcomes; report renders; replay verifier passes over a whole run.

---

## Definition of done

- Strict fidelity to `@response-quality-design.md` — every numbered build step implemented as written; deviations flagged, not improvised.
- **One-spot config:** all environment/tuning values in `config.yaml`; machine migration = edit config + run `install.sh` + `smoke_test.py`.
- **Offline-first:** entire suite green with `mode: mock`, no network; live providers isolated behind interfaces.
- **Booleans-only enforced structurally:** the judge interface exposes no numeric output; grounding/relevance floats are thresholded in our code; `scoring/` imports no model code.
- **Full audit:** every verdict an AtomRecord; every score replayable from the log without model calls; generated references pinned with version+corpus_hash+seed.
- Pinned `requirements.txt`; each B-item tested; all tests green.
