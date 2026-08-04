# pipeline

**Design ref:** §0 — build once: the shared claim-extraction pipeline
(accuracy, completeness, and relevance all consume its output).

**Purpose:** turn an answer into a cached set of atomic, decontextualized,
verifiable `Claim`s, each with a source-sentence pointer and any citation.
Decomposition is **deterministic parsing, not generation** (§0.2): the primary
path calls no judge and no GeneratorProvider. Every boolean decision is logged as
an atom; the pinned reference version is `pins.extractor_version`.

**Classes:**
- `PromptLibrary` — [pin] loads the versioned JSON prompt set (now just the optional realizer prompt).
- `Segmenter` — [T1] sentence segmentation via `NlpProvider`.
- `VerifiableSpanSelector` — [T1] lexical hedge/opinion filter (`T1Tools.is_verifiable`); keep provable spans, route opinions/hedges/hypotheticals. No judge.
- `ClaimExtractor` — [T1] decompose each sentence into content-unit clauses via `NlpProvider.parse_clauses`; abstractive-implied spans (`T1Tools.is_abstractive_implied`) are flagged, not generated; optional pinned [T2] surface-realizer behind `extraction.realizer_enabled` (default off, droppable).
- `Decontextualizer` — [T1/T2] `resolve_coref` (context carried forward) + a structural self-contained check (`T1Tools.has_leading_pronoun`), no judge.
- `StabilityHarness` — re-run extraction N times, report claim-set agreement.
- `ClaimPipeline` — orchestrates the steps; returns `PipelineResult`.
- `ClaimTripletExtractor` — [T1 parse-first, T3-gen residual] (Evidence §0) decompose each claim into RefChecker-style S-P-O `Triplet`s (pinned by `triplet_extractor_version`).
- `TripletStabilityHarness` — triplet-id set agreement across re-runs.

**Calculations:**
- `stability = |∩ claim-id sets| / |∪ claim-id sets|` over `pipeline.stability_runs`
  passes (1.0 = identical every run; empty union = 1.0).
- No score is produced here — the pipeline yields claims consumed downstream.

**Determinism:** the whole primary path is [T1] deterministic (segment / verifiable
filter / clause decomposition / structural decontext); the optional realizer is a
pinned [T2] step. Under the mock everything is fully reproducible (stability = 1.0).
Every yes/no decision is an atom; **no `pipeline.*` T3 judge atoms are emitted**.

**How to extend:** bump `pins.extractor_version` and add a new
`config/prompts/<version>.json`; add a step by composing a new class into
`ClaimPipeline._build` and logging its atoms.
