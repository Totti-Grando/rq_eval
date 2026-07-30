# pipeline

**Design ref:** §0 — build once: the shared claim-extraction pipeline
(accuracy, completeness, and relevance all consume its output).

**Purpose:** turn an answer into a cached set of atomic, decontextualized,
verifiable `Claim`s, each with a source-sentence pointer and any citation. Every
boolean decision along the way is logged as an atom; the generative steps are
pinned by `pins.extractor_version`.

**Classes:**
- `PromptLibrary` — [pin] loads the versioned JSON prompt set.
- `Segmenter` — [T1] sentence segmentation via `NlpProvider`.
- `VerifiableSpanSelector` — [T3] keep provable spans, route opinions/hedges.
- `ClaimExtractor` — [T3] disambiguate (flag, don't guess) + [T3-gen] extract atomic propositions.
- `Decontextualizer` — [T2 coref + T3] resolve references (context carried forward) and confirm self-containment.
- `StabilityHarness` — re-run extraction N times, report claim-set agreement.
- `ClaimPipeline` — orchestrates steps 1–5; returns `PipelineResult`.

**Calculations:**
- `stability = |∩ claim-id sets| / |∪ claim-id sets|` over `pipeline.stability_runs`
  passes (1.0 = identical every run; empty union = 1.0).
- No score is produced here — the pipeline yields claims consumed downstream.

**Determinism:** segmentation is [T1] deterministic; extraction/decontext are
`[T3-gen]` pinned by `extractor_version`; under the mock everything is fully
reproducible (stability = 1.0). Every yes/no decision is an atom.

**How to extend:** bump `pins.extractor_version` and add a new
`config/prompts/<version>.json`; add a step by composing a new class into
`ClaimPipeline._build` and logging its atoms.
