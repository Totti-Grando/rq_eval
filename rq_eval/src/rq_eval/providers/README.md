# providers

**Design ref:** build order B2 — provider interfaces + mock + live (the tier
adapters / portability layer). Enforces the booleans-only discipline at the
interface.

**Purpose:** every external dependency (Bedrock Claude, Titan, Guardrails,
fairseq-NLI, spaCy/coreferee) sits behind an abstract interface here, with a
deterministic `mock/` sibling and a `live/` sibling. Everything downstream
receives providers by dependency injection and never imports a concrete class;
construction is only via `ProviderFactory`, which reads config. This is what
makes the whole program build, run, and test offline (`providers.mode: mock`).

**Classes:**
- `JudgeProvider` — [T3] boolean judge; the ONLY method is `binary(...) -> {verdict: bool, reason}`. No numeric endpoint.
- `GeneratorProvider` — [T3-gen] pinned text generation (claims, units, objectives); returns text, never numbers.
- `EmbeddingProvider` — [T2] text → fixed-dim vectors.
- `GroundingProvider` — [T2] entailment/grounding → `raw_score` (thresholded in our code).
- `RelevanceProvider` — [T2] query↔response → raw score (thresholded in our code).
- `NlpProvider` — [T1/T2] sentence segmentation + coref (spaCy/coreferee; mock = regex/identity).
- `ProviderFactory` — config-selected construction; `Providers` — the injected bundle.

**Calculations:** none in the interfaces. Mock heuristics live in `mock/`
(see its README); float→boolean thresholding lives in the grader/dimension
layers, not here.

**Determinism:** mock providers are fully deterministic (seeded) and replay
bit-for-bit. Live providers are non-replayable model calls; their model+version
is stamped on every atom so drift is detectable (§0.5).

**How to extend:**
- Add a backend by subclassing the interface in `mock/` or `live/` and wiring it in `ProviderFactory`.
- Select a live grounding backend with `models.nli: bedrock | fairseq | mock`.
- Never construct a provider outside `ProviderFactory`.
