# rq_eval — Design → Code Navigation Index

Every design subcomponent and the exact file + class that implements it. Paths
are relative to `rq_eval/`. Specs: `../response-quality-design.md` (RQ) and
`../evidence-truthfulness-design.md` (E&T). For the *why*/tiers see `GUIDE.md`;
for the layer diagram see `ARCHITECTURE.md`.

Legend for tier: **T1** code · **T2** fixed model (thresholded in code) ·
**T3g** pinned generation · **T3** judge · **oracle** human-maintained config.

---

## 0. Shared foundation (used by every dimension)

| Concern | File | Class(es) |
|---|---|---|
| Config schema + loader (sole reader) | `src/rq_eval/config.py` | `Config` (+ nested `*Config`), `load_config`, `load_yaml`, `get_config` |
| Typed records (contracts) | `src/rq_eval/contracts.py` | `ContextChunk`, `EvalInput`, `Claim`, `Triplet`, `AtomRecord`, `DimensionResult`, `CalibrationExample` |
| Provider interfaces + result types | `src/rq_eval/providers/base.py` | `JudgeProvider`, `GeneratorProvider`, `EmbeddingProvider`, `GroundingProvider`+`EntailmentResult`, `RelevanceProvider`, `NlpProvider`, `ResolverProvider`, `SourceQualityProvider`, `AttributionProvider`+`AttributionResult` |
| Construction (the ONLY constructor) | `src/rq_eval/providers/factory.py` | `ProviderFactory`, `Providers` |
| Atom model+version stamps | `src/rq_eval/providers/model_stamp.py` | `ModelStamp` |
| Mock impls (lexical, offline) | `src/rq_eval/providers/mock/*.py` | `MockJudge/Generator/Embedding/Grounding/Relevance/Nlp/ResolverProvider`, `DeterministicText` |
| Live impls (Bedrock/spaCy/fairseq/urllib) | `src/rq_eval/providers/live/*.py` | `Bedrock*`, `Titan*`, `Guardrail*`, `Fairseq*`, `Spacy*`, `LiveResolverProvider`, `BedrockSession`, `PromptPrep` |
| Tier adapters (graders) | `src/rq_eval/graders/` | `T1Tools` (`t1.py`), `GroundingGrader`, `RelevanceGrader`, `JudgeGrader` |
| Pure scoring | `src/rq_eval/scoring/` | `FormulaRegistry`/`Formula` (`registry.py`), formulas (`formulas.py`), `WilsonInterval`, `BandMapper`, `OffAskCap`/`MinNAbstention` (`aggregation.py`), conformal (`conformal.py`) |
| Audit + replay | `src/rq_eval/audit/` | `AtomStore`/`Jsonl…`/`Sqlite…`, `AtomStoreFactory`, `AtomLogger`, `Clock`, `ReplayVerifier`, `CalibrationStore` |
| Orchestration / output | `src/rq_eval/` | `Evaluator`+`EvaluationResult`+`evaluate` (`runner.py`), `ReportRenderer` (`report.py`), `FixtureSuite`/`FixtureCase` (`fixtures.py`) |
| Dimension base + shared exports | `src/rq_eval/dimensions/` | `Dimension` (`base.py`), `ResponsivenessExport` (`responsiveness.py`) |

**Config data oracles (human-maintained, pinned):**
`config/prompts/claim-extractor-v1.json`, `config/requirement_templates.yaml`,
`config/task_templates.yaml`, `config/reliability_list.yaml`,
`config/calibration/calibration-v1.jsonl`.

---

## RQ §0 — shared claim-extraction pipeline  → `src/rq_eval/pipeline/`

| Design step | Tier | File · class |
|---|---|---|
| 1 Segment | T1 | `segmenter.py` · `Segmenter` |
| 2 Select verifiable spans | T3 | `span_selector.py` · `VerifiableSpanSelector` |
| 3 Claimify disambiguate + extract | T3 / T3g | `claim_extractor.py` · `ClaimExtractor` |
| 4 Decontextualize (coref + carry) | T2 + T3 | `decontextualizer.py` · `Decontextualizer` |
| 5 Pin & measure stability | code | `stability.py` · `StabilityHarness`; prompts `prompts.py` · `PromptLibrary` |
| Orchestrator / output | — | `pipeline.py` · `ClaimPipeline`, `PipelineResult` |

## RQ §0.5 — contracts, records & audit
Records → `contracts.py`. Atom log + replay → `audit/` (`AtomLogger`,
`atom_store*.py`, `replay.py::ReplayVerifier`). Formula registry → `scoring/registry.py`.

## RQ §1 — accuracy  → `src/rq_eval/dimensions/accuracy/`

| Design step | Tier | File · class / source |
|---|---|---|
| 1 grounded? | T2 | **imported** from groundedness → `groundedness/export.py::GroundednessExport`; consumed in `claim_accuracy.py::ClaimAccuracy._grounded` |
| 2 source-adequate? | T1/T2 | **imported** from `source_quality/provider.py::SourceQualityProviderImpl` |
| 3 attributed? | T2 | **imported** from `source_attribution/provider.py::AttributionProviderImpl` |
| 4 responsive? | T2 | **imported** from relevance → `responsiveness.py::ResponsivenessExport` |
| 5 Compose (conjunction, weighted mean) | code | `scoring/formulas.py::ConjunctionWeightedMeanFormula` |
| 6 Residual: unsourced truth-judge | T3 | `claim_accuracy.py` (`residual_truth`) |
| 6 Residual: inferred → inference-validity | T2* | `stubs.py::InferenceValidityStub` (**stub** — out-of-scope category) |
| 7 Importance weighting (toggle) | code | `importance.py::ImportanceWeights` |
| Orchestrator | — | `accuracy.py::AccuracyDimension` (+ `claim_accuracy.py::ClaimAccuracyDeps`) |

## RQ §2 — completeness  → `src/rq_eval/dimensions/completeness/`

| Design step | Tier | File · class |
|---|---|---|
| 1 Tier-1 requirements | oracle | `requirement_templates.py::RequirementTemplates`, `Requirement` (data: `config/requirement_templates.yaml`) |
| 2 Tier-2 units (top-down + bottom-up) | T3g | `unit_drafter.py::UnitDrafter` (unit shape: `unit.py::Unit`) |
| 3 Admissibility gate (atomic/self-contained/decidable) | T1+T2+T3 | `admissibility_gate.py::UnitAdmissibilityGate` |
| 4 Merge/dedupe | T2 | `deduper.py::UnitDeduper` |
| 5 Label vital/okay | (from oracle) | inherited from Tier-1 `vital` flag (see `GUIDE.md` §9) |
| 6 Assign (unit support) | T2 | `unit_assigner.py::UnitAssigner` |
| 7 Two-level scoring + strict vital recall | code | `two_level_scoring.py::TwoLevelScoring` + `scoring/formulas.py::MeanFormula` |
| 8 Wilson CI + min-n abstain + version/corpus_hash | code | `scoring/wilson.py`, `scoring/aggregation.py::MinNAbstention` |
| Orchestrator | — | `completeness.py::CompletenessDimension` |

## RQ §3 — relevance  → `src/rq_eval/dimensions/relevance/`

| Design step | Tier | File · class |
|---|---|---|
| 1 Method A (reverse-questions + cosine) | T3g+T2 | `method_a.py::MethodAReverseQuestions` |
| 2 Method B (guardrail relevance, default) | T2 | `method_b.py::MethodBGuardrail` |
| 3 Answer-level on-topic / on-ask | T2 | `relevance.py::RelevanceDimension` |
| 4 Claim-level responsive atom (exported) | T2 | `claim_responsiveness.py::ClaimResponsiveness` → `responsiveness.py::ResponsivenessExport` |
| 5 Combine + off-ask cap | code | `scoring/formulas.py::RelevanceCappedMeanFormula`, `scoring/aggregation.py::OffAskCap` |
| 6 Abstention (decline/unanswerable) | T3 | `relevance.py::RelevanceDimension._maybe_abstain` |
| 7 Residual (different sub-question) | T3 | `claim_responsiveness.py` (residual grader) |

## RQ §4 — task_success (verifier-routed)  → `src/rq_eval/dimensions/task_success/`

| Design step | Tier | File · class |
|---|---|---|
| 1 Infer objective | T3g | `objective.py::ObjectiveInference` |
| 2 Classify + verifier-typed template | (kw)/oracle | `task_templates.py::TaskTemplates`, `Outcome` (data: `config/task_templates.yaml`) |
| 3 Decompose outcomes | T3g | `objective.py::OutcomeDecomposer` |
| 4 Route each outcome to its verifier | T1/T2/import/T3 | `verifiers/base.py::VerifierRouter` + one class per tag ↓ |
| — artifact-presence | T1 | `verifiers/presence.py::PresenceVerifier` |
| — executable/test | T1 | `verifiers/execution.py::ExecutionVerifier` (+ `ExecutionSandbox` iface) |
| — state/end-condition | T1 | `verifiers/state.py::StateVerifier` |
| — constraint-satisfaction | T1 | `verifiers/constraint.py::ConstraintVerifier` |
| — coverage | T2 | `verifiers/coverage.py::CoverageVerifier` |
| — grounded/responsive (import) | import | `verifiers/import_verifier.py::ImportVerifier` |
| — adequacy (residue) | T3 | `verifiers/adequacy.py::AdequacyVerifier` |
| 5 Compute Σ achieved·w / Σ w | code | `scoring/formulas.py::TaskSuccessWeightedFormula` |
| Orchestrator | — | `task_success.py::TaskSuccessDimension` |

## E&T §0 — claims → triplets  → `src/rq_eval/pipeline/triplets.py`
`ClaimTripletExtractor` (T3g), `TripletStabilityHarness`. Triplet shape:
`contracts.py::Triplet`.

## E&T §1 — groundedness  → `src/rq_eval/dimensions/groundedness/`

| Design step | Tier | File · class |
|---|---|---|
| 1 Similarity pre-filter | T1 | `prefilter.py::SimilarityPreFilter` |
| 2 Three-way entailment per triplet | T2 | `graders/grounding_grader.py::GroundingGrader` → `GroundingProvider.entails` |
| 3 Score `|E|/|total|` | code | `scoring/formulas.py::MeanFormula` |
| Export (per-claim grounded + confidences) | — | `export.py::GroundednessExport` |
| Orchestrator | — | `groundedness.py::GroundednessDimension` |

## E&T §2 — hallucination  → `src/rq_eval/dimensions/hallucination/`

| Design step | Tier | File · class |
|---|---|---|
| 1 Unsupported rate (N/C split) | T2/code | `hallucination.py::HallucinationDimension` + `scoring/formulas.py::UnsupportedRateFormula` |
| 2 Fabricated-citation existence **gate** | T1 | `fabrication_gate.py::FabricationGate`, `FabricationResult` (uses `ResolverProvider`) |
| 3 Score + report split | code | `hallucination.py::HallucinationDimension` |

## E&T §3 — source_quality  → `src/rq_eval/dimensions/source_quality/`

| Design step (property) | Tier | File · class |
|---|---|---|
| 1–5 reachable / dated&fresh / authored / reputable-domain / corroborated | T1 | `scorer.py::SourceQualityScorer` (domain oracle: `reliability_list.py::ReliabilityList` + `config/reliability_list.yaml`) |
| 6 Supports the claim | T2 | `scorer.py::SourceQualityScorer` (grounding) |
| 7 Disinterested (sampled) | T3 | `scorer.py::SourceQualityScorer` |
| 8 Score mean(properties) → adequate | code | `provider.py::SourceQualityProviderImpl` (accuracy import) |
| Orchestrator | — | `source_quality.py::SourceQualityDimension` |

## E&T §4 — source_attribution  → `src/rq_eval/dimensions/source_attribution/`

| Design step | Tier | File · class |
|---|---|---|
| 1 Per-claim citation support (3/4-way) | T2 | `source_attribution.py` + `labels.py::AttributionLabeler` |
| 2 ALCE recall + precision | code | `alce.py::AlceScorer` |
| 3 Precision-favoring + no-citation excluded | code | `source_attribution.py::SourceAttributionDimension` |
| 4 Calibrated uncertainty (→ conformal) | code | `export.py::AttributionExport` → `scoring/conformal.py` |
| 5 Score = precision; attributed? = Attributable ∧ conformal | code | `provider.py::AttributionProviderImpl` (accuracy import) |

## E&T §5 — conformal factuality  → `src/rq_eval/scoring/conformal.py`

| Design step | Tier | File · class |
|---|---|---|
| 1 Confidence | T2 | `GroundingProvider.entails().raw_score` |
| 2 Calibrate `τ̂` quantile | code | `conformal.py::ConformalCalibrator` |
| 3 Retain & guarantee band | code | `conformal.py::ConformalCalibrator`/`ConformalResult` |
| Per-stratum calibration | code | `conformal.py::ConformalStratifier` |
| Calibration set (pinned, per-stratum) | oracle | `audit/calibration.py::CalibrationStore` + `config/calibration/calibration-v1.jsonl` |
| Wiring + band stamping | — | `runner.py::Evaluator._calibrate_conformal` / `_stamp_conformal` |

---

## Tests → what they cover (`tests/`)
`test_config*.py` (config + single-source), `test_providers.py`, `test_contracts.py`,
`test_audit_replay.py`, `test_pipeline.py`, `test_triplets.py`, `test_graders_t1.py`,
`test_scoring.py`/`test_scoring_pure.py`/`test_conformal.py`, `test_relevance.py`,
`test_accuracy.py`, `test_completeness.py`, `test_task_success.py`,
`test_groundedness.py`, `test_hallucination.py`, `test_source_quality.py`,
`test_source_attribution.py`, `test_calibration.py`, `test_runner.py`,
`test_navigation.py` (verifies this map's ARCHITECTURE table against the tree).

## Intentional deviations
Documented in `GUIDE.md` §9 and per-dimension READMEs. Notably: RQ §0.2 built as a
judge boolean; RQ §1.6 inference-validity is a stub; RQ §2.5 vital labels from the
oracle (not a judge); RQ §4.2 classify is deterministic; task_success executable is
a heuristic (sandbox is interface-only); Method A reimplemented (not `ragas`).
