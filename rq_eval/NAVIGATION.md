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
| Provider interfaces + result types | `src/rq_eval/providers/base.py` | `ScoringJudge`, `ExplanationJudge`, `GeneratorProvider`, `EmbeddingProvider`, `GroundingProvider`+`EntailmentResult`, `RelevanceProvider`, `NlpProvider`, `ResolverProvider`, `SourceQualityProvider`, `AttributionProvider`+`AttributionResult` |
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
`config/prompts/claim-extractor-v1.json` (realizer prompt),
`config/requirement_templates.yaml`, `config/question_archetypes.yaml`,
`config/completeness_recall_sample.jsonl`, `config/task_templates.yaml`,
`config/reliability_list.yaml`, `config/coi_denylist.yaml`,
`config/calibration/calibration-v1.jsonl`.

---

## RQ §0 — shared claim-extraction pipeline  → `src/rq_eval/pipeline/`

| Design step | Tier | File · class |
|---|---|---|
| 1 Segment | T1 | `segmenter.py` · `Segmenter` |
| 2 Select verifiable spans (lexical hedge/opinion filter) | T1 | `span_selector.py` · `VerifiableSpanSelector` (`T1Tools.is_verifiable`) |
| 3 Decompose into content-unit clauses (parse; abstractive flagged) | T1 | `claim_extractor.py` · `ClaimExtractor` (`NlpProvider.parse_clauses`) |
| 3a Optional surface-realizer (droppable, off by default) | T2 pinned | `claim_extractor.py` (`extraction.realizer_enabled`) |
| 4 Decontextualize (coref + structural self-contained) | T1/T2 | `decontextualizer.py` · `Decontextualizer` (`T1Tools.has_leading_pronoun`) |
| 5 Pin & measure stability | code | `stability.py` · `StabilityHarness`; realizer prompt `prompts.py` · `PromptLibrary` |
| Orchestrator / output | — | `pipeline.py` · `ClaimPipeline`, `PipelineResult` |

## RQ §0.5 — contracts, records & audit
Records → `contracts.py`. Atom log + replay → `audit/` (`AtomLogger`,
`atom_store*.py`, `replay.py::ReplayVerifier`). Formula registry → `scoring/registry.py`.

## RQ §0.3 — shared claim graph  → `src/rq_eval/pipeline/`

| Design step | Tier | File · class |
|---|---|---|
| Typed nodes (independent / inference / indexical + binding) | T1 + §1 byproduct | `claim_graph.py::ClaimGraphBuilder`, `ClaimGraph`, `GraphNode` |
| Edge detection (backward premise-BFS, minimal-complete, numeric) | T1 propose + T2 confirm | `edge_detection.py::EdgeDetector` |
| Edge-recall harness (the honest error bar, gates Layer 2) | code | `validation/edge_recall.py::EdgeRecallHarness` |
| Visualization (resolved graph → JSON/PNG diagnostic) | code | `graph_viz.py::GraphVisualizer` |

## RQ §1 — accuracy (two-layer DAG resolution)  → `src/rq_eval/dimensions/accuracy/`

| Design step | Tier | File · class / source |
|---|---|---|
| axiom: grounded? | T2 | **imported** from groundedness → `groundedness/export.py::GroundednessExport`; `claim_accuracy.py::ClaimAccuracy._grounded` |
| axiom: source-adequate? | T1 | **imported** from `source_quality/provider.py::SourceQualityProviderImpl` (supports/corroboration off `S`) |
| axiom: attributed? (C∩S) | T2 | **imported** from `source_attribution/provider.py::AttributionProviderImpl` |
| Layer 1 node verdict = AND(3 truth booleans) | code | `claim_accuracy.py` (`axiom` atom) — **responsiveness removed** |
| Layer 2 DAG rescue (flagged) | code + T2 edges | `accuracy.py::AccuracyDimension._rescue` over the shared `ClaimGraph` (`derived` atom) |
| Residual: unsourced truth-judge | T3 | `claim_accuracy.py` (`residual_truth`) |
| Score `successful / total` (per node) | code | `scoring/formulas.py::DagResolutionFormula` |
| Orchestrator | — | `accuracy.py::AccuracyDimension` (+ `claim_accuracy.py::ClaimAccuracyDeps`) |

## RQ §2 — completeness  → `src/rq_eval/dimensions/completeness/`

| Design step | Tier | File · class |
|---|---|---|
| 1 Tier-1 requirements (mode-selected) | oracle / T3g | `reference.py::ReferenceModeSelector` → `requirement_templates.py::RequirementTemplates` (templated) · `archetype_templates.py::ArchetypeTemplates` (archetype, data: `config/question_archetypes.yaml`) · generator (generated, **default**) |
| 1a Assurance mode + human recall miss-rate | code | stamped `DimensionResult.assurance_mode`; `recall_sample.py::RecallSample` → `extra.recall_miss_rate` |
| 2 Tier-2 units (top-down + extractive bottom-up) | T3g + T1 | `unit_drafter.py::UnitDrafter` (unit shape: `unit.py::Unit`) |
| 3 Admissibility gate (atomic/self-contained/decidable) | T1+T2 (T3 residual only on double-NLI disagreement) | `admissibility_gate.py::UnitAdmissibilityGate` |
| 4 Merge/dedupe | T2 | `deduper.py::UnitDeduper` |
| 5 Label vital/okay | (from oracle) | inherited from Tier-1 `vital` flag (see `GUIDE.md` §9) |
| 6 Assign (unit support) | T2 | `unit_assigner.py::UnitAssigner` |
| 7 Two-level scoring + strict vital recall | code | `two_level_scoring.py::TwoLevelScoring` + `scoring/formulas.py::MeanFormula` |
| 8 Wilson CI + min-n abstain + version/corpus_hash | code | `scoring/wilson.py`, `scoring/aggregation.py::MinNAbstention` |
| Orchestrator | — | `completeness.py::CompletenessDimension` |

## RQ §3 — relevance  → `src/rq_eval/dimensions/relevance/`

| Design step | Tier | File · class |
|---|---|---|
| **Layer 1 (default)** on-ask primitive + responsive export (on-topic ∧ on-ask, DIVER-QA) | T1+T2 (no judge) | `claim_responsiveness.py::ClaimResponsiveness` (`ClaimSignals`) → `responsiveness.py::ResponsivenessExport` |
| Layer 1 score = capped mean of responsive | code | `scoring/formulas.py::RelevanceCappedMeanFormula` |
| **Layer 2** (`tree_enabled`, off) edges | — | read from the shared `ClaimGraph` (`edges.py::Edge` view; relevance builds none) |
| Layer 2 Anchors (on-ask seed + centrality + conformal recall) | T2+code | `anchors.py::AnchorSelector` (`AnchorResult`) |
| Layer 2 Support tree (depth-graded reachability, max_hops, depth_decay) | code | `tree.py::SupportTree` |
| Layer 2 Orphan resolution (off-topic / stranded-veracity / background) | T1+T2 | `orphans.py::OrphanResolver`; routes to `providers/consistency.py::ConsistencyProvider` |
| Layer 2 score = depth-graded mean + off-ask cap | code | `scoring/formulas.py::RelevanceTreeCappedMeanFormula` |
| Abstention (decline/unanswerable, reference-grounded) | T3 | `relevance.py::RelevanceDimension._maybe_abstain` |
| Method A / B (answer-level diagnostics) | T3g+T2 / T2 | `method_a.py::MethodAReverseQuestions`, `method_b.py::MethodBGuardrail` |
| Orchestrator | — | `relevance.py::RelevanceDimension` |

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
`ClaimTripletExtractor` (**parse-first**: `T1` positional/dependency S-P-O via
`NlpProvider.parse_clauses` + `T1Tools.parse_triplet`; `T3-gen` generator only for
the nested/abstractive residual), `TripletStabilityHarness`. Triplet shape:
`contracts.py::Triplet`.

## E&T §1 — groundedness  → `src/rq_eval/dimensions/groundedness/`

| Design step | Tier | File · class |
|---|---|---|
| 1 Similarity pre-filter (top-`groundedness_k`) | T1 | `prefilter.py::SimilarityPreFilter.select_k` |
| 2 Per-chunk entailment → support set `S={chunk:E}` | T2 | `groundedness.py::GroundednessDimension._assess_triplet` → `GroundingProvider.entails` |
| 3 Score `|S≠∅|/|total|` | code | `scoring/formulas.py::MeanFormula` |
| Export the support set `S` (per-claim chunks + docs) + grounded + confidences | — | `export.py::GroundednessExport` (read by §3/§4) |
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
| 1–4 reachable / dated&fresh / authored / reputable-domain | T1 | `scorer.py::SourceQualityScorer` (domain oracle: `reliability_list.py::ReliabilityList` + `config/reliability_list.yaml`) |
| 5 Corroborated (`|distinct docs in S| ≥ min`) | T1 (from `S`) | `scorer.py::SourceQualityScorer` (reads `GroundednessExport`, **no NLI**) |
| 6 Supports the claim (`S ≠ ∅`) | T1 (from `S`) | `scorer.py::SourceQualityScorer` (imported, **no NLI**) |
| 7 Disinterested (COI rule + sampled residual) | T1/T3 | `scorer.py::SourceQualityScorer` + `coi.py::CoiRule` |
| 8 Score mean(properties) → adequate | code | `provider.py::SourceQualityProviderImpl` (accuracy import, per `claim_id`) |
| Orchestrator | — | `source_quality.py::SourceQualityDimension` |

## E&T §4 — source_attribution  → `src/rq_eval/dimensions/source_attribution/`

| Design step | Tier | File · class |
|---|---|---|
| 1 Resolve cited set `C` (explicit regex + implicit scope) | T1 | `citations.py::resolve_explicit`, `ScopePropagator` |
| 2 Attribution = set-op `C∩S` over §1's support set (**no NLI**) | code | `source_attribution.py::SourceAttributionDimension` |
| 3 ALCE recall + precision + `C−S`/`S−C` diagnostics | code | `alce.py::AlceScorer` + `source_attribution.py` |
| 4 Calibrated uncertainty (→ conformal) | code | `export.py::AttributionExport` → `scoring/conformal.py` |
| 5 attributed? = `C∩S≠∅` ∧ conformal | code | `provider.py::AttributionProviderImpl` (accuracy import) |

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
