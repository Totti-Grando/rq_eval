# dimensions/accuracy

**Design ref:** §1 accuracy — MAJOR, **DAG resolution** over the claim graph
(§0.3). `accuracy = successful nodes / total nodes`, equal weight, counted **per
node**.

**Purpose:** score factual *truth*. A claim succeeds as a passing **axiom**
(Layer 1) or a validly-derived **dependent** (Layer 2). Two-layer, de-risked: the
axiom floor never depends on edge detection.

**Layers:**
- **Layer 1 (protected floor, built):** each claim scored as an independent axiom
  — `grounded ∧ source-adequate ∧ attributed`. No graph, no edges.
- **Layer 2 (additive, G5, `accuracy.dag_rescue_enabled`, default off):** bare
  claims that failed Layer 1 are rescued if their sub-DAG resolves to passing
  axioms through locally-valid steps. When off, accuracy = Layer 1 exactly.

**Responsiveness is deliberately NOT here.** Accuracy is truth, not relevance — a
true, well-sourced, correctly-cited but off-topic claim counts as accurate (and
scores low on relevance separately). The two axes stay orthogonal.

**Classes:**
- `AccuracyDimension` — orchestrates the per-node axiom (Layer 1) → `DimensionResult`.
- `ClaimAccuracy` (+ `ClaimAccuracyDeps`) — the three truth booleans + the per-node `axiom` verdict + the bare residual.
- (grounded / source-adequate / attributed are **imported** from groundedness §1 / `SourceQualityProvider` §3 / `AttributionProvider` §4 — the last two now set-ops over the support set `S`, no fresh NLI.)

**Calculations:**
- Per node: `axiom = grounded ∧ source-adequate ∧ attributed` (truth only).
  - grounded: imported per-claim `grounded` atom (§1); for a numeric claim also `numeric_match(claim, source, tolerance)`; bare claim → unsourced truth-judge [T3].
  - source-adequate: `SourceQualityProvider.adequate(..., claim_id)` (supports/corroboration read off `S`).
  - attributed: `AttributionProvider.attributed(claim_id, C)` = `C∩S≠∅ ∧ conformal`.
- `accuracy = successful / total` (`dag_resolution`: mean over per-node `axiom`/`derived` verdicts).
- axiom-to-derived ratio reported as a diagnostic (trivially 1.0 at Layer 1).
- Wilson 95% CI over (successful, #claims).

**Determinism:** grounding/attribution/numeric/composition replay bit-for-bit;
only the unsourced truth-judge residual is a non-replayable T3 call.
**Nexa collapse:** source-adequacy ≈ 1 and residue ≈ 0 ⇒ accuracy ≈ grounded-axiom rate.

**How to extend:** enable Layer 2 with `accuracy.dag_rescue_enabled` once the
edge-recall harness (§0.3/G4) clears the bar; tune `numeric_tolerance`.
