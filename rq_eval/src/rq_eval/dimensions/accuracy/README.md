# dimensions/accuracy

**Design ref:** §1 accuracy — MAJOR, derived. Computed over the cached claims by
running four booleans each, then composing in code.

**Purpose:** score factual correctness of the answer's verifiable claims. Not an
independent scorer — it derives from the §0 claims and *imports* responsiveness
from §3 (never recomputes it).

**Classes:**
- `AccuracyDimension` — orchestrates the four booleans + composition → `DimensionResult`.
- `ClaimAccuracy` (+ `ClaimAccuracyDeps`) — the per-claim booleans and residual/inferred handling.
- `ImportanceWeights` — per-claim weight (vital/okay from §2; toggle).
- `SourceQualityStub` / `InferenceValidityStub` — typed stubs for imported categories (Nexa defaults).

**Calculations:**
- Per claim: `correct = grounded ∧ source_adequate ∧ attributed ∧ responsive`.
  - grounded: `grounding(context, claim) ≥ grounding_tau`; for a numeric claim also `numeric_match(claim, source, tolerance)`.
  - attributed: `grounding(cited_chunk, claim) ≥ attribution_tau` (True if no citation).
  - source_adequate: from `source_quality` stub (Nexa ≈ True).
  - responsive: imported atom from §3.
- `accuracy = Σ correct·w / Σ w` (`conjunction_weighted_mean`; w = importance weight).
- Residual: unsourced claim → truth-judge [T3]; inferred (grounded by whole
  context but no single chunk) → inference-validity stub.
- Wilson 95% CI over (correct claims, #claims).

**Determinism:** grounding/attribution/numeric/composition replay bit-for-bit;
only the unsourced truth-judge residual is a non-replayable T3 call.
**Nexa collapse:** source-adequacy ≈ 1 and residue ≈ 0 ⇒ accuracy ≈ groundedness.

**How to extend:** wire real `source_quality` / `inference-validity` modules in
place of the stubs; supply completeness's vital weights via `ImportanceWeights`.
