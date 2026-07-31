# scoring

**Design ref:** §0.5.4 replay guarantee (formulas + registry, B3) + the
statistical library (Wilson CI, band map, off-ask cap, min-n abstention, B5).

**Purpose:** the pure-math layer. Every number in an output is produced here
from atom booleans/weights. Dimensions compute their score *through* a
registered `Formula` and store its `formula_id`, which is what makes the score
replayable. **No model or provider code is ever imported here.**

**Classes:**
- `Formula` — abstract pure score function over atoms; carries a `formula_id`.
- `FormulaRegistry` — `formula_id → Formula`; the single lookup used by replay.
- `MeanFormula` — [code] fraction of true verdicts.
- `WeightedMeanFormula` — [code] weight-normalized mean of verdicts.
- `ConjunctionWeightedMeanFormula` — [code] accuracy's per-subject AND then weighted mean.
- `WilsonInterval` — [code] Wilson 95% CI for a proportion.
- `BandMapper` — [code] score → G/A/R.
- `OffAskCap` — [code] cap a relevance score when the ask is missed.
- `MinNAbstention` — [code] abstain when n < min_n.
- `ConformalCalibrator` / `ConformalStratifier` — [code] split-conformal threshold + guarantee band (§5); marginal or per-stratum.

**Calculations:**
- `mean = (Σ verdict) / n` over n atoms (0 if n = 0).
- `weighted_mean = (Σ verdict·w) / (Σ w)` (0 if Σw = 0).
- `conjunction_weighted_mean`: for each subject, `correct = AND(verdicts)`;
  `score = (Σ correct·w) / (Σ w)` with one weight per subject.
- Wilson CI: `center = (p̂ + z²/2n)/(1+z²/n)`,
  `half = (z/(1+z²/n))·sqrt(p̂(1-p̂)/n + z²/4n²)`; `p̂ = successes/n`, z = 1.96.
- band map: `score ≥ G → "G"; score ≥ A → "A"; else "R"`.
- off-ask cap: `on_ask ? score : min(score, cap)`.
- min-n abstention: abstain iff `n < min_n`.
- conformal: `νᵢ = 1 − confidenceᵢ`; `k = ⌈(1−α)(n+1)⌉` (clamped to n);
  `τ̂ = k-th smallest ν`; retain iff `confidence ≥ 1 − τ̂`; band `[1−α, 1−α+1/(n+1)]`.

**Determinism:** everything replays bit-for-bit — pure functions of the atoms,
no randomness, no I/O.

**How to extend:** add a `Formula` subclass, register it in `default_registry()`,
and have the dimension record that `formula_id`. Keep it a pure function of atoms
so replay stays model-free.
