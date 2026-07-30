# scoring

**Design ref:** §0.5.4 replay guarantee (formulas + registry land here in B3);
the statistical library (Wilson CI, bands, off-ask cap, min-n) is added in B5.

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

**Calculations:**
- `mean = (Σ verdict) / n` over n atoms (0 if n = 0).
- `weighted_mean = (Σ verdict·w) / (Σ w)` (0 if Σw = 0).
- `conjunction_weighted_mean`: for each subject, `correct = AND(verdicts)`;
  `score = (Σ correct·w) / (Σ w)` with one weight per subject.
- _(B5 adds:_ Wilson 95% CI, band map `≥G→"G", ≥A→"A", else "R"`, off-ask cap,
  min-n abstention.)

**Determinism:** everything replays bit-for-bit — pure functions of the atoms,
no randomness, no I/O.

**How to extend:** add a `Formula` subclass, register it in `default_registry()`,
and have the dimension record that `formula_id`. Keep it a pure function of atoms
so replay stays model-free.
