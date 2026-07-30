# dimensions

**Design ref:** §1–§4 — the four Response-Quality dimensions, one subpackage
each, over the shared `Dimension` base.

**Purpose:** each dimension is a class that turns an `EvalInput` (+ injected
providers, graders, atom logger, cached claims, and shared state) into a
`DimensionResult`. Relevance runs first and exports per-claim responsiveness;
accuracy imports it.

**Classes:**
- `Dimension` — abstract base; `evaluate(EvalInput) -> DimensionResult`.
- `ResponsivenessExport` — the §3→§1 hand-off (per-claim responsive verdict + atom id).

**Calculations:** none here — each subfolder (`relevance/`, `accuracy/`,
`completeness/`, `task_success/`) documents its own formulas.

**Determinism:** dimensions log every boolean as an atom and compute their score
through a registered `scoring/` formula, so results replay from the atom log.

**How to extend:** add a subpackage implementing `Dimension`; inject it in the
runner; register any new score formula in `scoring/`.
