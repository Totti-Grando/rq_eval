# dimensions/completeness

**Design ref:** §2 completeness — MAJOR, two-tier nugget recall. The false-
*negative* axis: of what a good answer should contain, how much is present.

**Purpose:** split evaluator trust into a fixed **requirement tier** (the
structural oracle, no AI recall risk on coverage) and, under it, gated atomic
**units** (nuggets). Score strict vital recall, report requirement coverage.

**Classes:**
- `RequirementTemplates` (+ `Requirement`) — [oracle] Tier-1 facets from versioned YAML; question-type classify.
- `UnitDrafter` (+ `Unit`) — [T3-gen] Tier-2 units, top-down + bottom-up.
- `UnitAdmissibilityGate` — [T1+T2] atomic (structural) + self-contained (coref) + entailment-decidable (**double-NLI**: `entails(answer,unit)` vs `entails(answer+sources,unit)` agree) → freeze; only disagreements hit a reference-grounded residual judge.
- `UnitDeduper` — [T2] embedding-cosine dedupe.
- `UnitAssigner` — [T2] per-unit support (answer = premise, unit = hypothesis).
- `TwoLevelScoring` — [code] per-requirement recall + requirement coverage + weighted recall.
- `CompletenessDimension` — orchestrates steps 1–8 → `DimensionResult`.

**Calculations:**
- headline `score = strict_vital_recall = |vital supported| / |vital|` (`mean` over vital support atoms).
- `per_requirement_recall(r) = supported units in r / units in r`.
- `requirement_coverage = |requirements with ≥1 supported unit| / |requirements|`.
- `weighted_recall = Σ recall(r)·w(r) / Σ w(r)`, `w = 2.0` for vital requirements when `vital_weighting`, else 1.0.
- Wilson 95% CI over (vital supported, vital total); abstain when vital total < `min_n`.

**Determinism:** Tier-1 is a fixed scaffold; the admissibility gate is now fully
T1/T2 (atomic structural + coref self-contained + **double-NLI** decidability) —
the per-unit admission judge is gone, replaced by a residual that fires only on
double-NLI disagreement (a world-knowledge unit). Assignment is T2; all scoring is
code → replays. Only Tier-2 unit *drafting* is generative (pinned by
`nuggetizer_version`); the frozen set carries version + corpus hash.

**How to extend:** edit `config/requirement_templates.yaml` (the oracle) to add
question-types/facets; bump `pins.template_version` / `pins.nuggetizer_version`;
tune `dedupe_tau`, `min_n`, `vital_weighting` in config.
