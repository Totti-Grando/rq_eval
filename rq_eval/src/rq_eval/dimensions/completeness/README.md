# dimensions/completeness

**Design ref:** §2 completeness — MAJOR, two-tier nugget recall. The false-
*negative* axis: of what a good answer should contain, how much is present.

**Purpose:** split evaluator trust into a **requirement tier** (the reference,
generated/authored once then frozen) and, under it, gated atomic **units**
(nuggets) scored deterministically. The reference is honest about its assurance:
one of three **modes** is stamped on every result.

**Reference modes (`completeness.reference_mode`, stamped as `assurance_mode`):**
- `generated` (**default**, open-domain) — requirement facets generated per-question `[T3-gen]`.
- `archetype` (middle) — the requirement skeleton instantiated from ~8–12 fixed question shapes (`question_archetypes.yaml`).
- `templated` (strongest, closed-domain) — a human per-type checklist (`requirement_templates.yaml`); real coverage guarantee.
Bottom-up Tier-2 units stay extractive from source spans in every mode.

**Classes:**
- `ReferenceModeSelector` — selects the Tier-1 requirement set per mode; exposes the stamped `mode`.
- `RequirementTemplates` (+ `Requirement`) — [oracle] templated Tier-1 facets from versioned YAML.
- `ArchetypeTemplates` — [oracle] fixed question-shape skeletons (`archetype_version`).
- `RecallSample` — human should-contain sample → `recall_miss_rate` (completeness's honest error bar).
- `UnitDrafter` (+ `Unit`) — [T3-gen top-down + extractive bottom-up] Tier-2 units.
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
- `recall_miss_rate = |sampled facts not surfaced by any unit| / |sampled facts for this question|` (reported when the sample applies).
- Wilson 95% CI over (vital supported, vital total); abstain when vital total < `min_n`.

**Determinism:** Tier-1 is a fixed scaffold; the admissibility gate is now fully
T1/T2 (atomic structural + coref self-contained + **double-NLI** decidability) —
the per-unit admission judge is gone, replaced by a residual that fires only on
double-NLI disagreement (a world-knowledge unit). Assignment is T2; all scoring is
code → replays. Only Tier-2 unit *drafting* is generative (pinned by
`nuggetizer_version`); the frozen set carries version + corpus hash.

**How to extend:** pick the assurance mode with `completeness.reference_mode`;
edit `config/requirement_templates.yaml` / `config/question_archetypes.yaml` (the
oracles) to add types/shapes; point `completeness.recall_sample_path` at a labeled
should-contain sample; bump `pins.template_version` / `pins.archetype_version` /
`pins.nuggetizer_version`; tune `dedupe_tau`, `min_n`, `vital_weighting`.
