# dimensions/task_success

**Design ref:** §4 task_success — MAJOR, goal accomplishment. Built strictly to
`response-quality-design.md` §4 (the work order's B9 is superseded).

**Purpose:** score whether the user's *objective* would actually be achieved
(fit to goal, not question). Genuinely Tier-3: the per-outcome verdicts are real
judge calls; the taxonomy + templates are pinned references.

**Classes:**
- `TaskTemplates` (+ `Outcome`) — [pinned] versioned task-type taxonomy; keyword classify.
- `ObjectiveInference` — [T3-gen] infer intent from the question.
- `OutcomeDecomposer` — [T3-gen] instantiate the template's outcomes for this instance.
- `TaskSuccessDimension` — orchestrates steps 1–5 → `DimensionResult`.

**Calculations:**
- `task_success = |achieved| / |required outcomes|` (`achieved_ratio` formula, `mean` over outcome atoms).
- impossible task: a well-scoped "can't be done because X" → `score = 1.0` (an `impossible_success` atom short-circuits the formula, like relevance's abstention).
- Wilson 95% CI over (achieved, required).
- partial is captured by the ratio (2/3 = 0.67); multi-goal weighting is left unweighted per step 5 (primacy weighting is a documented extension point).

**Determinism:** composition is code + replayable; classification is deterministic
(config taxonomy keywords). The genuine non-replayable residue is the per-outcome
judge verdicts + objective/outcome generation (pinned by the taxonomy version).

**How to extend:** edit `config/task_templates.yaml` to add task types / outcomes
/ cues; bump its `version`; add primacy weights per outcome for multi-goal
weighting.
