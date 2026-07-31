# dimensions/task_success

**Design ref:** §4 task_success — MAJOR, **verifier-routed** goal accomplishment
(new design v2; the work order's B9 and the earlier T3-only §4 are superseded).

**Purpose:** score whether the user's *objective* would actually be achieved.
Not irreducibly T3: each required outcome routes to the cheapest verifier that
fits, and the judge fires **only** on `adequacy` outcomes — so for
executable/structured tasks the dimension is effectively T1.

**Classes:**
- `TaskTemplates` (+ `Outcome`) — [pinned] versioned taxonomy; each outcome tagged with its verifier + weight + params.
- `ObjectiveInference` / `OutcomeDecomposer` — [T3-gen] infer intent; instantiate outcomes.
- `verifiers/` — one verifier per routing-table tag (see its README).
- `VerifierRouter` — dispatches an outcome to its tagged verifier.
- `TaskSuccessDimension` — orchestrates steps 1–5 → `DimensionResult`.

**Routing (design §4 table):**
- artifact_presence / executable / state / constraint → **[T1]**
- coverage → **[T2 NLI]**   ·   grounded / responsive → **[import]** (reuse accuracy/relevance graders)
- adequacy → **[T3]** judge (the only judge call)

**Calculations:**
- `task_success = Σ achieved·w / Σ w` over required outcomes (`task_success_weighted`).
- impossible task → `1.0` (an `impossible_success` atom short-circuits the formula).
- Wilson 95% CI over (achieved, required); partial captured by the ratio.

**Determinism:** for code/SQL/structured tasks the score is T1 (execution/state/
constraint) — replays bit-for-bit; the T3 residue is confined to `adequacy`
outcomes. Execution is gated by `task_success.execution_sandbox` (default off →
deterministic text heuristic; on → injected `ExecutionSandbox`). Every outcome's
verifier tag + result is an `AtomRecord`.

**How to extend:** edit `config/task_templates.yaml` (bump `version`) to add task
types / outcomes / verifier tags / weights; add a verifier class under
`verifiers/` and register it in the router; wire a real `ExecutionSandbox` on the
target for true execution.
