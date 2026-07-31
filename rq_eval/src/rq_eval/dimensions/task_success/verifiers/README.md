# dimensions/task_success/verifiers

**Design ref:** §4 v2 routing table — one verifier per outcome tag; the judge
fires only on `adequacy`.

**Purpose:** decide a single required outcome (achieved?) with the cheapest
verifier that fits, returning the logged `AtomRecord` (role `outcome`, tier =
the verifier's tier). `VerifierRouter` dispatches by tag.

**Classes:**
- `Verifier` (ABC) + `VerifyContext` + `VerifierRouter` — the interface + dispatch.
- `PresenceVerifier` — [T1] answer contains any required pattern (artifact present).
- `ExecutionVerifier` — [T1] run it (sandbox, gated) or a code+run-claim heuristic; `ExecutionSandbox` is the pluggable interface.
- `StateVerifier` — [T1] terminal-state match against `params.expected` (exact/substring).
- `ConstraintVerifier` — [T1] include/exclude tokens + word-count bounds.
- `CoverageVerifier` — [T2] grounding(answer supports the requirement) ≥ tau.
- `ImportVerifier` — [import] responsive = relevance(question, answer); grounded = grounding(context, answer).
- `AdequacyVerifier` — [T3] the sole judge call, per-outcome binary on soft outcomes.

**Calculations:** each returns a boolean; per-verifier rules:
- presence: `any(pattern in answer)`.
- executable (heuristic): `any(signal in answer) AND any(run_claim in answer)`.
- state: `expected == answer` (exact) or `expected in answer` (substring); True if no expected.
- constraint: `all(includes) AND not any(excludes) AND min_words ≤ |words| ≤ max_words`.
- coverage / import: T2 grounding/relevance thresholded in the grader.
- adequacy: judge overlap of the outcome cues vs the answer.

**Determinism:** T1 verifiers replay bit-for-bit; T2 (coverage/import) replay from
stamped fixed-model outputs; only `adequacy` (T3) is non-replayable.

**How to extend:** add a `Verifier` subclass, register its tag in the router
(`TaskSuccessDimension._build_router`), and reference the tag from a template.
