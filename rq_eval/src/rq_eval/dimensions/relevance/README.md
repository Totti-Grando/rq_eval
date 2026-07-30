# dimensions/relevance

**Design ref:** §3 relevance — on-topic + responsiveness. Built **first** because
accuracy (§1) imports the per-claim responsive atom computed here.

**Purpose:** score how well the answer fits the *question* (on-topic + on-ask),
and publish the per-claim responsive boolean to `ResponsivenessExport` for
accuracy. Two methods behind `relevance.method`: Method A (diagnostic) and
Method B (deterministic-first gate, default).

**Classes:**
- `MethodAReverseQuestions` — [T3-gen + T2 cosine] RAGAS answer-relevancy (diagnostic).
- `MethodBGuardrail` — [T2] raw query↔response relevance score (default gate).
- `ClaimResponsiveness` — [T2] per-claim `on_topic ∧ on_ask` → the exported `responsive` atom; thin [T3] residual for on-topic∧¬on-ask.
- `RelevanceDimension` — orchestrates steps 1–7 → `DimensionResult`.

**Calculations:**
- Method A: `AR = (1/N) Σ cos(E_gi, E_o)` (reverse-questions vs original question).
- on-topic (per claim): `relevance(question, claim) ≥ relevance_tau`.
- on-ask (per claim/answer): judge coverage of the question's terms `≥ 0.5`.
- responsive = `on_topic ∧ on_ask`.
- score = `relevance_capped_mean`: `mean(responsive)`, capped at `off_ask_cap`
  when the answer-level on-ask atom is False; `= 1.0` when a proper decline to an
  unanswerable question is detected (abstention).
- Wilson 95% CI over (responsive count, #claims).

**Determinism:** Method B / cosine / composition are deterministic + replayable;
the off-ask cap and abstention travel in atom roles/weights so the score replays
from atoms alone. Only the thin residual + abstention detection are judge calls.

**How to extend:** switch methods via `relevance.method`; tune `relevance_tau`
and `off_ask_cap` in config; the responsive atom is the single import surface for
accuracy — do not recompute it elsewhere.
