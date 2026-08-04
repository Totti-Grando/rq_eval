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
- `ClaimResponsiveness` — [T1+T2] per-claim `on_topic ∧ on_ask` → the exported `responsive` atom (**no judge**, DIVER-QA reform).
- `RelevanceDimension` — orchestrates steps 1–7 → `DimensionResult`.

**Calculations:**
- Method A: `AR = (1/N) Σ cos(E_gi, E_o)` (reverse-questions vs original question).
- on-topic (per claim): `relevance(question, claim) ≥ relevance_tau` `[T2]`.
- on-ask = `on_ask_nli ∨ on_ask_lex` (DIVER-QA): `on_ask_nli = entails(premise=claim,
  hypothesis=ask) == E` `[T2]` where `ask = T1Tools.ask_hypothesis(question)`;
  `on_ask_lex = key_term_overlap(question, claim) ≥ lexical_min_overlap` `[T1]`.
- responsive = `on_topic ∧ on_ask`.
- score = `relevance_capped_mean`: `mean(responsive)`, capped at `off_ask_cap`
  when the answer-level on-ask atom is False; `= 1.0` on abstention.
- Wilson 95% CI over (responsive count, #claims).

**Determinism:** Method B / cosine / **on-ask (NLI + lexical)** / composition all
replay bit-for-bit; the off-ask cap and abstention travel in atom roles/weights.
The **only** judge here is **abstention** (proper decline to an unanswerable
question); the on-ask path no longer calls a judge.

**How to extend:** switch methods via `relevance.method`; tune `relevance_tau`
and `off_ask_cap` in config; the responsive atom is the single import surface for
accuracy — do not recompute it elsewhere.
