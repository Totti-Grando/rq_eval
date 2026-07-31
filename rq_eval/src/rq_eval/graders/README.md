# graders

**Design ref:** build order B5 — T1 grader toolbox; the T2/T3 tier adapters used
by every dimension.

**Purpose:** turn raw provider outputs into audited booleans. T1 tools are pure
deterministic checks; T2/T3 adapters call a provider, apply the config threshold
**in code** (never in the provider), and log one `AtomRecord` per check. This is
the layer that enforces "float → boolean thresholding happens in our code."

**Classes:**
- `T1Tools` — [T1] pure: numeric exact-match, citation membership, atomicity/conjunction split, word count.
- `JudgeGrader` — [T3] ask a yes/no question, log the atom, return the bool.
- `GroundingGrader` — [T2] three-way entailment adapter; `verdict = supported = (label == E)`; `assess()` returns (atom, `EntailmentResult`).
- `RelevanceGrader` — [T2] `raw ≥ relevance_tau → relevant`; logs raw + verdict.

**Calculations:**
- numeric match: `|na - nb| ≤ tolerance · max(|na|, |nb|)` (tolerance 0 = exact).
- grounding boolean: `label == E` (E/N/C from the provider over `entail_tau`/`contra_tau`).
- relevance boolean: `raw_score ≥ relevance_tau`.
- atomicity: no `;`/`and`/`but`/`whereas` clause join.

**Determinism:** T1 tools replay bit-for-bit (pure). T2 adapters replay from the
logged verdict; the raw score + tau are recorded as evidence. T3 adapters are
judge calls (non-replayable, model+version stamped).

**How to extend:** add a tier adapter as a small class taking its provider +
config threshold + logger; keep thresholding here, not in the provider.
