# dimensions/relevance

**Design ref:** §3 relevance — **anchor-and-support tree + orphan resolution**.
Built **first** because accuracy (§1) imports the per-claim responsive atom
computed here.

**Purpose:** score how well the answer fits the *question*, modelled as an
argument-mining support tree over the whole answer (not a blunt per-claim
filter), and publish the per-claim `responsive` boolean to `ResponsivenessExport`
for accuracy. Relevance owns *structure*; edge *soundness* and stranded
*contradictions* route to the Reasoning `ConsistencyProvider`.

**Classes:**
- `ClaimResponsiveness` — [T1+T2] per-claim `on_topic ∧ on_ask` → the exported `responsive` atom + `ClaimSignals` (on_topic/on_ask) that seed the tree (**no judge**, DIVER-QA).
- `EdgeBuilder` — [T1 prior + T2 confirm] entailment-confirmed premise→conclusion support edges (`entails(A,B) ≥ edge_tau`); markers are a candidate prior only.
- `AnchorSelector` — [T2 seed + code] on-ask seeds expanded by graph centrality (`anchor_centrality_min`); anchor recall wrapped in a conformal band (`anchor_alpha`).
- `SupportTree` — [code] fixpoint reachability from anchors; depth = relevance grade, bounded by `max_hops`, weight = `depth_decay ** depth`.
- `OrphanResolver` — [T1 on-topic + T2 orphan→anchor NLI] off-topic / stranded-veracity / background split; routes contradictions.
- `MethodAReverseQuestions` — [T3-gen + T2 cosine] RAGAS answer-relevancy (diagnostic).
- `MethodBGuardrail` — [T2] raw query↔response relevance score (default gate).
- `RelevanceDimension` — orchestrates edges → anchors → tree → orphans → score.

**Calculations:**
- on-topic (per claim): `relevance(question, claim) ≥ relevance_tau` `[T2]`.
- on-ask = `on_ask_nli ∨ on_ask_lex` (DIVER-QA): `on_ask_nli = entails(claim, ask) == E` `[T2]` (`ask = T1Tools.ask_hypothesis(question)`); `on_ask_lex = key_term_overlap(question, claim) ≥ lexical_min_overlap` `[T1]`. `responsive = on_topic ∧ on_ask`.
- edge `A→B ⟺ entails(A,B).raw_score ≥ edge_tau ∧ label ≠ C`.
- anchors = on-ask seeds ∪ {claims with in-degree ≥ `anchor_centrality_min`}; recall carries a conformal band `[1−anchor_alpha, …]`.
- per-claim relevance grade (code-computed `claim_relevance` weight): anchor/background/stranded = 1.0; in-tree depth d = `depth_decay ** d`; off-topic orphan = 0.0.
- score = `relevance_tree_capped_mean`: `mean(claim_relevance.weight)`, capped at `off_ask_cap` when the answer-level on-ask atom is False; `= 1.0` on abstention.
- Wilson 95% CI over (relevant count, #claims).

**Determinism:** edges/anchors/tree/orphans are fixed NLI + code and replay
bit-for-bit; the depth-decay grade is a **code-computed** atom weight. The **only**
judge here is **abstention** (proper decline to an unanswerable question); the
anchor-tree scoring path calls no judge. Stranded contradictions are routed to the
stub `ConsistencyProvider` (swaps cleanly when Reasoning lands — no relevance change).

**How to extend:** tune `edge_tau`, `anchor_alpha`, `anchor_centrality_min`,
`max_hops`, `depth_decay` in config; the `responsive` atom is the single import
surface for accuracy — do not recompute it elsewhere.
