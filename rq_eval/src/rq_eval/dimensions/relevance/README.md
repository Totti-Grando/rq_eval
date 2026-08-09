# dimensions/relevance

**Design ref:** §3 relevance — **direct core + scaffolded support-tree** (two
layers, de-risked like accuracy).

**Purpose:** score how well the answer fits the *question*.
- **Layer 1 (the built score, default):** per-claim direct on-topic + on-ask
  (fixed NLI + lexical, DIVER-QA) → capped mean with abstention.
- **Layer 2 (`relevance.tree_enabled`, default off):** a support-tree that rescues
  indirectly-relevant claims — reading the edges of the **one shared `ClaimGraph`**
  (§0.3), building none of its own; when off, relevance = the direct core.
Relevance owns *structure*; edge *soundness* + stranded *contradictions* route to
the Reasoning `ConsistencyProvider`. The per-claim `responsive` atom is exported
for reuse either way.

**Classes:**
- `ClaimResponsiveness` — [T1+T2] per-claim `on_topic ∧ on_ask` → the exported `responsive` atom + `ClaimSignals` (on_topic/on_ask); this is the **direct core** (the default score).
- `Edge` — the lightweight support-edge view; **edges come from the shared `ClaimGraph`** (§0.3), relevance builds none.
- `AnchorSelector` — [T2 seed + code] on-ask seeds expanded by graph centrality (`anchor_centrality_min`); anchor recall wrapped in a conformal band (`anchor_alpha`).
- `SupportTree` — [code] fixpoint reachability from anchors; depth = relevance grade, bounded by `max_hops`, weight = `depth_decay ** depth`.
- `OrphanResolver` — [T1 on-topic + T2 orphan→anchor NLI] off-topic / stranded-veracity / background split; routes contradictions.
- `MethodAReverseQuestions` — [T3-gen + T2 cosine] RAGAS answer-relevancy (diagnostic).
- `MethodBGuardrail` — [T2] raw query↔response relevance score (default gate).
- `RelevanceDimension` — orchestrates edges → anchors → tree → orphans → score.

**Calculations:**
- on-topic (per claim): `relevance(question, claim) ≥ relevance_tau` `[T2]`.
- on-ask = `on_ask_nli ∨ on_ask_lex` (DIVER-QA): `on_ask_nli = entails(claim, ask) == E` `[T2]` (`ask = T1Tools.ask_hypothesis(question)`); `on_ask_lex = key_term_overlap(question, claim) ≥ lexical_min_overlap` `[T1]`. `responsive = on_topic ∧ on_ask`.
- **Layer 1 score** = `relevance_capped_mean`: `mean(responsive)`, capped at `off_ask_cap` when the answer-level on-ask is False; `= 1.0` on abstention. (Default.)
- edges (shared graph): `A→B ⟺ entails(A,B).raw_score ≥ edge_tau ∧ label ≠ C` (built in `pipeline/edge_detection.py`).
- anchors = on-ask seeds ∪ {claims with in-degree ≥ `anchor_centrality_min`}; recall carries a conformal band `[1−anchor_alpha, …]`.
- per-claim relevance grade (code-computed `claim_relevance` weight): anchor/background/stranded = 1.0; in-tree depth d = `depth_decay ** d`; off-topic orphan = 0.0.
- **Layer 2 score** = `relevance_tree_capped_mean`: `mean(claim_relevance.weight)`, off-ask capped, abstain-aware.
- Wilson 95% CI over (relevant count, #claims).

**Determinism:** edges/anchors/tree/orphans are fixed NLI + code and replay
bit-for-bit; the depth-decay grade is a **code-computed** atom weight. The **only**
judge here is **abstention** (proper decline to an unanswerable question); the
anchor-tree scoring path calls no judge. Stranded contradictions are routed to the
stub `ConsistencyProvider` (swaps cleanly when Reasoning lands — no relevance change).

**How to extend:** enable Layer 2 with `relevance.tree_enabled` once the
edge-recall harness (§0.3/G4) clears the bar; tune `anchor_alpha`,
`anchor_centrality_min`, `max_hops`, `depth_decay`; edge detection params live
under `graph.*` (the shared graph). Relevance reads the graph — it adds no edges.
