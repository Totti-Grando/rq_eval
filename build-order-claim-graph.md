# Claude Code work order — claim graph (CG1–CG10)

Builds the claim graph per `@claim-graph-design.md` (with `@node-completeness-frame-semantics.md` and `@calibration-datasets-inventory.md`). This is the **graph layer** — the additive, flagged, recall-gated part of the system. It assumes the support-set refactor and accuracy Layer-1 (axiom-truth floor) from the G-series (`build-order-claim-graph-support-set.md`, phases G0–G2) are **already built**: the graph rescues claims that Layer-1 scored as bare, and it never breaks Layer-1.

**Governing discipline (do not violate):**
- **The whole graph layer is behind flags** (`accuracy.dag_rescue_enabled`, `relevance.tree_enabled`), default **off**, gated on the reasoning-structure calibration harness (CG9) clearing a bar. With flags off, the system runs on the robust cores exactly.
- **One graph, built once.** Accuracy reads derivation, relevance reads reachability, reasoning reads soundness — projections, not copies. No dimension builds its own edges.
- **Committed edges** — thresholded, never fuzzy fractions. **Relationships gate; only parent truths propagate (min).** No relationship-probability multiplication.
- **Every layer ships with its error measured** against the datasets in `@calibration-datasets-inventory.md` before the next layer depends on it. Where no calibration set exists → emit a flag, never a manufactured number.
- Mocks green offline in `mode: mock`; all knobs in `config.yaml`; style addendum each phase (OOP, one class/file, per-folder README with the math, `mypy --strict`).

## How to run
- **Read first:** `@claim-graph-design.md` end to end, then the two research companions, then existing `pipeline/` and `dimensions/accuracy/` (Layer-1 must be present).
- **Plan mode:** propose CG1–CG10, wait for approval, one phase at a time, tests green after each. Each phase's "Accept" includes its **calibration check** where a dataset exists.

---

## CG1. Graph scaffold + claim typing (design §2)
`pipeline/claim_graph.py` — `networkx.DiGraph`; nodes = extracted claims. **Claim typing:** indexical via `T1Tools` deixis/comparative/evaluative tagging `[T1]`; inference-dependent as byproduct of empty support-set `S` but sibling-entailed; independent = default. Typed edge stubs `{supports, derives, binds, contradicts}`. No scoring, no resolution yet.
**Accept:** three types tagged correctly on a fixture; graph object builds and serializes; inert (no dimension score changes).

## CG2. Indexical binding (design §2)
Bind indexical claims before scoring: `T1Tools` NER finds the sibling filling each free slot; fill; verify completed claim entails original ∧ contains filler. Unbindable → flag `context-incomplete`, route out of grounding.
**Accept:** "it is dark" + sibling location/time binds; unbindable case flagged, not guessed; bound claim re-enters grounding correctly.

## CG3. Edge detection — backward premise-BFS (design §3)
`pipeline/edge_detection.py` — recursive **backward** premise-BFS from each conclusion; candidates restricted to **earlier unvisited nodes** (acyclic by construction). Per candidate: discourse-marker propose `[T1]` → topical/entity narrow (coref + `EmbeddingProvider`) → `GroundingProvider.entails(⋀parents, claim)` confirm `[T2]`, greedy-minimal premise set. **Numeric convergence:** `T1Tools` number-provenance identifies arithmetic parents exactly `[T1]`. Cycle-cut via `networkx` → unreachable-to-axiom nodes fail.
**Accept + CALIBRATION:** single-parent, convergent, divergent, arithmetic, diamond fixtures build correct edges; cycles cut. **Run edge detection over EntailmentBank + STREET; report edge-precision/recall vs the gold trees** — this is the first real reasoning-structure number.

## CG4. Sufficiency structure — AND/OR/k-of-n (design §4)
Ablation over the used parent set (`[T2]`): remove a parent, re-check entailment → label the convergent node `AND` / `OR` / `k-of-n` / `structure-uncertain` (conservative-AND fallback beyond bounded budget). Store `logical_condition` on the edge group.
**Accept + CALIBRATION:** AND node (both required), OR node (either suffices), k-of-2 fixtures typed correctly. **Calibrate AND/OR typing against LogiQA 2.0** (conjunctive/disjunctive labels); report accuracy.

## CG5. Frame-semantic completeness (design §6, `@node-completeness-frame-semantics.md`)
`pipeline/completeness.py` — SRL identifies the claim's frame; look up **core roles** from FrameNet (the fixed denominator). Fill each role: **direct** (SRL-parse parents too, align role-to-role by filler type + entailment-confirm) OR **derived** (arithmetic number-provenance `[T1]`, or `⋀parents ⊨ role-content` `[T2]`). `completeness_ratio = filled/total core roles` (binary per component). Unfilled role → typed gap, classify **DNI** (completer-search over the answer's claims) vs **INI** (assumption → route to assumption_quality). Tag each fill by source (direct/arithmetic/entailment).
**Accept + CALIBRATION:** cost+revenue→profit fixture: entity/time direct-filled, polarity **derived**-filled (arithmetic) → ratio = 1.0; drop costs → polarity unfilled → ratio < 1, gap named "polarity/value", classified INI. Frame-disambiguation calibrated against **FrameNet**; sufficiency against **RDTE**; report both. Predicate with no frame → PredPatt fallback denominator (degraded, flagged).

## CG6. Node-value resolution — the core formula (design §5)
`pipeline/resolution.py` — roots-first topological eval to fixpoint. Per derived node: identify **load-bearing parents** by counterfactual (flip parent → does condition still hold); `node_value = completeness_ratio × min(truth-likelihood of load-bearing parents)`. AND → min over all; OR → min over satisfying subset (false-but-irrelevant siblings excluded). Two verdicts stored (local validity, propagated truth); valid-on-false-premise = valid-but-false, localized. **Log both factors** (completeness_ratio, min, limiting parent). **Threshold once, conformally, at the node.**
**Accept:** OR node with [true, true, false] where one of a pair suffices → false sibling excluded from min, node not poisoned; 90%-complete-from-true-premises → value ≈ 0.9 (penalized, not killed, propagates forward); false load-bearing parent → node collapses via min; mid-way branch death fails only its branch. Both factors recoverable per node.

## CG7. Accuracy Layer-2 — DAG rescue (design §7)
Extend `dimensions/accuracy/`: claims that failed Layer-1 as **bare** get resolved via the graph — succeed if `node_value` clears the conformal threshold. **Per-node counting** (no double-count; shared axioms once; convergence = one node). Report **axiom-to-derived ratio**. Behind `config.accuracy.dag_rescue_enabled` (default off). Axiom definition unchanged: `grounded ∧ source-adequate ∧ attributed` (truth-only, no responsive).
**Accept:** flag ON — a validly-derived claim bare in Layer-1 now counts; flag OFF — accuracy bit-identical to Layer-1; branching/convergence/diamond resolve; axiom-derived ratio reported.

## CG8. Error tracking — two-type carry + report (design §8)
Extend `AtomRecord`: `{source, error_band (lookup by decision-type), conformal_covered}`. Node carries `{value, error_bound (weakest-link), limiting_atom, structural_flags}`. **Type 1** (decision) propagates worst-case alongside value; **Type 2** (structural: missing-edge/frame-uncertain/INI/unbindable) flagged, counted at answer level, **never fused into value**. Answer-level report: `value + decision-error profile (by source) + structural-coverage profile (flag counts) + limiting factors`.
**Accept:** a node dominated by an entailment-fill shows a higher error_bound than an all-arithmetic node; structural flags counted and surfaced separately; no bundled "value ± x" that hides Type-2; report renders.

## CG9. Reasoning-structure calibration harness (design §11 — the gate)
`validation/reasoning_structure.py` — one harness measuring edge-recall, frame-disambiguation, node-completeness/sufficiency, AND/OR typing, fill-confidence **together** against the datasets (`@calibration-datasets-inventory.md`): EntailmentBank/STREET (edges/steps), RDTE (sufficiency), LogiQA 2.0 (AND/OR), FrameNet (frames), FinReflectKG-HalluBench (financial grounding). Emits the **reasoning-structure accuracy** number + per-decision-type conformal error bands that CG8 reads. **This number gates the two-layer flags** — a config-checked bar below which `dag_rescue_enabled`/`tree_enabled` refuse to enable.
**Accept:** harness runs over the public datasets, emits per-type error bands and an aggregate; the gate mechanism works (flags refuse to enable below the bar); where a set is absent, emits "uncalibrated" flag not a number. Document each error band + its source dataset.

## CG10. Visualization + relevance projection + docs (design §10, §7)
- `pipeline/graph_viz.py` — `networkx` force-directed render (view only, no new computation): axioms green, derived green/red by survival, broken step + limiting factor highlighted, contradictions red, orphans floating, named completeness gaps.
- **Relevance reads the shared graph** (reachability projection) — assert it adds **zero** edges; anchors as roots, orphan resolution (off-topic / stranded / background). Behind `relevance.tree_enabled`.
- Replace `response-quality-design.md` §0.3 with a pointer to `claim-graph-design.md`. Per-folder READMEs with the node math. ARCHITECTURE.md: the one-graph-projections diagram.
**Accept:** broken-chain fixture renders with the red step; relevance adds zero edges (test-proven); §0.3 points to the sheet; determinism-ledger asserts one graph + relevance-reads-only + node math replays from stamped labels with no model call.

---

## Definition of done
- Graph builds once; accuracy/relevance/reasoning read projections; a test proves no second graph and relevance adds no edges.
- Node value = `completeness_ratio × min(load-bearing parents)`, committed edges (gate), min-propagation (no relationship-probability multiply), conformal node threshold, both factors logged.
- completeness_ratio = frame-role coverage (direct OR derived fill), gaps named + INI/DNI classified.
- Two-type error tracking: Type-1 propagates (weakest-link), Type-2 flagged as coverage, never fused.
- **The calibration harness runs against public datasets, emits real per-type error bands, and gates the layer flags.** Where uncalibrated → flag, not a number.
- Both layer flags default OFF; system runs on robust cores with them off; graph degrades gracefully.
- Offline-green in mock; replays with no model call; style addendum honored; determinism-ledger + single-graph invariants pass.
