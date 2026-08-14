# Claim graph — mathematical design sheet

Standalone consolidation of the graph model. Supersedes the sketch in `response-quality-design.md` §0.3; that section should be replaced with a pointer to this sheet. Companion research: `node-completeness-frame-semantics.md` (completeness), `calibration-datasets-inventory.md` (error measurement). Scope: **one shared claim graph**, built once, read as projections by accuracy (derivation), relevance (reachability), reasoning (soundness). This sheet is the graph's structure, node math, and error model. Two-layer discipline throughout: the graph is the **additive, flagged, recall-gated** layer; the per-dimension robust cores never depend on it.

## 0. What the graph is and why

Not every claim is independently checkable. Some are **derived** ("margins improved" — no source states it, but it follows from grounded figures); some are **indexical** ("it is dark" — not a complete proposition until siblings supply where/when). Scoring these in isolation is meaningless. So over the extracted claims we build a directed acyclic **claim graph** — nodes are claims, edges are typed dependencies — and resolve it. The graph is **shared infrastructure, not a scorer**: atomic checks (groundedness) ignore it and stay per-claim; structural metrics read projections. Research basis: entailment trees/DAGs with multi-premise hyperedges ([Dalvi 2021](https://aclanthology.org/2021.emnlp-main.585.pdf), [Deliberate Reasoning 2024](https://arxiv.org/pdf/2410.03136), [Chains-to-DAGs 2026](https://arxiv.org/pdf/2601.17593)).

## 1. The committed-edge principle (the founding simplification)

A relationship is **topology, not degree.** A "90%-related" edge is incoherent — it blends two different graphs (one where the claim is dependent, one where it is independent), and no single number represents that. So **edges are committed**: thresholded to real/not-real at a calibrated cutoff, never carried as a fuzzy fraction. This is what lets the conditional math stay clean — you condition on a *committed* structure, not a probability-weighted smear of structures. The same discipline applies to completeness (a gate/ratio, not a vague score) and to relationship validity (a gate, not a multiplier that erodes truth). Bold but fortified: committing the structure is the price of sound math over it.

## 2. Three claim types (typed at extraction, mostly `[T1]`)

- **Independent** — a complete proposition, groundable alone (default).
- **Inference-dependent** — a complete proposition no single source attests, entailed by siblings. *Detected as a byproduct of grounding*: well-formed claim with empty support set `S` but sibling-entailed. No separate detector.
- **Indexical-dependent** — an *incomplete* proposition with free spatiotemporal/deictic slots. *Detected by `T1Tools`* deixis/comparative/evaluative tagging (spaCy POS+dep). **Bound before scoring**: NER finds the sibling filling each free slot; fill, then verify the completed claim still entails the original and contains the filler. Unbindable → flag `context-incomplete`, route out of grounding (reported, not guessed).

## 3. Construction — edges top-down, truth bottom-up

Two operations, opposite directions:

- **Edge detection is top-down** (backward from conclusions). For each non-axiom claim, search *backward* for what supports it: the [recursive backward premise-BFS](https://arxiv.org/pdf/2605.13793) — find supporting priors for one target, expand each as a new target, **restrict candidates to earlier unvisited nodes → acyclic by construction**. Per candidate edge: **discourse markers propose** `[T1]` → **topical/entity restriction narrows** (coref + embedding cluster) `[T1/T2]` → **entailment confirms** `⋀parents ⊨ claim` `[T2]`, to the **minimal-complete premise set** ([ReasoningFlow 2026](https://arxiv.org/pdf/2606.05402): complete = suffices to entail; minimal = removing any parent breaks it), found by greedy reduction not subset enumeration. **Numeric convergence** gets a near-deterministic aid `[T1]`: number-provenance — if a claim's figure is a function of parent figures (profit = revenue − costs), the arithmetic signature identifies the parents exactly. Cycles that never reach an axiom → cut (`networkx`), those nodes fail (no infinite regress).
- **Truth resolution is bottom-up** (from axioms). Roots-first topological evaluation to a fixpoint.

All structural types fall out of this one pass: **series** (chain), **convergence** (many→one, hyperedge), **divergence** (one→many, out-degree>1), **diamonds** (fan-out then fan-in), **parallel** (= convergence + independent components). Axioms and non-axioms can be **related without being sequential** — topical relatedness (a separate projection, for relevance clustering) is not a dependency edge; edges form only where derivation happens.

## 4. Sufficiency structure — AND / OR / k-of-n (per convergent node)

A derived node's parents carry a **logical condition**, not just a set. Determined by **ablation** (`[T2]`, bounded to the used parent set): remove a parent, re-check entailment.
- `⋀parents ⊨ C` but no single parent alone ⊨ C → **AND** (all load-bearing).
- Some parent alone ⊨ C, and multiple such → **OR** (alternative sufficient sets).
- Threshold count → **k-of-n**, enumerable for small sets; beyond a bounded budget → flag `structure-uncertain`, conservative-AND fallback.
Calibrated against **LogiQA 2.0** (human conjunctive/disjunctive labels). Scope: we verify **the answer's actual derivation is sufficient**, not enumerate every possible sufficient set (that is exponential and unnecessary for evaluation).

## 5. The node-value formula (the core result)

For each derived node, given parents resolved to committed states:

> **node_value = completeness_ratio × min(truth-likelihood of load-bearing parents)**

**Load-bearing parents** = those whose state, if flipped, would change whether the node's logical condition holds (counterfactual test). Satisfied-OR siblings are **excluded** from the min — so a false-but-irrelevant sibling does not poison an OR node. AND-node → min over all parents; OR-node → min over the satisfying subset only.

**Why min, not product of relationship-probabilities.** A *valid* step does not degrade truth — if premises are true and the step is valid, the conclusion is fully true, not 0.8-true. Treating relationship-confidence as a multiplicative fraction conflates *uncertainty about validity* (belongs in the calibrated edge threshold) with *degradation of truth* (zero for a valid step), producing pathological rot. So: **relationships GATE** (committed/thresholded, calibrated), **only parent truths propagate** (as the min). The min is weakest-link and assumption-free (no independence assumption, unlike a product).

**Why a false parent fails the node automatically.** Multiplication can't increase; a load-bearing false parent contributes ~0 to the min → node collapses. Both failure modes are absorbed: low completeness_ratio → fails via the ratio; false determining parent → fails via the min. No separate true/false/underdetermined machinery needed — one number, weakest determining link governs.

**Two verdicts, kept separate** for honest diagnosis: **local validity** (does the step follow from parents — verification-independent, checked in isolation) vs **propagated truth** (valid ∧ parents true). A valid step on a false premise = valid-but-false, localized to the false parent, not blamed on the reasoning.

**Log BOTH factors** (completeness_ratio and the min, with the limiting parent) so a failing node's *reason* is recoverable — the elegant single number for scoring, the two factors for the audit/visual.

**Threshold once, conformally, at the node** — the committed boolean for the level above, with the value retained as reported confidence. Not a literal probability; a defensible measured surrogate that behaves like P(claim | support) and is calibrated at the decision.

## 6. completeness_ratio — frame-semantic role coverage (the measurable completeness)

The problem: a claim can be logically completed a million ways, so we can't enumerate the ideal premise set. The solution: a **fixed, curated, minimal schema** from the claim's own predicate.

> **completeness_ratio = fully-filled schema components / total schema components** (binary per component, no partial credit)

**The schema comes from the CLAIM, not the parents** (deliberate — a parent-derived schema is circular and can never detect a gap). The claim's predicate **evokes a FrameNet frame** whose **core roles** are a fixed 2–4 role set, looked up from FrameNet's 1200+ frames — the minimal denominator. SRL (`[T2]`, F1 85–91) identifies the frame and fills roles. See `node-completeness-frame-semantics.md`.

**A role is filled two ways** (your cost+revenue→profit case forces this):
- **Direct fill** — a parent's matching role supplies it (entity, time — the *contextual* roles). Determined by SRL-parsing the parents too and aligning **role-to-role by filler type** (entity↔entity, time↔time) + per-slot entailment-confirm (so topical-nearness alone doesn't falsely fill a slot).
- **Derived fill** — the role's filler is *computed/inferred* across parents (the *asserted* role — profitability from revenue−costs). No parent contains "profitable"; it's derived by **arithmetic** (number-provenance, `[T1]`, near-deterministic) or **entailment** (`⋀parents ⊨ role-content`, `[T2]`). This is the whole point of a derived claim — its assertion is computed, not copied — and frame-matching alone would miss it.

A component is covered if **direct OR derived**. So: **frame roles handle context-completeness; arithmetic/entailment handles assertion-completeness.** Tag each component's fill by source (direct / arithmetic / entailment) — since "no partial" makes each binary call worth a full 1/n, the fill-confidence composition shows whether the ratio rests on solid (direct/arithmetic ≈0 error) or soft (entailment, NLI-ceiling) fills.

**Unfilled core role** → a *named, typed* gap, classified by FrameNet's null-instantiation taxonomy: **DNI** (definite — recoverable from discourse → completer-search over the answer's other claims: does adding some claim X close the entailment) vs **INI** (indefinite — genuine gap → unstated assumption, route to assumption_quality). The completer-search is feasible *because node-completeness is confined* to the answer's own small claim set — unlike answer-level completeness against the open world.

## 7. Accuracy as DAG resolution (the projection)

**accuracy = successful claims / total claims**, equal weight (weighting removed; vital-weighting a noted future item), counted **per node** (not per path — so branching/convergence/diamonds need no special case and shared axioms are never double-counted). A claim is successful if:
- **Axiom** (directly-verifiable-*true*): `grounded ∧ source-adequate ∧ attributed` (three, truth-only — **responsiveness is NOT here**; that's relevance's axis). Axiom = length-1 chain, counted once.
- **Dependent**: its sub-DAG resolves to passing axioms through locally-valid steps, i.e. `node_value` (§5) clears the conformal threshold. Counts positively even if never grounded/attributed on its own — forming a valid claim from evidence is itself accurate.
- **Bare** (no source, no valid parents) → failed length-1 node (penalized). ScoringJudge unsourced-residual fires only here, corpus-grounded `[T3]`.

**Independence, defended honestly:** we count **verification-independence** (each node's local check is self-contained — "does B follow from A" is independent of "does C follow from A"). We do **not** claim statistical independence — branches sharing an axiom are correlated, and that correlation is honored by truth-propagation (a shared false axiom fails everything downstream, in proportion to how much rested on it). The **axiom-to-derived ratio is reported** so evidential breadth is visible (ten conclusions on one axiom ≠ ten independent axioms).

## 8. Error tracking — two types, carried and reported

**Type 1 — decision uncertainty (calibratable, propagates).** Each atomic decision (role fill, edge, axiom) inherits the **measured error rate of its decision-TYPE** — a class frequency, not an instance probability ("decisions of this kind are right ≥X%", not "this call is X% right"). AtomRecord gains `{source: direct|arithmetic|entailment|frame-srl, error_band (lookup by type), conformal_covered}`. Propagates **weakest-link / worst-case** alongside the value (mirrors the min): node carries `{value, error_bound, limiting_atom}`.

**Type 2 — structural uncertainty (recall gaps, flagged NOT fused).** Missed edge, wrong frame, missed implicit role — you can't put an honest ± on "something we didn't detect." So these are **flags** (`missing_edge_suspected`, `frame_uncertain`, `implicit_role_INI`, `unbindable_indexical`), counted at answer level as a **coverage/recall statistic**, never folded into the value.

**The rule:** value carries a Type-1 band; Type-2 is reported as coverage. Never one bundled "0.82 ± 0.09" that hides undetected-structure risk. Answer-level output = **value + decision-error profile (by source type) + structural-coverage profile (flag counts) + limiting factors** (which atoms dominated error on the important nodes → where a reviewer looks first).

## 9. Where the error numbers come from — measurement, not derivation

Error is **measured against human-labeled ground truth per decision-type**, conformal-wrapped for a coverage guarantee. The key finding (`calibration-datasets-inventory.md`): **most decision-types calibrate against existing benchmarks** — no bespoke labeling:
- **Groundedness** → LLM-AggreFact (cleaned), TRUE, RAGTruth; **financial** → FinReflectKG-HalluBench.
- **Edges / step validity** → EntailmentBank (1,840 trees, 5,881 steps), STREET, PRM800k.
- **Sufficiency / completeness** → RDTE (relevance/acceptability/**sufficiency** labels).
- **AND/OR** → LogiQA 2.0.
- **Attribution** → Fact-Level Attribution, AttributionBench.
- **Frame/role** → FrameNet.
- **Arithmetic/direct fills** → ≈0 error by construction, tiny confirmation set only.
Bespoke labeling shrinks to a small **financial completeness/sufficiency** set. Benchmarks give a strong prior; the **review loop** (flagged/high-error nodes → human queue → labels) refines to the real distribution over time and *bootstraps* the calibration data. **Where no calibration set exists yet, output a flag ("uncalibrated"), never a manufactured number.**

## 10. Visualization (first-class diagnostic)

The resolved graph is logged (nodes + typed edges + two verdicts + both value-factors + flags), so a `networkx` force-directed render is a *view*, not new computation: axioms green roots; derived nodes green/red by chain survival with the **broken step highlighted**; the **limiting factor** per node visible; contradiction edges red; orphans floating; a mid-way branch death visibly severing one branch while its sibling survives; the completeness gap named by role. For certification this is the payoff — a human sees *where* reasoning failed and *what rested on what*, which a scalar can't convey.

## 11. The single reasoning-structure calibration harness (what gates the whole layer)

Edge-recall, frame-disambiguation, node-completeness, AND/OR typing, and fill-confidence are **the same underlying uncertainty** — "did we determine the reasoning structure correctly," all bottoming out in NLI's ceiling. So **one harness** measures them together against the human-linked datasets (§9), producing the reasoning-structure accuracy number. **This number gates whether the graph layer's outputs are trusted / enabled** — the two-layer flags (accuracy DAG-rescue, relevance tree) stay off until it clears a bar. This harness is the deliverable that turns the elegant math into something certifiable: it is the honest answer to "how do you know how wrong this might be."

## Output object

`ClaimGraph { nodes: Claim[+type, two-verdicts, value, completeness_ratio, min_factor, limiting_atom, flags], edges: [type ∈ {supports, derives, binds, contradicts}, parent_set, logical_condition ∈ {AND,OR,k-of-n}, confirmed_by], axioms[], failed[], axiom_derived_ratio, structural_flags[], decision_error_profile }` — logged, replayable (set-ops + topological eval over stamped labels, no model call), the substrate accuracy/relevance/reasoning project from.

## Honest bounds (stated, not hidden)
- **Edge recall** is the weakest link — unmarked/world-knowledge/convergent-multi-premise edges get missed; a missed edge makes a valid derived claim look like a failed orphan. Measured and reported.
- **Frame disambiguation** errors → wrong role schema; bounded by SRL F1, measured.
- **INI vs DNI** (real gap vs recoverable) is the irreducible enthymeme residue — named taxonomy, entailment-recovery, but NLI-ceiling bounded. Flagged.
- **Cross-frame role-label alignment** (claim's `time` ↔ parent's `duration`) is where role-matching errors concentrate; in the harness.
- **Domain transfer** — most calibration sets are science/news/general-logic; financial error may differ (why FinReflectKG-HalluBench + a small financial set matter).
- **Node-value is a surrogate, not a literal probability** — defensible because calibrated at the threshold, but not claimed as a true posterior.
All of these are why the graph is the **flagged, recall-gated, additive** layer — never the robust core.
