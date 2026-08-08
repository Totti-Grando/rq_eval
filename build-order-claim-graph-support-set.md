# Claude Code work order — claim-graph & support-set series (G1–G9)

**Goal:** bring the **existing `rq_eval`** codebase up to the current `@response-quality-design.md` and `@evidence-truthfulness-design.md`, which have moved past the earlier U-series (`build-order-design-sync-update.md`) in four areas: the **support-set** model (Evidence §1/§3/§4), the **single shared claim graph** (RQ §0.3), **accuracy as two-layer DAG resolution** (RQ §1), and the **groundedness scope + attribution axiom-subset** seams. This order **supersedes** the stale parts of the U-series and is a **refactor of a working system**, not a new build.

**The core discipline — protect what's defensible.** Everything here is layered so the **robust core never depends on the ambitious part**:
- **Accuracy Layer 1** (axiom-truth: `grounded ∧ source-adequate ∧ attributed`, every claim independent) is the **protected floor** — no edge detection, no graph.
- **Accuracy Layer 2** (DAG derivation-rescue) and **Relevance Layer 2** (support-tree reachability) are **additive, behind config flags, gated on measured edge-recall**. If edge detection underperforms, both degrade gracefully to their Layer-1 cores rather than mis-scoring.
- **Responsiveness is NOT in the axiom** — accuracy is truth-only; relevance owns responsiveness. Do not re-couple them.

**Unchanged constraints:** no AWS on the dev machine — all new behavior behind provider interfaces with **mocks**, green offline in `mode: mock`; every knob in the single **`config.yaml`**; style addendum (`@build-order-addendum-style-docs.md`) each phase (OOP, one class/file, per-folder README with the calculations, `mypy --strict`, ARCHITECTURE.md). Assumes the base RQ + Evidence builds and the judge-split (`ScoringJudge`/`ExplanationJudge`) already landed; if not, do those first (prior orders).

## How to run this

- **Read first:** the two design docs — esp. RQ §0.3 (claim graph, one-graph-two-projections, three claim types, two-verdict resolution), RQ §1 (two-layer accuracy), RQ §3 (relevance core + scaffolded tree), Evidence §1 (support set `S` + scope statement), §3/§4 (set-ops over `S`) — and the current `pipeline/`, `dimensions/`, `providers/` code.
- **Plan mode.** Propose a phased plan for **G0–G9**, wait for approval, one phase at a time, tests green after each.
- **Non-negotiable:** AI never emits numbers; scoring is code/NLI; every score replays from atoms; the `ExplanationJudge` touches no formula. New generative steps (none core here — extraction/triplets already deterministic) stay pinned. **One claim graph, built once; dimensions read projections — never a second graph.**
- **This order refactors an already-built system** (the U-series `build-order-design-sync-update.md` has run). Several phases *change* or *replace* existing code — most importantly, relevance already built its **own** edges/tree in U2/U3, and this order **repoints it at the shared graph and removes those edges** (G6). So G0 reconciles against the real code before any refactor, and phases state what they supersede.

---

## G0. Reconcile against the current code (no behavior change)

The U-series is built; this order edits it. Before refactoring, map the actual shapes so later phases change the right thing rather than an assumed one.
- Read and report the current structure of: `dimensions/groundedness/` (is grounding one merged-context check, or already per-span/per-chunk?), `dimensions/source_quality/` + `dimensions/source_attribution/` (do they call `entails` independently today, or already share a verdict?), `dimensions/relevance/` (**where does it build its tree edges** — U2/U3 built them locally; note the exact module/functions, since G6 removes them), `dimensions/accuracy/` (confirm it's the four-term `conjunction_weighted_mean` incl. `responsive` + weighting that G2 rewrites), and `pipeline/` extraction (confirm claims are already ClausIE/PredPatt-extracted per U1).
- Produce a short **reconcile note** in the PR/plan: for each of G1/G2/G6, one line — "current shape is X; this phase changes it to Y; code removed = Z." No code changes in G0.
**Accept:** a written map of the current groundedness/source-quality/attribution/relevance/accuracy/extraction shapes, and an explicit note of **what G6 will delete** (relevance's local edge construction) and **what G2 will change** (accuracy four-term → three-term truth). If any current shape contradicts a later phase's assumption, flag it now.

---

## G1. Evidence support-set refactor (Evidence §1/§3/§4)

The foundation — do first; everything downstream reads `S`. *(G0 has reported groundedness's current shape; refactor against that, not against an assumed merged-context version.)*
- `dimensions/groundedness/`: change to a **per-chunk pass** (if G0 found it already per-span, adapt the wording — the target is one `entails` per kept chunk producing a logged support set, however close the current code already is). `EmbeddingProvider` pre-filters top-`groundedness_k` chunks; `GroundingProvider.entails(chunk, triplet)` runs **once per kept chunk**; code builds the **support set** `S = {chunk : E}` (grouped by source document). `groundedness = |triplets with S≠∅| / |total|`. Log `S` per triplet (chunk-ids + labels).
- Add the **scope statement** to the groundedness README/docstring: per-claim direct source-presence, the axiom-*builder*, **not** the headline factuality number (that's accuracy's DAG).
- `dimensions/source_quality/`: **corroboration** = `|distinct docs in S| ≥ corroboration_min`; **supports** = `S≠∅` (imported, not re-run); reachability = `urllib` HEAD; date/author from retrieval metadata (`live_metadata_fetch` flag, default off); reputable-domain = `reliability_list.yaml`; disinterest = COI rule + sampled residual. No live full-text extraction on the core path.
- `dimensions/source_attribution/`: **set operation over `S`**, not a second NLI pass. Resolve cited set `C` (explicit regex `[T1]`; implicit = scope-propose `[T1]` + confirm-in-`S` `[T2]`, tagged `explicit`/`implicit-confirmed`). `attributed ⟺ C ∩ S ≠ ∅`. Emit diagnostics `C\S` (mis-citation), `S\C` (uncited-supported). Attribution applies to the **axiom subset** only; non-source-referencing claims are N/A → route out.
**Accept:** one per-chunk pass builds `S`; groundedness/corroboration/attribution all derive from it with **no extra NLI calls** (grep: attribution/source-quality make no new `entails` calls beyond implicit-cite confirm); `attributed ⊆ grounded` holds on a fixture; scope statement present; all prior tests green.


## G2. Accuracy Layer 1 — axiom-truth floor (RQ §1)

The protected core. Reformulate accuracy to per-node, but **axioms only** for now.
- `dimensions/accuracy/`: `accuracy = successful / total`, per node, equal weight. A claim succeeds iff **axiom-passing** = `grounded ∧ source-adequate ∧ attributed` (three, truth-only — **responsive removed**). Numeric claims → `T1Tools` exact-match, not NLI. Bare claim (fails axiom) → `ScoringJudge` unsourced residual, corpus-grounded. `formula_id = dag_resolution` but with **Layer 2 disabled** — every claim treated as an independent axiom.
- **Remove** the `responsive` import from accuracy entirely (it stays in relevance). Remove `conjunction_weighted_mean`/weighting from accuracy's path (weighting is a noted-not-built future item).
- Wilson CI over (successful, total); report axiom-to-derived ratio (trivially all-axiom at this layer).
**Accept:** accuracy computes with **no graph, no edge detection** — pure per-claim axiom conjunction; `responsive` appears nowhere in accuracy code (grep); `conjunction_weighted_mean`/weighting removed from accuracy's path; the determinism-ledger still passes; **before/after fixture**: a true, well-sourced, correctly-cited but *off-topic* claim was **excluded/penalized** under the old four-term conjunction and now **counts as accurate** (and separately scores low on relevance) — proving the responsive decoupling.

## G3. ClaimGraph service — structure + typing (RQ §0.3, no scoring yet)

Build the shared graph as infrastructure. No dimension scores from it yet.
- `pipeline/claim_graph.py`: nodes = claims; **claim typing** — **indexical** via `T1Tools` deixis/comparative/evaluative tagging `[T1]`; **inference-dependent** as a byproduct of G1 (well-formed claim with empty `S` but sibling-entailed); **independent** = default. `networkx` `DiGraph` scaffold; typed edges `{supports, derives, binds, contradicts}`.
- **Indexical binding**: `T1Tools` NER finds the sibling supplying each free slot; fill + verify (bound claim entails original ∧ contains filler); uncertain → flag `context-incomplete`, route out of grounding.
**Accept:** graph builds over a fixture answer with all three claim types correctly tagged; an indexical claim ("it is dark" with sibling location/time) binds; an unbindable one is flagged `context-incomplete`; no dimension score changed yet (graph is inert).

## G4. Edge detection + measured-recall harness (RQ §0.3)

The soft-underbelly — build it **with its validation harness in the same phase**, because the harness decides whether G5/relevance-Layer-2 are trustable.
- `pipeline/edge_detection.py`: **recursive backward premise-BFS from each conclusion** ([LLM argument-mining](https://arxiv.org/pdf/2605.13793) pattern) — for one target at a time, find supporting prior claims, expand each as a new target, restrict candidates to **unvisited earlier nodes** (acyclicity by construction). Per candidate: discourse-marker propose `[T1]` → topical-narrow (coref + embedding cluster) → `GroundingProvider.entails(⋀parents, claim)` confirm `[T2]`, **minimal-complete premise set** (greedy reduction). **Numeric convergence**: `T1Tools` number-provenance — a claim whose figure is a function of parent figures identifies parents precisely `[T1]`. `networkx` cycle-cut → source-less cycles fail.
- `validation/edge_recall.py`: against a human-linked fixture set, measure edge-detection **recall/precision**; report as the system's honest edge error bar. This gates whether Layer 2 is enabled in production.
**Accept:** edges detected on fixtures (single-parent, convergent-multi-parent, arithmetic, divergent); cycles cut; **edge-recall harness runs and reports a number**; a planted convergent edge (C1∧C2→C3) is found via the arithmetic signature.

## G5. Accuracy Layer 2 — DAG derivation-rescue (RQ §1, §0.3)

Additive, flagged, gated on G4.
- Extend accuracy: claims that failed Layer 1 as **bare** get a second chance — resolve their sub-DAG (roots-first topological eval, two verdicts: **local validity** = `⋀parents ⊨ claim`; **propagated truth** = valid ∧ parents-true). A dependent succeeds iff it resolves to passing axioms through locally-valid steps. **Per-node counting** (no double-count: shared axioms counted once, convergence = one node, diamonds need no special case). Valid-step-on-false-premise = valid-but-false (fails truth, localizes to the false parent).
- Behind `config.accuracy.dag_rescue_enabled` (default off until edge-recall clears the bar); when off, accuracy = Layer 1 exactly.
**Accept:** with the flag on, a validly-derived claim ("profitable" from grounded revenue/costs) that scored bare in Layer 1 now **counts as successful**; with the flag off, accuracy is bit-identical to G2; branching/convergence/diamond fixtures resolve correctly; a mid-way branch death fails only its branch.

## G6. Relevance Layer 2 — shared-graph reachability (RQ §3) · **supersedes U2/U3 edges**

**This phase replaces, not extends, U3.** U2/U3 built relevance's edges and tree *locally*; the design now says there is **one** shared graph. So this phase **removes relevance's own edge construction** (the modules G0 identified) and repoints relevance at the shared `ClaimGraph`. Relevance keeps its anchors and its orphan-resolution *logic*, but the edges come from the shared graph, not from relevance.
- `dimensions/relevance/`: keep the **direct core** (on-topic + on-ask NLI+lexical) as the built score — untouched. **Delete** U3's local edge-building; the scaffolded tree now **seeds anchors** (question-facing roots, centrality-confirmed, conformal-bounded recall) into the **shared `ClaimGraph`** and runs **reachability** over *its* support edges (bounded depth, depth-decay). **Orphan resolution** (three-way, reused checks): off-topic (fails on-topic vs question → penalize), stranded/veracity-bearing (on-topic ∧ entails/contradicts an anchor → keep + route to consistency/completeness via the `ConsistencyProvider` stub — **confirm the stub exists from U6; if not, build it here**), independent-background (on-topic, no relation → keep).
- Behind `config.relevance.tree_enabled` (default off, recall-gated); when off, relevance = direct core.
- **Assert relevance creates no edges** — it reads `ClaimGraph` and adds none of its own.
**Accept:** U3's local edge construction is **removed** (grep: relevance builds no edges); with the flag off, relevance = direct core exactly; with it on, the GDP-premise fixture attaches via the **shared** graph and the flood-zone orphan is classified **stranded** (kept + routed), not off-topic; a test proves relevance adds **zero** edges to the graph (reads only); the `ConsistencyProvider` stub is present (from U6 or built here).

## G7. Visualization (RQ §0.3)

- `pipeline/graph_viz.py`: `networkx` force-directed render over the resolved graph — axioms green roots, derived green/red by chain survival (broken step highlighted), contradiction edges red, orphans floating, mid-way branch death visible. Optional artifact alongside DimensionResults; pure view over logged nodes/edges/verdicts, no new computation.
**Accept:** renders a fixture answer's graph as a diagnostic file; reads only logged data (no model calls); a broken-chain fixture visibly shows the red step.

## G8. Groundedness/accuracy/attribution seam consistency + docs

- Ensure the three share `S` and the graph cleanly: groundedness builds `S` + axioms; attribution reads the axiom subset; accuracy reads derivation. No dimension recomputes another's edges.
- Per-folder READMEs updated with the new calculations (support-set ops, two-layer accuracy, one-graph-two-projections, orphan resolution). ARCHITECTURE.md: the single `ClaimGraph` service + its projections diagram.
**Accept:** READMEs/ARCHITECTURE current; the one-graph-two-projections structure documented; a reader can trace `S` → axioms → accuracy/attribution/relevance.

## G9. Config, determinism ledger, fixtures, offline sweep

- New config: `groundedness_k`; `corroboration_min`; `live_metadata_fetch` (off); implicit-cite scope rules; `accuracy.dag_rescue_enabled` (off); `relevance.tree_enabled` (off); `edge_tau`, `max_hops`, `depth_decay`; `numeric_tolerance`.
- **Determinism-ledger test** (extend): score-affecting `[T3]` set == exactly {accuracy bare-residual, task_success adequacy, relevance abstention, admissibility-decidability residual, source-quality disinterest residual}; assert **one** claim graph exists and relevance/accuracy read projections (no second graph); assert accuracy carries no `responsive` and no `edge` dependency when Layer 2 is off.
- Fixtures: G1 support-set + mis-citation; G2 true-but-off-topic scores accurate; G3 three claim types + indexical bind; G4 convergent/divergent/arithmetic edges + recall harness; G5 derivation-rescue on/off parity; G6 stranded orphan + zero-edges-added; G7 broken-chain render.
- Full offline run green in `mode: mock`; ReplayVerifier passes over a whole run including the graph.
**Accept:** full offline suite green; both Layer-2 flags off by default and the system runs on the protected cores; ledger locks the T3 surface and the single-graph invariant; ReplayVerifier passes.

---

## Definition of done

- **Protected cores stand alone:** accuracy Layer 1 (axiom-truth, no graph) and relevance direct core (no tree) fully score the system with both Layer-2 flags off.
- **Ambitious layers are additive + gated:** accuracy DAG-rescue and relevance tree are behind flags, gated on the measured edge-recall harness, and degrade gracefully.
- **One shared claim graph**, built once; relevance-reachability and accuracy-derivation are projections; a test proves no second graph and relevance adds no edges.
- **Support-set model:** one per-chunk pass builds `S`; groundedness/corroboration/attribution are set-ops over it; `attributed ⊆ grounded`.
- **Responsiveness is relevance-only**; the axiom is truth-only (three terms).
- All new knobs in `config.yaml`; offline-green in mock; mock→live swaps with no formula change; style addendum honored; determinism-ledger + single-graph invariants pass; ReplayVerifier passes.
