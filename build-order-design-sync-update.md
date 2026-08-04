# Claude Code work order — design-sync update (extraction, relevance tree, completeness modes)

**Goal:** bring the **existing `rq_eval`** codebase up to the current `@response-quality-design.md` and `@evidence-truthfulness-design.md`. This supersedes the parts of the earlier judge-minimizing change order that the design has since moved past. It is a **refactor of a working system**: keep every existing test green, change in place, and where a step moves tier, prove the old path is gone. **Design docs win** on any conflict.

**What changed in the design since the last build order (implement all):**
1. **Claim decomposition is now deterministic** — dependency-parse (ClausIE/PredPatt-style over spaCy), *not* generative; optional pinned surface-realizer that is **droppable** if the verifier tolerates parse-form units.
2. **Relevance is rebuilt** — from a per-claim on-ask filter to an **anchor-and-support tree + orphan resolution** (argument-mining structure). New concepts: anchors (centrality + conformal recall), entailment-backed edges, bounded-depth traversal, three-way orphan classification.
3. **Completeness reference is now mode-based** — `generated` (primary, open-domain), `archetype` (fixed question-shape skeletons), `templated` (human checklist); assurance mode **stamped on the result**; human-recall-sample **miss-rate reported**.
4. **Forward-declared `ConsistencyProvider`** — relevance routes edge-soundness and stranded-contradiction orphans to a Reasoning-category interface that doesn't exist yet; build as a stub with defaults.
5. **Evidence triplets** — parse-first (PredPatt/OpenIE), generation only for the residual.

**Unchanged constraints:** no AWS on the dev machine; all new behavior behind existing provider interfaces with **mocks**, green offline in `mode: mock`; every new knob in the single **`config.yaml`**; style addendum (`@build-order-addendum-style-docs.md`) each phase (OOP, one class/file, per-folder README with the calculations, `mypy --strict`, ARCHITECTURE.md). This order assumes the base RQ + Evidence builds and the judge-split (`ScoringJudge`/`ExplanationJudge`) already landed; if the judge-split hasn't, do that phase first (see prior order R1).

## How to run this

- **Read first:** the two updated design docs (esp. RQ §0.2 extraction, §2 completeness modes, §3 relevance tree, §0.5 forward-declared interfaces), and the current `pipeline/`, `dimensions/relevance/`, `dimensions/completeness/` code.
- **Plan mode.** Phased plan for U1–U7, wait for approval, one phase at a time, tests green after each.
- **Non-negotiable:** AI never emits numbers; scoring is code/NLI; every score replays from atoms; the `ExplanationJudge` touches no formula. New generative steps (completeness reference, triplet residual) are **pinned + stability-measured**, not live judges.

---

## U1. Deterministic claim extraction (RQ §0.2)

Replace the generative claim-extraction step with a parse-based `ClaimExtractor`.
- `pipeline/claim_extractor.py`: spaCy dependency parse → **ClausIE/PredPatt-style** clause/predicate-argument decomposition → content-unit claims `[T1]`. Mock: the existing rule/clause splitter (already deterministic).
- verifiable-vs-opinion filter: `re` + POS hedge/opinion markers `[T1]`, residual to a fixed classifier `[T2]`.
- decontextualization: `coreferee` substitution `[T1/T2]` (already present).
- **optional surface-realizer** `[T2, pinned]` behind `config.extraction.realizer_enabled`; add the **realizer-impact test**: run the NLI verifier on parse-form vs realized claims over a fixture and report verdict-agreement — if agreement is high, default `realizer_enabled: false`.
- keep `pins.extractor_version` + stability metric; flag abstractive-implied claims rather than generating them.
**Accept:** claim decomposition calls no GeneratorProvider on the primary path (grep-proven); realizer is off by default unless the impact test shows it's needed; stability still logged; all downstream dimensions consume the new claims unchanged; tested.

## U2. Relevance — anchors + edges (RQ §3, part 1)

Build the graph substrate.
- `dimensions/relevance/anchors.py`: seed anchors via the on-ask NLI+lexical check (already built), then **confirm/expand by graph centrality** over the edge graph; wrap anchor-set recall in **conformal** (reuse Evidence §5 `conformal.py`) with `config.relevance.anchor_alpha`. `[T2 seed + code centrality]`
- `dimensions/relevance/edges.py`: candidate edges from discourse markers (`because`, `tied to`, …) `[T1]`; **confirm** each with `GroundingProvider.entails(A, B) ≥ edge_tau` `[T2]` — markers alone never make an edge.
**Accept:** an asserted-but-unentailed "because" link does not become an edge (fixture); anchors expand by centrality (a low-direct-match but heavily-supported claim becomes an anchor); anchor recall carries a conformal band; tested.

## U3. Relevance — tree + orphan resolution (RQ §3, part 2)

- `dimensions/relevance/tree.py`: fixpoint reachability from anchors over confirmed edges; `max_hops` + depth-decay weight from config; depth = relevance grade. `[code]`
- `dimensions/relevance/orphans.py`: for each unreachable claim, classify — **off-topic** (fails on-topic vs question), **stranded/veracity-bearing** (on-topic ∧ `GroundingProvider.entails`-or-contradicts an anchor — the edge the answer didn't draw), **independent-background** (on-topic, no relation). `[T1 + T2]`
- routing: off-topic → penalize; stranded-contradiction → `ConsistencyProvider.route_contradiction` (U6 stub) + completeness signal, **kept relevant**; background → relevant.
- score: `relevance_capped_mean` over depth-graded tree + background; off-ask cap retained.
**Accept:** the flood-zone fixture (true, on-topic, contradicts anchor, answer drew no edge) is classified **stranded-veracity**, kept relevant, and routed — not dropped as off-topic; a genuinely off-topic claim is the only thing scored down; premise-chain fixture (A→B→anchor) attaches via two hops; tested.

## U4. Completeness — reference modes (RQ §2)

- `dimensions/completeness/reference.py`: three modes behind `config.completeness.reference_mode` — `templated` (`requirement_templates.yaml`), `archetype` (`question_archetypes.yaml`, ~8–12 fixed question-shape skeletons instantiated per question), `generated` (**default**, per-question requirement + unit generation). Bottom-up unit drafting stays **extractive** from source spans where possible.
- **stamp `assurance_mode` on the DimensionResult.**
- admissibility gate, dedupe, assignment, scoring: unchanged (already deterministic).
- add the **human-recall-sample** hook: a config-pointed labeled sample whose miss-rate is computed and **reported on the result** as completeness's error bar.
**Accept:** all three modes run; `generated` is the default and works on an arbitrary question with no template; `assurance_mode` + recall-sample miss-rate appear on the DimensionResult; τ-validation unchanged; tested per mode.

## U5. Evidence — parse-first triplets (Evidence §0)

- `pipeline/triplets.py`: primary path = PredPatt/OpenIE predicate-argument tuples → S-P-O `[T1]` (reuse the U1 parse); GeneratorProvider only for the residual (nested/abstractive) `[T3-gen]`, pinned.
**Accept:** cleanly-parseable claims produce triplets with no GeneratorProvider call; only residual claims invoke it; stability metric unchanged; tested.

## U6. Forward-declared `ConsistencyProvider` (RQ §0.5)

- `providers/consistency.py`: interface `edge_sound(premise, conclusion) → bool` (default stub `true` + flag-for-review) and `route_contradiction(claim, anchor)` (default stub records the routed atom, no-op). Constructed by `ProviderFactory`; documented as a Reasoning-category placeholder.
**Accept:** relevance's soundness + contradiction routes resolve to the stub (no dangling reference); a test proves relevance never penalizes on `edge_sound` while the stub returns `true`; swapping the stub later needs no relevance change.

## U7. Config, docs, determinism sweep, fixtures

- New config keys: `extraction.realizer_enabled`; `relevance.{anchor_alpha, edge_tau, max_hops, depth_decay}`; `completeness.reference_mode` + archetype path; consistency stub flags.
- Per-folder READMEs updated with new calculations (tree traversal + depth-decay, orphan classification, reference-mode selection, triplet parse-first) and determinism lines.
- ARCHITECTURE.md: the relevance graph pipeline, the completeness mode selector, the forward-declared consistency interface.
- **Determinism-ledger test** (extend the prior one): assert the score-affecting `[T3]` set is exactly {unsourced_residual, task_success_adequacy, relevance_abstention, decidability_residual, disinterest_residual} and that claim decomposition + relevance contain **no** GeneratorProvider/judge on their scoring paths.
- Fixtures: U1 realizer-impact; U3 flood-zone stranded + premise-chain + off-topic; U4 all three completeness modes on the same question; U5 parse-first triplet.
**Accept:** full offline run green in mock; each fixture shows its expected outcome; ReplayVerifier passes over a whole run; ledger test locks the T3 surface.

---

## Definition of done

- Claim decomposition and relevance are **judge-free on their scoring paths**; extraction is deterministic parse-first with a droppable pinned realizer; relevance is the anchor-tree with orphan resolution.
- Completeness runs **generated-primary** with `archetype`/`templated` upgrades, stamps its assurance mode, and reports the human-recall miss-rate as its error bar — the honest reference-vs-scoring split.
- `ConsistencyProvider` is a working stub; relevance's cross-category routes resolve with defaults and swap cleanly later.
- Evidence triplets are parse-first.
- All new knobs in `config.yaml`; offline-green in mock; providers swap mock→live with no formula change; style addendum honored; determinism-ledger test passes; all tests green.
