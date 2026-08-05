# Reasoning — build spec (v1, implementation-synced)

Four sub-dimensions across two dimensions — **logical_consistency (internal)**, **logical_consistency (validity)**, **assumption_quality**, **uncertainty_handling**. Same format as `response-quality-design.md` v8: a **design flow** paragraph per sub-dimension (pipeline in order, research + exact computations woven in), then a compact role-annotated table. Rule throughout: **AI extracts/reconstructs and judges yes/no; code computes every number.**

**This category is the *owner* of the argument graph.** logical_consistency and validity are checks that accuracy and relevance already half-build (accuracy's inferred-claim residual, relevance's edge-soundness and contradiction-to-anchor). To avoid three teams building three disagreeing NLI sweeps, the **claim graph is a shared service** and **Reasoning owns the semantic labels** (what counts as a contradiction, what counts as a valid step). Consumers import; nobody recomputes. Independence is preserved by the **disjoint-scoring-edge rule** (§6): each dimension's `formula_id` reads only its own atom type, so sharing a structure never means sharing a score.

**Environment realities baked in (confirmed on target):** Bedrock grounding is **E/N only — it cannot emit Contradiction**, so contradiction detection *requires* the **fairseq RoBERTa-large-MNLI** 3-way path (validated running on the target machine); it is a load-bearing dependency here, not optional. **Bedrock Automated Reasoning is not available** in this account/region, so the *sound-verification* path for formalizable steps is a **dormant config flag** (`validity.automated_reasoning: false`) — NLI-primary validity is the built path, Automated Reasoning drops in later without redesign if access appears.

**Tier legend:** `[T1]` pure code · `[T2]` fixed model (NLI/grounding/embeddings) thresholded in code · `[T3-gen]` pinned generation building a frozen reference · `[T3]` judge (one yes/no, ScoringJudge) · `[oracle]` human-maintained config.

---

## 0.1 Provider roster additions (beyond the shared roster)

Reuses the shared providers (GroundingProvider, ScoringJudge, ExplanationJudge, EmbeddingProvider, T1Tools, ClaimExtractor). New to this category:

| Provider / service | Signature | Mock | Live | Used by |
|---|---|---|---|---|
| **ClaimGraph** (shared service, *owned here*) | `build(claims) → {nodes, edges[type∈{entails,contradicts,supports}, score]}` | token-overlap + negation heuristic edges | fairseq-MNLI edges + Titan clustering + `networkx` | internal-consistency, validity; **imported by** relevance (support edges) + accuracy (validity labels) |
| **NLIProvider (3-way, contradiction-capable)** | `label(premise, hypothesis) → {E,N,C, raw}` | coverage+negation heuristic | **fairseq RoBERTa-large-MNLI** (torch.hub, S3-stageable) | contradiction (internal), step-support (validity) — **fairseq required; Bedrock can't emit C** |
| **ReasoningReconstructor** | `reconstruct(answer) → steps[{premises[], conclusion}]` | rule/discourse-split stub | **Bedrock Claude** (pinned prompt) `[T3-gen]` | validity — a *generated, frozen* reference (completeness-style honesty) |
| **AutomatedReasoning** *(dormant)* | `verify(step) → {sound: bool}` | n/a | Bedrock Automated Reasoning policy — **unavailable; flag off** | validity formalizable-step oracle, when/if access appears |

Supporting: `spaCy`+`coreferee` (entity/topic keys, coref), `networkx` (graph + cycle/DAG checks), `negspacy` (negation prefilter), stdlib `math` (ratios, ECE bins). No numpy/scipy needed (ECE is stdlib bins).

---

## 1. logical_consistency (internal) — CRITICAL gate · self-contradiction

**Design flow.** Internal consistency asks whether the answer **contradicts itself** — the property the logical-reasoning survey calls the prerequisite for a trustworthy system: an answer that asserts *A* and *¬A* is unreliable regardless of how well-sourced each half is, so this is a **gate (FAIL on any real contradiction, else PASS)**, not an averaged score. The naive method — NLI every claim pair — is O(n²) and, worse, **atomic-pair NLI is noisy**, so the design contains both problems. **Containment of the pair space:** claims are bucketed by shared topic before pairing — entity/attribute keys from `spaCy`+`coreferee`, plus embedding clusters (Titan/`gensim`), merged with `networkx` — so only *plausibly-related* pairs are checked (contradictions live within a topic, not across unrelated ones), turning O(n²) into O(Σ within-bucket pairs). **Deliberately over-group** (assign a claim to multiple buckets) so a contradiction is never missed by mis-bucketing — the recall test is a *planted, topically-distant contradiction* that must still be caught. **The verifier is 3-way NLI** (`NLIProvider`), and this is exactly where **fairseq is required**: a contradiction is `label == C`, which Bedrock grounding cannot emit. A `negspacy` negation prefilter cheaply flags likely-contradiction pairs before the NLI call. This design follows the contradiction-detection-via-NLI literature ([Contradiction Detection in RAG](https://www.researchgate.net/publication/390405338), 2025, casts it as the N/E/C NLI task and finds contradiction detection *importance-weighted*) and the self-contradiction work ([Self-Contradictory Reasoning](https://arxiv.org/pdf/2311.09603), which shows LLMs are *poor at self-detecting* contradiction — an argument **against** a judge here and **for** the fixed NLI). A **`[T3-res]` scope backstop** handles the one case NLI misidentifies: apparent contradictions that are actually scoped/conditional ("rates rose in Q1" vs "rates fell in Q3" — both true, different scope), sent to the ScoringJudge only when NLI says C but entities/time-scope differ. Score: any surviving `C` → **gate FAIL (1/0)**; the contradiction edges are written into the **shared ClaimGraph** for relevance and completeness to consume (a stranded orphan that contradicts an anchor is *this* edge).

| Step | Tier | Provider/tool | Computation |
|---|---|---|---|
| topic buckets | T1 keys + T2 embed + coref — containment | spaCy/coreferee · Titan/gensim · networkx | over-group claims into shared-topic buckets |
| negation prefilter | T1 — support | negspacy | flag likely-contradiction pairs cheaply |
| per in-bucket pair | T2 — primary, **fairseq required** | NLIProvider (3-way) | contradiction ⟺ `label == C` |
| scope backstop | T3 — residual only | ScoringJudge | C-but-different-scope → not a contradiction |
| score | code — **gate** | — | any C survives → FAIL (0) else PASS (1); edges → ClaimGraph |

**Confirm:** bucketing keys + over-group factor; `negspacy` on/off; scope-backstop trigger rule.

---

## 2. logical_consistency (validity) — MAJOR · does each step follow

**Design flow — honest framing first.** Validity asks whether the answer's **inferences actually follow** — "GDP fell, therefore air quality worsened" can be internally consistent (no contradiction) yet *invalid* (non-sequitur). Like completeness, this needs a **generated reference**: the argument structure (premise→conclusion steps) is not laid out in the text as such, so `ReasoningReconstructor` must **produce** it `[T3-gen]` — and, exactly as with completeness's nuggets, the defensibility is the **reference-vs-scoring split**: the reconstruction is *bounded generation, pinned, stability-measured, τ-validated*, then the *scoring of each step is deterministic*. This matters because the step-verification literature is clear that **verifying only final answers misses flawed intermediate steps that propagate and distort conclusions** ([Efficient Verification of LLM Reasoning Steps](https://openreview.net/pdf?id=svQuvBYaCA), 2025), so step-level checking is the right unit. **Per reconstructed step, three checks:** **premises-support-conclusion?** — an **entailment** question, so fixed 3-way NLI (`NLIProvider`: does the conjunction of premises entail the conclusion) `[T2]`, *not* a judge, by the same DIVER-QA/argument-mining logic used for relevance edges; **no-hidden-premise?** — the enthymeme test: if premises alone don't entail the conclusion but do *with* an obvious added premise, the step relies on an unstated assumption → flag and route to **assumption_quality** (§3), an overlap contract not a recompute; **scoped?** — the conclusion doesn't over-generalize beyond the premises' scope `[T2/T3-res]`. A **broken link gates** (an invalid load-bearing step invalidates the conclusion), otherwise `score = valid-steps / steps`. **Automated Reasoning is the dormant upgrade:** for the *formalizable* subset of steps, a sound logical oracle would replace NLI's ~90% with a proof — but it's unavailable here, so it sits behind `validity.automated_reasoning` (off), and NLI-primary is the built path; the honest note is that NLI validity is *fixed and reproducible* but not *sound*, and the residual judge covers genuinely ambiguous steps. **Overlap contracts:** the per-step support label is written to the **ClaimGraph** as the edge-soundness property — this *is* `ConsistencyProvider.edge_sound` that relevance forward-declared, and *is* the inference-validity check accuracy's inferred-claim residual imports. Computed once here, read by both.

| Step | Tier | Provider/tool | Computation |
|---|---|---|---|
| reconstruct steps | T3-gen — reference build (pinned, τ-validated) | ReasoningReconstructor | `steps[{premises, conclusion}]` |
| premises-support? | T2 — primary (NLI, fairseq/Bedrock) | NLIProvider | `⋀premises ⊨ conclusion` ⟺ E |
| no-hidden-premise? | T2 — primary; **routes to §3** | NLIProvider | entails only *with* added premise → enthymeme flag |
| scoped? | T2 primary · T3 residual | NLIProvider · ScoringJudge | conclusion within premises' scope |
| *(formalizable step)* | dormant oracle | AutomatedReasoning (**off**) | sound proof — future upgrade |
| score | code — broken link **gates** | `valid / steps` | edge-soundness → ClaimGraph (relevance + accuracy import) |

**Confirm:** reconstruction pinning + τ cadence; gate on any broken link vs only load-bearing; `automated_reasoning` (off now).

---

## 3. assumption_quality — MAJOR · quality of what's assumed

**Design flow.** assumption_quality scores the **unstated things the answer relies on** — and its boundary with neighbours must be crisp to avoid triple-counting (the overlap you flagged): an **assumption** is a premise the argument *needs but doesn't establish* (reasoning); a fact a *complete* answer should contain is a **nugget** (completeness); a *stated* claim with no source is **unsourced** (accuracy). Same "not grounded" observation, three different questions — assumption_quality owns only the *load-bearing unstated premise*, which §2's no-hidden-premise test already surfaces (the enthymeme flags feed straight in, so assumptions aren't re-discovered). **Surfacing** the assumptions is `[T3-gen]` (pinned), but the **per-assumption checks are mostly reducible**: **reasonable?** — is the assumption defensible — the one genuine judgment residual `[T3]` (it *gates* the assumption: an unreasonable load-bearing assumption is the failure); **explicit?** — is it actually stated vs silently relied on `[T1/T2]` (string/NLI presence check); **necessary?** — an **NLI ablation**, not a judgment: remove the assumption and re-run the step's entailment — if the conclusion still follows, the assumption wasn't necessary `[T2]`; **hedged-if-load-bearing?** — if necessary and not certain, is it flagged as an assumption `[T2/T3-res]`. Per assumption `= passes / applicable` (reasonable gates), then `score = mean over made assumptions`, and **abstain if none** — the denominator is *assumptions that exist*, not slots, so an answer that needs no assumptions isn't penalized. This keeps the judge to the single "reasonable?" atom; the rest is NLI + code.

| Step | Tier | Provider/tool | Computation |
|---|---|---|---|
| surface assumptions | T3-gen — reference build; **imports §2 enthymeme flags** | ReasoningReconstructor | load-bearing unstated premises |
| reasonable? | T3 — residual, **gates the assumption** | ScoringJudge | defensible? |
| explicit? | T1/T2 — primary | T1Tools/NLIProvider | stated vs silently relied on |
| necessary? | T2 — primary (**ablation, not judgment**) | NLIProvider | remove → does step still entail? |
| hedged-if-load-bearing? | T2 primary · T3 residual | NLIProvider · ScoringJudge | flagged when necessary ∧ uncertain |
| score | code | `mean(passes/applicable)`; abstain if none | reasonable gates each |

**Confirm:** assumption-surface pinning; does "reasonable" hard-gate or weight; abstain threshold.

---

## 4. uncertainty_handling — MAJOR · does it handle what it doesn't know

**Design flow.** uncertainty_handling asks whether the answer **appropriately flags what's uncertain or unknown** — and it has two distinct, mostly-deterministic parts. **Part A, open-slot handling:** enumerate the question's open slots (ambiguous referents, missing preconditions, unanswerable sub-parts) `[T3-gen]` — reusing relevance's abstention machinery and the assumption slots, not a fresh pass — and per slot ask *clarified-or-hedged?* — was it resolved or appropriately flagged rather than silently guessed. Hedge/clarification detection is **substantially lexical/syntactic** (hedge cues, epistemic markers via `T1Tools` + POS), residual to a fixed classifier `[T1/T2]`, so the judge shrinks to genuinely ambiguous cases. `handled = appropriately-handled / open-slots`. This shares the abstention verdict with relevance (a proper decline to an unanswerable question is *good* uncertainty handling), an overlap contract not a recompute. **Part B, calibration (ECE):** *when the answer states confidences and ground truth is available*, compute **Expected Calibration Error** — bin predictions by stated confidence, `ECE = Σ_b (n_b/N)·|acc_b − conf_b|`, pure stdlib `math` `[T1]`. The literature warns two things the design respects: verbalized confidence **clusters on round-number anchors and is often miscalibrated** ([confidence-calibration review](https://www.emergentmind.com/topics/confidence-calibration-in-llms), 2026; [Reasoning's Razor](https://arxiv.org/pdf/2510.21049), 2025), so ECE is reported as a *diagnostic* alongside the handling score, not blended into one number that hides which is failing; and ECE has known limitations, so it's presented with its bin count and n, not as a lone verdict. Where confidences+truth are absent (the common case), Part B abstains and only Part A scores. This keeps uncertainty_handling almost entirely `[T1]` (hedge detection + ECE math) with a thin judge residual.

| Step | Tier | Provider/tool | Computation |
|---|---|---|---|
| enumerate open slots | T3-gen — reference build; **reuses §3 slots + relevance abstention** | ReasoningReconstructor | ambiguous referents, missing preconditions |
| clarified-or-hedged? | T1 primary (hedge cues) · T2/T3 residual | T1Tools/POS · NLIProvider · ScoringJudge | resolved or appropriately flagged |
| ECE *(if confidences+truth)* | T1 — primary, diagnostic | stdlib `math` | `Σ (n_b/N)·|acc_b − conf_b|`; report bins + n |
| score | code | `handled / open-slots` (+ ECE reported) | abstain Part B if no confidences |

**Confirm:** open-slot source (shared vs own); ECE bin count; hedge lexicon; ECE blended vs reported-separately (recommend separate).

---

## 5. Overlap & isolation contracts (the reason this category is owner)

- **ClaimGraph is shared structure; Reasoning owns the labels.** `contradicts` edges (§1), `edge_sound` labels (§2) are computed here and written to the graph. **relevance** imports `supports` edges (reachability) + consumes `contradicts`-to-anchor (routing); **accuracy** imports `edge_sound` for its inferred-claim residual. One contradiction set, one soundness set — no dimension recomputes another's.
- **Disjoint scoring edges (the isolation guarantee).** relevance scores *reachability*; internal-consistency scores *contradiction edges*; validity scores *soundness of support edges*; accuracy imports a label it doesn't recompute. A claim can be **connected (relevance ✓) but unsound (validity ✗)** — the dissociation proves independence. Enforced by a **no-shared-scoring-atom test**: assert no AtomRecord feeds more than one dimension's `formula_id`.
- **Assumption boundary:** load-bearing unstated premise → assumption_quality; should-be-present fact → completeness; stated-but-unsourced → accuracy. The enthymeme flag (§2) is the single hand-off; no double-count.
- **Uncertainty boundary:** abstention verdict shared with relevance; ECE calibration store shared with Evidence §5 conformal (same calibration data, different statistic).
- **Forward-declared stubs now filled:** `ConsistencyProvider.edge_sound` = §2's step-support label; `route_contradiction` = §1's contradiction edge. Relevance's stubs resolve to these when Reasoning is built, no relevance code change.

**Development summary** — where each tier does work:

| Sub-dimension | Output | T1 | T2 (fixed NLI) | T3 |
|---|---|---|---|---|
| consistency (internal) | **gate** 1/0 | bucketing; negation prefilter; gate | contradiction (C) — primary, **fairseq** | scope backstop (residual) |
| consistency (validity) | `valid/steps` (+gate) | ablation bookkeeping; gate | step-support; hidden-premise; scope — primary | gen: reconstruction · scope residual |
| assumption_quality | `mean(passes/applicable)` | explicit; composition | necessary (ablation); hedged — primary | gen: surface · reasonable (residual gate) |
| uncertainty_handling | `handled/slots` (+ECE) | hedge cues; **ECE math** — primary | clarified residual | gen: slots · thin residual |

The judge footprint here is small and honest: **reconstruction/surfacing** are pinned generation (like completeness's nuggets — a generated reference, deterministically scored); the score-affecting `ScoringJudge` residuals are **scope backstop, "reasonable?", and thin clarified/scope residuals** — each a single boolean, reference-grounded, off the gate path. Contradiction and step-validity — the load-bearing checks — are **fixed 3-way NLI**, which is *preferable* to a judge here since LLMs self-detect contradiction poorly. Automated Reasoning would upgrade validity to *sound* on formalizable steps but is dormant (unavailable).

## Sources
Consistency: [Empowering LLMs with Logical Reasoning survey](https://arxiv.org/pdf/2502.15652) (2025, consistency as trustworthiness prerequisite) · [Measuring/Improving Logical Consistency](https://www.researchgate.net/publication/384630941) (2024, DAG/transitivity) · [Contradiction Detection in RAG](https://www.researchgate.net/publication/390405338) (2025, NLI N/E/C, importance-weighted) · [Self-Contradictory Reasoning](https://arxiv.org/pdf/2311.09603) (LLMs self-detect contradiction poorly → use fixed NLI) · [Transitive self-consistency of NLI](https://aclanthology.org/2025.emnlp-main.1152.pdf) (EMNLP 2025).
Validity / step verification: [Efficient Verification of LLM Reasoning Steps](https://openreview.net/pdf?id=svQuvBYaCA) (2025, step-level > final-answer) · [Reasoning with Confidence / UHead](https://arxiv.org/html/2511.06209v2) (2025) · [Trustworthiness in Reasoning survey](https://arxiv.org/pdf/2509.03871) (2025) · ReasoningFlow (infer-present vs infer-valid) — *cite by name*.
Uncertainty / calibration: [Confidence Calibration in LLMs review](https://www.emergentmind.com/topics/confidence-calibration-in-llms) (2026, verbalized-confidence anchoring; conformal calibration) · [Reasoning's Razor](https://arxiv.org/pdf/2510.21049) (2025, ECE limitations, ranking metrics) · [Process Supervision of Confidence Margin](https://arxiv.org/html/2604.23333) (2026).
Tooling: fairseq RoBERTa-large-MNLI (3-way NLI, contradiction) · `spaCy`/`coreferee`/`networkx`/`negspacy`/`gensim` · Bedrock Automated Reasoning (dormant) — *cite by name*.
