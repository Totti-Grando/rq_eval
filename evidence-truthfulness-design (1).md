# Evidence & Truthfulness — build spec (v2, implementation-synced)

Four dimensions — **groundedness, hallucination, source_quality, source_attribution** — plus the **conformal factuality** layer (§5) and the import contracts accuracy consumes (§6). Same format as `response-quality-design.md` v8: each section opens with a **design flow** paragraph (the pipeline in order, with research reasoning and exact computations in the narrative), then a compact role-annotated table. Rule throughout: **AI extracts and judges yes/no; code computes every number.**

## 0.1 The provider roster — every tool in this category, defined

All model/tool access goes through the same typed interfaces as Response Quality (§0.1 there), constructed only by the `ProviderFactory` from `config.yaml`. Restated here in full so nothing in this doc is ambiguous:

| Provider | Signature | Mock implementation (offline) | Live implementation (target machine) | Used here by |
|---|---|---|---|---|
| **GroundingProvider** | `entails(premise, hypothesis) → {label ∈ {E,N,C}, raw_score}` | token-coverage + negation heuristic: `cov = overlap(hyp, prem)`; `cov ≥ contra_tau ∧ negation-mismatch → C`; `cov ≥ entail_tau → E`; else `N`; `raw = cov` | **Bedrock ApplyGuardrail** contextual-grounding (`score ≥ entail_tau → E`, else `N` — **no C label**) **or** **fairseq RoBERTa-large-MNLI** via `torch.hub` (native 3-way, `argmax{C,N,E}`, `raw = P(entail)`) — selected by `models.nli` | §1 entailment (premise = context span) · §3 supports-check (premise = source) · §4 attribution (premise = *cited* chunk) — **one verifier, three premises** |
| **GeneratorProvider** | `generate(prompt, seed) → text` | parse/rule-based stand-ins (triplets: clause-boundary splitter) | **Bedrock Claude** via `boto3` Converse, versioned prompts | §0 triplet decomposition `[T3-gen]` |
| **JudgeProvider** | `binary(question, context) → {verdict: bool, reason}` | seeded hash → stable bool | **Bedrock Claude** via Converse | §3 disinterest (sampled) — the category's only judge call |
| **EmbeddingProvider** | `embed(texts) → vectors` | 64-dim md5-hashed token counts, L2-normalized | **Bedrock Titan Text Embeddings V2** via InvokeModel | §1 similarity pre-filter |
| **ResolverProvider** | `resolve(url_or_doi) → bool` | fabricated-marker string check | `urllib` HEAD (HTTP 200–399) · DOI-registry lookup (config-gated) | §2 fabrication gate · §3 reachable |

Supporting non-model tools: **`T1Tools`** (stdlib `re` — citation-id parsing, number parsing, pattern checks), **`config/reliability_list.yaml`** (`[oracle]` — the human-maintained domain allow/deny-list for §3), **`CalibrationStore`** (`config/calibration/*.jsonl`, pinned human-labeled examples for §5), **`alce.py`** (citation precision/recall math), **`conformal.py` + `ConformalStratifier`** (split-conformal in stdlib `math`), the **formula registry** (`mean`, `unsupported_rate`, …), and stdlib-`math` statistics (Wilson CI, bands, min-n abstention). Packages actually imported: **pydantic**, **PyYAML**, stdlib (`hashlib`, `re`, `math`, `json`, `sqlite3`, `urllib`), **boto3**, optional **torch+fairseq**. **No numpy, no scipy, no ragas.**

**The organizing taxonomy** ([factuality-area synthesis](https://papers.lunadong.com/area/factuality)): the field separates **Source Faithfulness** — is the claim supported by the *provided* source — from **World Factuality** — is the claim *true*. Groundedness + attribution measure faithfulness; source_quality is the bridge to factuality (a source-faithful claim is only *true* if its source is trustworthy). That is exactly accuracy's `groundedness × source_quality × attribution`, with a name for each axis.

---

## 0. Build once — claim-triplets (`pipeline/triplets.py`)

**Design flow.** The category reuses the cached, decontextualized, verifiable claims from Response Quality §0.2 and refines the *verification unit* one level further: each claim is decomposed into **claim-triplets** — `subject | predicate | object` — because [RefChecker](https://aclanthology.org/2024.emnlp-main.395.pdf) (Hu et al., EMNLP 2024) showed that **triplet-level checking outperforms sentence- and sub-sentence-level checking by 4–9 points** and is the current gold standard for fine-grained hallucination detection: "Einstein developed quantum mechanics in Berlin in 1905" becomes (Einstein, developed, quantum-mechanics), (Einstein, was-in, Berlin), (event, occurred-in, 1905), each checked separately so a partially-wrong compound claim can't pass whole. The decomposition runs through the **GeneratorProvider** `[T3-gen]` (mock: a parse-splitter on clause boundaries; live: Bedrock with a versioned prompt), each triplet carrying its claim id, citation, and source pointer. Like every generated reference it is **pinned** (`pins.triplet_extractor_version`) and its **stability is measured**: `stability = |∩ triplet-ids| / |∪ triplet-ids|` across N re-runs — RefChecker is backend-agnostic (its extractor/checker take any LLM backend), so nothing here touches Hugging Face.

| Step | Tier | Provider/tool | Computation |
|---|---|---|---|
| decompose claim → triplets | T3-gen — reference build (pinned, stability-measured) | GeneratorProvider (parse-split → Bedrock) | `subject | predicate | object`, provenance carried |
| stability | code | pipeline harness | `|∩ ids| / |∪ ids|` over N re-runs |

---

## 1. groundedness — MAJOR · source faithfulness — `|E| / |total|`

**Design flow.** Groundedness is the category's reproducible core: per triplet, is it entailed by the retrieved context. A **similarity pre-filter** `[T1]` first picks the nearest context span per triplet — `EmbeddingProvider` cosine — purely to hand the verifier a focused premise; it is never the score, because similarity cannot see negation ("revenue rose" / "revenue did not rise" are near-identical vectors). The verifier is the single **GroundingProvider**, three-way per RefChecker's checker: **Entailment** (supported), **Neutral** (source silent), **Contradiction** (source says the opposite) — a fixed model, never a generative judge, which is what keeps the dimension `[T2]` and bit-for-bit reproducible from stamped labels. One honesty note the backend table carries: the **Bedrock backend cannot emit C** (its grounding score thresholds to E/N only), so if the Neutral-vs-Contradiction severity split matters — and §2 argues it does — run `models.nli: fairseq` or add a judge backstop for suspected contradictions. Code computes `groundedness = |E-labeled triplets| / |total triplets|` (the RAGAS-faithfulness form `|supported|/|total|`), and two things are exported: the per-claim boolean **`grounded = AND(all its triplets == E)`** — accuracy's atom 1 — and each triplet's `raw_score`, which feeds the conformal layer (§5), where in [Mohri & Hashimoto's](https://proceedings.mlr.press/v235/mohri24a.html) framing the correctness of an output is an uncertainty problem over its *entailment set*.

| Step | Tier | Provider/tool | Computation |
|---|---|---|---|
| pre-filter | T1 — support only (never the score) | EmbeddingProvider | nearest context span per triplet |
| three-way entailment | T2 — primary | GroundingProvider | label ∈ {E, N, C} per triplet |
| score | code | `mean` | `|E| / |total|` |
| exports | code | — | per-claim `AND(E)` → accuracy; `raw_score` → conformal (§5) |

**Confirm:** verifier backend (fairseq if C matters); `entail_tau`.

---

## 2. hallucination — MAJOR · unsupported rate + fabrication gate

**Design flow.** Hallucination is two different failures wearing one name, and the design splits them because one is semantic and one is fully deterministic. The **unsupported-claim rate** is read directly off §1's triplet verdicts — `unsupported_rate = 1 − groundedness` — but reported with the **Neutral/Contradiction split** (`neutral_rate = |N|/total`, `contradiction_rate = |C|/total`), because asserting *against* the evidence is a materially worse failure than asserting *beyond* it, and a single blended rate hides which one you have. The **fabrication gate** is the deterministic half `[T1]` and it gates the run: does the cited source *exist* — `cited id ∈ retrieved set` (set-membership) **or** `ResolverProvider.resolve(url/doi)` succeeds (mock: not a fabricated-marker string; live: `urllib` HEAD returning HTTP 200–399, or a DOI-registry lookup, config-gated). Existence is deliberately a **separate axis from support** — a citation can exist and not support (that's attribution's job, §4), or be fabricated outright — and the fabricated-reference problem is real at scale: audits found **50+ citation hallucinations across 300 ICLR 2026 submissions** ([CiteAudit](https://arxiv.org/html/2605.06635v1), 2026). Any `¬exists` forces the band to **R** through the standard gate path — a fabricated citation is an integrity failure, not a quality deduction.

| Step | Tier | Provider/tool | Computation |
|---|---|---|---|
| unsupported rate | code — reads §1 (no new model calls) | triplet verdicts | `1 − groundedness`; `|N|/tot` and `|C|/tot` reported separately |
| fabrication gate | T1 — primary, **gates** | set-membership + ResolverProvider | `exists = (id ∈ retrieved) ∨ resolve(url/doi)`; any `¬exists → FAIL (band R)` |

**Confirm:** DOI/URL registry in scope (config-gated); Contradiction weighting in downstream reporting.

---

## 3. source_quality — MAJOR · the bridge to world factuality — `mean(7 properties)`

**Design flow.** Source_quality is what turns "faithful to its source" into "true in the world," and it is **what accuracy imports as `source-adequate?`**. The credibility-signals literature ([survey](https://arxiv.org/pdf/2410.21360), 2024; [media background-check store](https://arxiv.org/pdf/2607.02383), 2026) organizes trustworthiness into signal families — *source/domain reputation, author identifiability, recency, corroboration across independent sources, support for the specific claim* — and shows most are **deterministically extractable** from the document and its metadata, which is why six of the seven properties here are `[T1]`/`[T2]` and only one needs judgment. Per source: **reachable** — internal-corpus chunks pass by construction, external URLs must `resolve()` `[T1]`; **dated & fresh** — a date is present ∧ `date ≤ as_of_date` (the point-in-time window; critical in finance where a stale filing is a wrong filing) `[T1]`; **authored** — an author or issuing body is present `[T1]`; **reputable domain** — the domain is checked against **`config/reliability_list.yaml`**, a human-maintained allow/deny-list `[oracle]`: deny → False, allow → True, unknown-under-allow-list-mode → False (conservative default); **corroborated** — `|distinct domains/authors among sources that entail the claim| ≥ corroboration_min` `[T1 count over §1 verdicts]` — independence measured by distinct provenance, not source count; **supports the claim** — `GroundingProvider.entails(source, claim) == E` `[T2]`, because a good source that doesn't support *this* claim is not adequate *for it*; and **disinterested** — is the source self-serving or conflicted — the one genuinely semantic residue, a `JudgeProvider` boolean **sampled** at `disinterest_sample_rate` (assumed True when unsampled, and the sampling rate is a config dial between cost and coverage) `[T3-samp]`. Code computes `source_quality = mean(property booleans)` per source, and the exported boolean accuracy consumes is **`source-adequate? = source_quality ≥ adequacy_threshold`**. Per profile: Nexa's internal corpus satisfies the metadata properties by construction, so adequacy defaults true (`source_adequate_default`); RavenPack's open web keeps every property live.

| Step | Tier | Provider/tool | Computation |
|---|---|---|---|
| reachable / dated / authored | T1 — primary | ResolverProvider · `str`/date · code | resolve; `date ≤ as_of`; author present |
| reputable domain | oracle — primary (human-maintained) | `reliability_list.yaml` | deny→F; allow→T; unknown→F (allow-list mode) |
| corroborated | T1 — primary (counts §1 verdicts) | code | distinct entailing domains/authors ≥ `corroboration_min` |
| supports the claim | T2 — primary | GroundingProvider | `entails(source, claim) = E` |
| disinterested | T3 — sampled residual (the only judge use) | JudgeProvider | sampled at `disinterest_sample_rate`, else True |
| score / export | code | `mean` | `mean(7)`; `adequate = score ≥ adequacy_threshold` → accuracy |

**Confirm:** reliability-list contents + mode (allow vs deny); `adequacy_threshold`; `corroboration_min` + independence rule; `disinterest_sample_rate`.

---

## 4. source_attribution — MAJOR · ALCE citation precision — `attributed?`

**Design flow.** Attribution asks whether each claim is credited to the source that *actually supports it* — a failure axis fully distinct from correctness, since **an answer can be correct while its citations don't support its claims**, and in a regulated setting a broken evidence trail is its own finding. The standard is **[ALCE](https://arxiv.org/pdf/2606.23915)** (Gao et al., 2023): per citation, the **GroundingProvider** runs on the **cited chunk as premise** — the same single verifier as §1, different premise — with the verdict mapped to [AttrScore's](https://arxiv.org/pdf/2411.14199) three-way labels (**Attributable** ⟺ `E`, **Contradictory**, **Extrapolatory**), or the four-way CAQA scheme (Supported / Insufficient / Contradictory / Irrelevant) when finer diagnostics are wanted (`labels: three | four` in config). Code then computes both ALCE metrics in `alce.py`: **citation precision** = `|attributable citations| / |citations|` (no padding an answer with decorative cites) and **citation recall** = `|statements whose citation set supports them| / |cited statements|`. Two design rules from the literature: **precision-favoring** — false positives (wrongly calling a claim attributed) are more harmful than false negatives in faithfulness judging ([Evidence-Based QA](https://arxiv.org/pdf/2402.08277), 2024), so the threshold biases toward precision; and **no-citation claims are excluded here** — they route to accuracy's unsourced residual, asserted once, no double-count, since a claim with no relevant source should *say so* rather than manufacture a cite. Because support-judging is hard even for strong models — **[AttributionBench](https://arxiv.org/html/2605.06635v1)**: fine-tuned GPT-3.5 reaches only ~80% macro-F1 — a single verdict is not trusted bare: each verdict's confidence feeds the **conformal layer (§5)**, and the exported boolean accuracy consumes is **`attributed? = Attributable ∧ (confidence ≥ conformal threshold)`**, i.e. a verdict *with a calibrated guarantee attached*, which is the "structural guarantee" post-hoc attribution otherwise lacks ([RegOps](https://arxiv.org/pdf/2605.29742), 2026).

| Step | Tier | Provider/tool | Computation |
|---|---|---|---|
| per-citation verdict | T2 — primary (same verifier, cited chunk as premise) | GroundingProvider | 3-way (Attributable ⟺ E) or 4-way per config |
| ALCE metrics | code | `alce.py` | `precision = |attributable| / |citations|`; `recall = |supported statements| / |cited statements|` |
| no-citation handling | code — routing rule | — | excluded here → accuracy's unsourced residual (no double-count) |
| score / export | code + §5 | `mean` + conformal | dimension = citation precision (recall reported); `attributed? = Attributable ∧ conformal-confident` → accuracy |

**Confirm:** three- vs four-way labels; precision threshold; conformal coverage level (§5's α).

---

## 5. Conformal factuality — the statistical guarantee (`scoring/conformal.py`)

**Design flow.** The NLI verdicts in §1/§4 are strong but imperfect (~80% F1 on attribution), and a certification harness cannot rest on "the classifier is usually right." **Conformal factuality** ([Mohri & Hashimoto, ICML 2024](https://proceedings.mlr.press/v235/mohri24a.html)) converts the verifier's confidences into a **distribution-free, finite-sample guarantee**, works with **any black-box LM, and needs very few human-annotated samples** — exactly the Bedrock, no-model-internals setting. The implementation is split-conformal in pure `math`: the verifier's `raw_score` per claim/triplet is the **confidence**; on a pinned, human-labeled **calibration set** (`CalibrationStore`, `config/calibration/*.jsonl`, each row `{claim, context, label, stratum}`), nonconformity is computed over the *factual* examples as `νᵢ = 1 − confidenceᵢ`; the threshold index is `k = ⌈(1−α)(n+1)⌉` (clamped to n), `τ̂` = the k-th smallest ν, and the retention rule is **`confidence ≥ 1 − τ̂`**. The guarantee that results is
```
1 − α  ≤  P(retained claim is factual)  ≤  1 − α + 1/(n+1)
```
(the finite-sample coverage bound of split conformal; [Angelopoulos & Bates 2023](https://arxiv.org/abs/2107.07511); Vovk et al. 2005; Lei et al. 2018). Reading it: `α` is *your chosen error budget*; the **left side is the floor** — retained claims are factual at least `1−α` of the time, whatever the score distribution looks like; the **right side is the ceiling** — overshoot is at most `1/(n+1)`, the slack from picking a quantile among finitely many calibration points. Worked numbers at α = 0.05: n = 100 → band **[95.0%, 95.99%]**; n = 1000 → **[95.0%, 95.1%]**; the floor arrives immediately even at small n (hence "very few samples") and more data only tightens the band — letting the harness state, defensibly: *"of the claims retained as source-supported, at least 95% are genuinely supported, calibrated on n human-verified examples."* Sub-claim-level use is validated across domains ([Conformal-RAG](https://arxiv.org/pdf/2603.16817), 2025; TRAQ 2023; Conflare 2024). **The caveat is that the guarantee is *marginal*** — it holds on average, not automatically within every subgroup (97% on easy claims and 92% on hard ones still averages 95%); per [Adaptive Conformal Prediction](https://arxiv.org/html/2604.13991v1) (2026) a global threshold over-covers hard strata and under-covers easy ones, so when a regulator needs uniform coverage the **`ConformalStratifier`** calibrates per stratum (question-type / source-type, config `per_stratum`). Everything here is deterministic given the calibration set + α: the threshold, retention, and band `[1−α, min(1, 1−α + 1/(n+1))]` are stamped on the DimensionResult, and the layer **abstains below `min_calibration_n`**.

| Step | Tier | Provider/tool | Computation |
|---|---|---|---|
| confidence | T2 — input (from §1/§4 raw_scores) | GroundingProvider | per calibration example + per live claim |
| calibrate | code — primary (pure `math`) | `conformal.py` + CalibrationStore | `νᵢ = 1−confᵢ`; `k = ⌈(1−α)(n+1)⌉`; `τ̂` = k-th smallest; threshold `= 1−τ̂` |
| retain + guarantee | code | — | retain ⟺ `conf ≥ threshold`; band `[1−α, 1−α+1/(n+1)]` stamped; abstain if `n < min_calibration_n` |
| per-stratum | code | `ConformalStratifier` | same, grouped by config strata (marginal→conditional coverage) |

**Confirm:** `α`; calibration-set size + refresh cadence; `per_stratum` keys.

---

## 6. Contracts — what accuracy imports (no stubs remain)

- `GroundingProvider.entails(premise, hypothesis) → {label, raw_score}` — **one verifier, three premises**: context span (§1), cited chunk (§4), source (§3.supports). Backend by `models.nli`; Bedrock = E/N only, fairseq = full 3-way.
- `SourceQualityProvider.adequate(source, claim) → bool` — §3's `mean(7) ≥ adequacy_threshold`; `source_adequate_default: true` for the Nexa profile.
- `AttributionProvider.attributed(claim, cited_chunk) → {bool, confidence}` — §4's verdict ∧ §5's conformal retention.
- All emit AtomRecords; every score above replays from `atom_ids + formula_id` via the ReplayVerifier with no model call.

**Development summary** — where each tier does work:

| Dimension | formula_id | T1 | T2 | T3 |
|---|---|---|---|---|
| groundedness | `mean` | pre-filter (support) | 3-way entailment — primary | gen: triplets (pinned) |
| hallucination | `unsupported_rate` | fabrication existence — primary, **gates** | reads §1 verdicts | — |
| source_quality | `mean` | 5 metadata/count properties — primary · oracle list | supports-claim | judge: disinterest, sampled residual |
| source_attribution | `mean` (precision) | ALCE math; routing | cited-chunk entailment — primary | — |
| conformal (§5) | — | calibration + retention (pure math) | confidences in | — |

The category's single gate (fabrication) is pure `[T1]`; the judge appears only in pinned triplet extraction and the sampled disinterest check; every number is a formula over stamped labels — with §5 adding the calibrated, regulator-grade guarantee on top of the hardest verdicts.

## Sources
Faithfulness / hallucination: [RefChecker](https://aclanthology.org/2024.emnlp-main.395.pdf) (EMNLP 2024, triplets, +4–9 pts) · [factuality-area synthesis](https://papers.lunadong.com/area/factuality) (Source-Faithfulness vs World-Factuality) · [FinReflectKG-HalluBench](https://arxiv.org/html/2603.20252v1) (2026, RefChecker in financial QA) · MiniCheck (2024), HHEM (Vectara), SelfCheckGPT — *cite by name*.
Attribution / citation: [ALCE](https://arxiv.org/pdf/2606.23915) (2023) · [AttrScore](https://arxiv.org/pdf/2411.14199) (2023) · [AttributionBench / CAQA / CiteAudit](https://arxiv.org/html/2605.06635v1) · [CiteGuard](https://arxiv.org/html/2510.17853v4) (2026) · [VeriCite](https://arxiv.org/html/2510.11394v1) (2025) · [precision-favoring + no-citation](https://arxiv.org/pdf/2402.08277) (2024) · [RegOps](https://arxiv.org/pdf/2605.29742) (2026) · AIS (Rashkin et al. 2023) — *cite by name*.
Source quality: [credibility-signals survey](https://arxiv.org/pdf/2410.21360) (2024) · [media background-check store](https://arxiv.org/pdf/2607.02383) (2026).
Conformal: [Mohri & Hashimoto](https://proceedings.mlr.press/v235/mohri24a.html) (ICML 2024) · [Angelopoulos & Bates](https://arxiv.org/abs/2107.07511) (2023) · [Conformal-RAG](https://arxiv.org/pdf/2603.16817) (2025) · [Adaptive Conformal Prediction](https://arxiv.org/html/2604.13991v1) (2026) · Vovk et al. (2005), Lei et al. (2018), TRAQ (2023), Conflare (2024) — *cite by name*.
Tooling: [Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-components.html) · [fairseq RoBERTa-MNLI](https://pytorch.org/hub/pytorch_fairseq_roberta/) · [RefChecker repo](https://github.com/amazon-science/RefChecker).
