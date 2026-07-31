# Evidence & Truthfulness — build spec (v1)

Four dimensions — **groundedness, hallucination, source_quality, source_attribution** — same style as `response-quality-design.md`: one section each, numbered build steps, tier tags, tools named, research linked inline. Rule throughout: **AI extracts and judges yes/no; code computes every number.**

**Why this category is written next:** accuracy (Response Quality §1) *imports* `source-adequate?` from **source_quality** and `attributed?` from **source_attribution**. Those were interface stubs; this doc specifies them and names the concrete providers accuracy will call. The import contracts are in §5.

**The organizing taxonomy** (resolves the usual terminological muddle): the field now separates **Source Faithfulness** (is the claim supported by the *provided* source) from **World Factuality** (is the claim *true*) — a distinction the RefChecker line established. Groundedness + attribution measure **source faithfulness**; source_quality is the bridge to **world factuality** (a source-faithful claim is only *true* if the source is *trustworthy*). This is exactly accuracy's `groundedness × source_quality × attribution` — now with names for each axis.

---

## 0. Build once — claims → claim-triplets

Reuse the cached atomic, decontextualized, verifiable claim set from Response Quality §0. Add one verification-unit refinement:

1. **Decompose each claim into claim-triplets** `[T3-gen]` — subject-predicate-object, per **[RefChecker](https://aclanthology.org/2024.emnlp-main.395.pdf)** (Hu et al., EMNLP 2024): "Einstein developed quantum mechanics in Berlin in 1905" → (Einstein, developed, quantum-mechanics), (Einstein, was-in, Berlin), (this, occurred-in, 1905), each checked separately. RefChecker showed **triplet-level checking beats sentence/sub-sentence checking by 4–9 points** and is the current gold standard for fine-grained hallucination detection. Triplets are the unit fed to the verifiers below (pinned + stability-measured, like the claim extractor).

*Output:* per claim, a set of checkable triplets, each carrying the claim's citation (if any) and source pointer.

---

## 1. groundedness — MAJOR · source faithfulness

Is each claim entailed by the retrieved context — **source faithfulness**, the reproducible T2 core the whole category rests on.

1. **Similarity pre-filter** `[T1]` — for each triplet, retrieve the nearest context spans (Titan embeddings) to hand the verifier a focused premise; cheap, not the score.
2. **Three-way entailment** `[T2]` — classify each triplet against its context as **Entailment / Neutral / Contradiction** (RefChecker's checker `C`; ALCE's NLI baseline). Entailment = supported; Neutral/Contradiction = unsupported. Use a **fixed model**, never a generative judge: [Bedrock contextual-grounding](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-components.html), [fairseq RoBERTa-MNLI](https://pytorch.org/hub/pytorch_fairseq_roberta/) (torch.hub, non-HF), or [RefChecker](https://github.com/amazon-science/RefChecker) configured with a Bedrock backend (its extractor/checker are backend-agnostic — like RAGAS, no HF weights required).
3. **Score** `[code]`: `groundedness = |supported triplets| / |total triplets|` (the RAGAS-faithfulness form `|V|/|S|`).

*Export:* the per-claim `grounded?` boolean → **accuracy imports this** (Response Quality §1 atom 1); the verifier's per-claim confidence also feeds the **conformal factuality** layer (§5), which is the formal sense in which "supported" becomes a *guaranteed* claim — the correctness of an output being, in Mohri & Hashimoto's framing, an uncertainty problem over the output's *entailment set*. **Tools:** Bedrock grounding / fairseq-NLI / RefChecker(-on-Bedrock) · Titan (pre-filter). **Confirm:** verifier; whether Contradiction is scored worse than Neutral (recommended — see §2).

---

## 2. hallucination — MAJOR · fabrication gates

Two distinct failures, verified differently — and the split matters because one is deterministic and gates.

1. **Unsupported-claim rate** `[T2]` — `= 1 − groundedness`, read directly from §1's triplet verdicts. **Contradiction** triplets (the source says the *opposite*) are the severe sub-case and are counted separately from **Neutral** (source silent), since asserting against the evidence is worse than asserting beyond it.
2. **Fabricated-citation / reference existence** `[T1]` — **this is fully deterministic and it gates.** Does the cited source *exist* and is its metadata correct: the cited id ∈ the retrieved set (set-membership); a URL resolves; a DOI validates against a registry; a reference's title/author/year are internally consistent. This is a **separate line from support** ("whether a cited reference *exists*… orthogonal to whether an existing passage *supports* the claim") — motivated by findings like **over 50 citation hallucinations in 300 ICLR 2026 submissions** and addressed by fabricated-reference auditors ([CiteAudit](https://arxiv.org/html/2605.06635v1), Yuan et al. 2026). Any fabricated citation → **gate FAIL**.
3. **Score** `[code]`: report the unsupported rate (Neutral vs Contradiction split); the fabrication check is a hard gate, not a rate.

**Tools:** groundedness verdicts (unsupported) · `urllib` / DOI-registry lookup / set-membership / `re` (existence + metadata). **Confirm:** DOI/URL registry in scope; Contradiction weighting.

---

## 3. source_quality — MAJOR · the bridge to world factuality

Whether the sources a claim relies on are *trustworthy* — what turns source-faithfulness into world-factuality, and **what accuracy imports as `source-adequate?`**. The credibility-signals literature ([survey of textual credibility signals](https://arxiv.org/pdf/2410.21360), 2024; [media background-check store](https://arxiv.org/pdf/2607.02383), 2026) organizes trustworthiness into signal families — *source/domain reputation, author identifiability and credentials, recency, corroboration across independent sources, and support for the specific claim* — and shows most are **extractable deterministically** from the document and its metadata, with only the residual "is this source disinterested" needing judgment. That maps to the eight checks below (seven deterministic or NLI, one judged).

1. **Reachable?** `[T1]` — the source resolves / is retrievable (`urllib`).
2. **Dated & fresh?** `[T1]` — a date is present and within the point-in-time window (`re`/metadata; critical for finance — bind to the as-of date).
3. **Authored / identifiable?** `[T1]` — an author or issuing body is present.
4. **On reputable domain?** `[T1]` — the domain is on a **configurable reliability allow/deny-list** (MBFC-style media-reliability store, or your curated financial-source list) — a structural oracle, human-maintained, pinned.
5. **Corroborated?** `[T1]` — the claim is supported by **≥2 independent** sources (count over the grounded set; independence = distinct domains/authors).
6. **Supports the claim?** `[T2]` — the source actually entails the claim (reuse §1's NLI; a good source that doesn't support *this* claim is not adequate *for it*).
7. **Disinterested / credible?** `[T3-res]` — the source isn't self-serving or conflicted (the one genuinely semantic residue; judge, sampled).
8. **Score** `[code]`: `source_quality = mean(property booleans)` per source; **`source-adequate?` = score ≥ config threshold** (the boolean accuracy imports).

*Profile:* Nexa (trusted internal corpus) → properties ≈ satisfied by construction, `source-adequate` defaults true. RavenPack (open web) → all live. **Tools:** `urllib` · `re` · reliability list (config) · §1 NLI · judge (disinterest, sampled). **Confirm:** the reliability list source; adequacy threshold; corroboration independence rule.

---

## 4. source_attribution — MAJOR · ALCE citation recall/precision

Whether each claim is credited to the source that *actually supports it* — a **faithfulness gap distinct from factual correctness** ("an answer can be correct while its citations do not support its claims"). This is **what accuracy imports as `attributed?`**, and the standard is ALCE.

1. **Per-claim citation support** `[T2 directional NLI]` — does the *cited* chunk entail *this* claim, classified three-way as **Attributable / Contradictory / Extrapolatory** ([AttrScore](https://arxiv.org/pdf/2411.14199), Yue et al. 2023) — or four-way **Supported / Insufficient / Contradictory / Irrelevant** ([CAQA](https://arxiv.org/html/2605.06635v1), Hu et al. 2024) if you want finer diagnostics. Same verifier as §1, premise = the *cited* chunk (not any chunk). AttrScore's own model is HF-hosted → use Bedrock/fairseq-NLI behind the interface.
2. **ALCE recall + precision** `[code]` — [ALCE](https://arxiv.org/pdf/2606.23915) (Gao et al. 2023) defines the two: **citation recall** = the cited set supports the statement; **citation precision** = each individual citation is relevant (no padding with unsupporting cites). Compute both in code from the per-citation verdicts.
3. **Precision-favoring, and the no-citation case** — false positives (wrongly calling a claim attributed) are more harmful than false negatives in faithfulness judging, so bias the threshold toward precision; and score attribution **only on claims carrying ≥1 citation**, while a claim with *no* relevant source should *state so* rather than fabricate a cite ([Evidence-Based QA](https://arxiv.org/pdf/2402.08277), 2024).
4. **Calibrated uncertainty** — attribution is hard even for strong models (**[AttributionBench](https://arxiv.org/html/2605.06635v1)**: fine-tuned GPT-3.5 only ~80% macro-F1), so a single verdict is not trusted alone; the verifier's per-claim confidence feeds the **conformal factuality** layer (§5), which attaches a *distribution-free, finite-sample* confidence to each retained attribution rather than a bare boolean.
5. **Score** `[code]`: `source_attribution = citation-precision (recall reported alongside)`; **`attributed?` per claim = Attributable ∧ (conformal-confident)** → the boolean accuracy imports.

*Regulated-domain note:* citation faithfulness is a stated deployment prerequisite for legal/compliance AI, where post-hoc attribution "lacks structural guarantees" ([RegOps](https://arxiv.org/pdf/2605.29742), 2026) — the conformal wrapper is how you attach a guarantee. **Tools:** directional NLI (Bedrock/fairseq) · ALCE recall/precision in `code` · `MAPIE` (conformal). **Confirm:** three-way vs four-way; precision threshold; conformal coverage level.

---

## 5. Statistical guarantee — conformal factuality (the auditability capstone)

The NLI verdicts in §1 and §4 are individually strong but imperfect — **[AttributionBench](https://arxiv.org/html/2605.06635v1)** shows even a fine-tuned model reaches only ~80% macro-F1 on support judgments. For a certification harness that isn't enough on its own, so the category's verdict layer is wrapped in **conformal factuality** ([Mohri & Hashimoto, ICML 2024](https://proceedings.mlr.press/v235/mohri24a.html)) to attach a **distribution-free, finite-sample guarantee**: with confidence `1 − α`, a retained claim is supported.

1. **Score** — the verifier assigns each claim/triplet a confidence `p(claim | context)` `[T2]`.
2. **Calibrate** `[code]` — on a small **held-out, human-labeled** set of `n` claims, compute nonconformity scores and set the threshold to the `(1−α)`-quantile: `τ̂ = Quantile_{⌈(1−α)(n+1)⌉ / n}({νᵢ})` (split conformal; Vovk et al. 2005; [Angelopoulos & Bates 2023](https://arxiv.org/abs/2107.07511)).
3. **Retain & guarantee** `[code]` — keep claims with confidence `≥ τ̂`; the bound
   ```
   1 − α  ≤  P(retained claim is factual)  ≤  1 − α + 1/(n+1)
   ```
   holds **distribution-free under exchangeability** (the finite-sample coverage bound of split conformal prediction; [Angelopoulos & Bates 2023](https://arxiv.org/abs/2107.07511); Vovk et al. 2005; Lei et al. 2018).

   **Reading the bound.** `α` is *your chosen error budget* (you set it); `1 − α` is the target confidence; `n` is the size of the human-labeled calibration set. The inequality sandwiches the true factuality rate of retained claims:
   - **Left side — the guarantee floor (the important half):** retained claims are factual **at least `1 − α`** of the time, *whatever* the score distribution looks like. Set `α = 0.05` → **≥ 95% of retained claims are genuinely supported**, guaranteed.
   - **Right side — the ceiling:** the method doesn't silently over-deliver either; it overshoots the target by at most `1/(n+1)`, a slack that comes from having to pick a quantile among finitely many calibration points, and shrinks as `n` grows.

   *Worked numbers* (α = 0.05, target 95%): `n = 100` → slack ≈ 1/101 ≈ 0.0099, so the true rate is provably in **[95.0%, 95.99%]**; `n = 1000` → slack ≈ 0.001, band **[95.0%, 95.1%]**; as `n → ∞` the slack vanishes and you land essentially exactly at `1 − α`. So even a small calibration set delivers the floor immediately (hence "very few human-annotated samples"), and more data only *tightens* the band. This is what lets the harness state, defensibly: *"of the claims retained as source-supported, at least 95% are genuinely supported — a distribution-free guarantee calibrated on n human-verified examples."*

Mohri & Hashimoto's framing is that **the correctness of an output is an uncertainty-quantification problem whose uncertainty sets are the *entailment set* of the output**, so conformal prediction becomes a *back-off* that makes an output progressively less specific until the guarantee holds — and it **applies to any black-box LM and needs very few human-annotated samples**, which fits the Bedrock / no-model-internals setting exactly.

**Why it's the capstone.** It converts "the NLI said supported" (a point estimate) into "we certify, at confidence `1 − α`, that retained claims are supported" — a *statistical*, regulator-grade guarantee rather than a model's say-so. Sub-claim-level conformal factuality is validated across domains: **[Conformal-RAG](https://arxiv.org/pdf/2603.16817)** (Feng et al. 2025) filters unreliable sub-claims to guarantee factuality; **TRAQ** (Li et al. 2023) applies it at retriever *and* generator; **Conflare** (Rouzrokh et al. 2024) calibrates retrieval so contexts contain the true answer at user-specified confidence.

**Caveat — the guarantee is *marginal*, so calibrate per stratum.** The bound above holds *on average across all retained claims* — it does not by itself promise the same rate *within* every subgroup (it could be 97% on easy claims and 92% on hard ones while averaging 95%). This is the conditional-coverage gap: a single global threshold can **over-cover hard categories and under-cover easy ones** per [Adaptive Conformal Prediction](https://arxiv.org/html/2604.13991v1) (2026). So when the guarantee must hold *uniformly* — e.g. equally for every source-type or question-type, which a regulator may require — calibrate `α` separately per stratum rather than once globally.

**Determinism/audit:** calibration is deterministic given the held-out set + `α`; the guarantee is a computed statistic; `α` is a config knob; the calibration set is a pinned, human-labeled reference (like the τ-validation sets). **Tools:** `MAPIE`, or a direct split-conformal implementation (a few lines of `numpy`). **Confirm:** `α` (coverage target); calibration-set size + refresh cadence; per-stratum calibration.

---

## 6. Contracts — the interfaces accuracy imports

Named so Response Quality §1 resolves cleanly (no more stubbed imports):

- `GroundingProvider.entails(premise, hypothesis) -> {label ∈ {E,N,C}, raw_score}` — shared by groundedness (§1), attribution (§4, premise = cited chunk), and source_quality's "supports" (§3.6). One verifier, three premises.
- `SourceQualityProvider.adequate(source, claim) -> bool` — §3's `source_quality ≥ threshold`. Config default `source_adequate_default: true` for the Nexa profile.
- `AttributionProvider.attributed(claim, cited_chunk) -> {bool, conformal_confidence}` — §4.
- All three emit AtomRecords (Response Quality §0.5). accuracy's atoms 1/2/3 (`grounded`/`source-adequate`/`attributed`) are **exactly these three calls**.

## Development summary — how each tier is used

**T1** = deterministic code (existence, metadata, reachability, date, domain-list, corroboration count, all recall/precision math) — replays bit-for-bit. **T2** = fixed NLI three-way, thresholded in code. **T3-gen** = pinned triplet extraction. **T3** = judge, confined to source disinterest only.

| Dimension | Output | T1 — code | T2 — fixed NLI | T3 |
|---|---|---|---|---|
| groundedness | `supported/total` | similarity pre-filter | three-way entailment vs context | — |
| hallucination | unsupported rate + **fabrication gate** | citation existence/metadata (gates) | (reads §1 verdicts) | — |
| source_quality | `mean(properties)` → adequate bool | reachable/date/author/domain-list/corroboration | supports-claim | disinterest (sampled) |
| source_attribution | ALCE precision (+recall) → attributed bool | recall/precision math; conformal (MAPIE) | cited-source entailment (3/4-way) | — |

This category is **heavily T1/T2**: the one gate (fabrication) is pure `[T1]`, the judge appears only for source disinterest, and every score is a code computation over NLI booleans — with **conformal prediction adding calibrated, auditable confidence** to the hardest verdict (attribution). That is the determinism/auditability posture the category is designed around.

## Sources
Faithfulness / hallucination
- [RefChecker](https://aclanthology.org/2024.emnlp-main.395.pdf) (Hu et al., EMNLP 2024) — claim-triplet 3-way checking, +4–9 pts; [Factuality area summary](https://papers.lunadong.com/area/factuality) (Source-Faithfulness vs World-Factuality taxonomy)
- [FinReflectKG-HalluBench](https://arxiv.org/html/2603.20252v1) (2026) — RefChecker in financial QA
- MiniCheck (Tang et al. 2024), HHEM (Vectara), SelfCheckGPT (Manakul et al.) — *cite by name*

Attribution / citation
- [ALCE citation recall/precision](https://arxiv.org/pdf/2606.23915) (Gao et al. 2023, via attribution-transfer audit) · [AttrScore three-way + citation eval](https://arxiv.org/pdf/2411.14199) (Yue et al. 2023, via OpenScholar)
- [Attribution parsing — AttributionBench, CiteME, CiteEval, CiteGuard, CiteAudit, CAQA](https://arxiv.org/html/2605.06635v1) (2026) · [CiteGuard](https://arxiv.org/html/2510.17853v4) (2026) · [VeriCite](https://arxiv.org/html/2510.11394v1) (2025)
- [Precision-favoring + no-citation handling](https://arxiv.org/pdf/2402.08277) (2024) · [Conformal-prediction citation guarding](https://arxiv.org/html/2607.20527) (2026) · [Regulatory per-rule attribution](https://arxiv.org/pdf/2605.29742) (2026)
- AIS framework (Rashkin et al. 2023) — *cite by name*

Source quality / credibility
- [Credibility-signals survey](https://arxiv.org/pdf/2410.21360) (2024) · [Media background-check knowledge store](https://arxiv.org/pdf/2607.02383) (2026)

Statistical guarantee / conformal factuality
- [Conformal Factuality (Mohri & Hashimoto, ICML 2024)](https://proceedings.mlr.press/v235/mohri24a.html) — distribution-free correctness guarantee, black-box, few samples
- [Conformal prediction tutorial (Angelopoulos & Bates 2023)](https://arxiv.org/abs/2107.07511) · Vovk et al. (2005) — *cite by name*
- [Conformal-RAG sub-claim filtering](https://arxiv.org/pdf/2603.16817) (Feng et al. 2025) · [Adaptive Conformal Prediction (conditional coverage)](https://arxiv.org/html/2604.13991v1) (2026) · TRAQ (Li et al. 2023), Conflare (Rouzrokh et al. 2024) — *cite by name*
- `MAPIE` (Python conformal-prediction library) — *cite by name*

Tooling
- [Bedrock contextual grounding](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-components.html) · [fairseq RoBERTa-MNLI](https://pytorch.org/hub/pytorch_fairseq_roberta/) · [RefChecker](https://github.com/amazon-science/RefChecker) · `MAPIE` (conformal prediction) — *cite by name*
