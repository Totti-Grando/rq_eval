# Calibration datasets — measuring error against existing human labels

The error-tracking layer needs a measured error rate per decision-type, and that needs human-labeled ground truth. The good news from the research: **most of your decision-types map onto tasks that already have human-annotated benchmark datasets.** You calibrate by running your verifier over the benchmark and counting agreement — no bespoke labeling for the common cases. Below, each decision-type → the dataset(s) that calibrate it, what's labeled, and how solid the match is. Bespoke labeling shrinks to a small residue.

## Groundedness / hallucination (the biggest, best-covered)

**LLM-AggreFact** — the anchor. A collection of **11 human-annotated faithfulness datasets** for "is this claim supported by this evidence document," spanning diverse domains with **real LLM hallucinations** ([HalluGuard](https://arxiv.org/pdf/2510.00880); [FaithLens](https://arxiv.org/pdf/2512.20182)). Directly your groundedness decision: (claim, evidence) → supported/not. MiniCheck-7B and Qwen3-32B sit ~77% balanced-accuracy on it — so it *also* tells you the ceiling you're calibrating against. Includes **RAGTruth** (word-level human hallucination-span annotations in the RAG setting — summarization, QA, data-to-text, 6 LLMs) and **WiCE, REVEAL, ClaimVerify, ExpertQA, LFQA**.

**TRUE** ([Granite Guardian](https://arxiv.org/pdf/2412.07724)) — **100K+ annotated examples** for factual consistency across NLP tasks; the standard groundedness meta-benchmark (FRANK, SummEval, MNBM, and others).

**SummEdits** ([PrefixNLI](https://arxiv.org/pdf/2511.01359)) — human factual-consistency verdicts on LLM-edited summaries; **RAGTruth + SummEdits** together give clean supported/unsupported labels with error spans.

**A cleaned version exists** — Seo et al. 2025 found 9.1% ambiguous / 6.6% mislabeled in the raw benchmarks and released a corrected LLM-AggreFact + HoVer ([FaithLens](https://arxiv.org/pdf/2512.20182)). Use the cleaned split — label noise otherwise floors your measured error.

→ **Groundedness calibration is essentially solved off-the-shelf.** Run your `GroundingProvider` over cleaned LLM-AggreFact, count agreement, conformal-wrap. No hand-labeling.

## Domain match — you have a FINANCIAL grounding benchmark

**FinReflectKG-HalluBench** ([2026](https://arxiv.org/html/2603.20252v1)) — a **GraphRAG hallucination benchmark for financial QA**, labeled grounded-vs-hallucinated, and it *benchmarks the exact stack you're using*: a DeBERTa-v3 NLI model (premise=context, hypothesis=question+answer, 3-class E/N/C thresholded to binary), the Vectara HHEM factual-consistency scorer, a fine-tuned Lynx-8B judge, and LettuceDetect span detection. This is the closest-to-Nexa calibration set available — **use it to estimate error on financial-domain claims specifically**, since general-domain error rates won't transfer perfectly to financial language.

## Reasoning-chain / edge validity / step verification (the graph layer — better covered than expected)

**EntailmentBank** ([Dalvi et al. 2021](https://aclanthology.org/2021.emnlp-main.585.pdf)) — **1,840 expert-annotated entailment trees**, avg 7.6 nodes / 3.2 steps, **5,881 annotated entailment steps**. Each tree is exactly your structure: leaf facts (axioms) → intermediate conclusions → hypothesis (root), with **which premises feed each step explicitly annotated**. This calibrates *two* of your hardest decisions: **edge detection** (does the annotated tree agree with your backward-BFS on which premises feed a node) and **step validity** (is a combination a valid deduction).

**STREET** ([2023](https://arxiv.org/pdf/2302.06729)) — multi-task structured reasoning; humans annotated **which premises were used in each entailment step**, including the AR-LSAT logical-reasoning task. Includes EntailmentBank. Directly calibrates edge/premise-attribution.

**PRM800k** ([reasoning-trace survey](https://aclanthology.org/2025.findings-emnlp.94.pdf)) — **crowdsourced step-by-step validity labels** (positive/negative/neutral) on reasoning steps. Calibrates local-validity.

**RDTE (Recognizing Decompositional Textual Entailment)** ([2024](https://arxiv.org/pdf/2402.14798)) — **1000+ expert annotations** built on the informal-logic **"Relevance, Acceptability, Sufficiency"** criteria — i.e. it labels exactly the *sufficiency/completeness* of a decomposition step (is the premise set sufficient to entail the conclusion). This is the closest thing to a **node-completeness / minimal-complete-premise** calibration set, and it's the criterion we adopted.

**LogiQA 2.0** ([improved dataset](https://frcchang.github.io/pub/An%20Improved%20Dataset%20for%20Logical%20Reasoning%20in%20Natural%20Language%20Understanding.pdf)) — human-labeled by **reasoning type**: categorical, sufficient-conditional, necessary-conditional, **disjunctive, conjunctive**. This calibrates the **AND/OR structure detection** — the sufficiency-condition typing we needed for load-bearing-parent identification.

→ **The graph layer's error rates are measurable against existing expert-annotated tree/step data.** EntailmentBank for edges+steps, RDTE for sufficiency/completeness, LogiQA 2.0 for AND/OR typing. This is the biggest surprise — the "hardest to calibrate" layer has real data.

## Attribution / citation

**Fact-Level Attribution** ([2026](https://arxiv.org/pdf/2602.11509)) — human-annotated, **recall via combined sources / precision via individual sources** (exactly the ALCE set-operations over the support set), F1/BAcc metrics, 917 test + 129 val human-annotated examples. Calibrates attribution directly. (Plus AttributionBench from the earlier evidence-doc research.)

## Numeric / arithmetic edges

**GSM8K / AQUA-RAT** (via STREET) — arithmetic reasoning chains. But note: **arithmetic fills are near-deterministic** (provable by number-provenance), so they need only a tiny confirmation set, not a full calibration corpus — error ≈ 0 by construction.

## Frame-semantic completeness (the residue that needs the most bespoke work)

**FrameNet** (3,353-sentence / 34k-frame-element annotated split; [FRASE](https://arxiv.org/pdf/2503.22144)) calibrates **frame-disambiguation + role-filling** (does your SRL pick the right frame and fill the right roles). But **whether a filled role-schema equals true logical completeness** — the direct-or-derived fill call — is the one decision with no perfect off-the-shelf set; **RDTE's sufficiency labels are the closest proxy**, and a small bespoke financial-claim completeness set is the residue to hand-label.

## The staging this implies

| Decision-type | Dataset | Bespoke labeling? |
|---|---|---|
| Groundedness | LLM-AggreFact (cleaned), TRUE, RAGTruth | **None** — off-the-shelf |
| Financial groundedness | FinReflectKG-HalluBench | **None** — domain match |
| Edge detection / step validity | EntailmentBank, STREET, PRM800k | **None** — off-the-shelf |
| Sufficiency / node-completeness | RDTE | Minimal — RDTE proxy + small financial set |
| AND/OR structure | LogiQA 2.0 | **None** |
| Attribution | Fact-Level Attribution, AttributionBench | **None** |
| Arithmetic fill | (provable) | Tiny confirmation set only |
| Frame disambiguation / role fill | FrameNet | **None** for parsing; small set for completeness-equivalence |

So bespoke human labeling collapses to: **a small financial-claim completeness/sufficiency set** (RDTE covers the general case), and **confirmation samples** for the near-deterministic checks. Everything else calibrates against existing human-annotated benchmarks.

## Honest caveats
- **Domain transfer:** most sets are science (EntailmentBank), news/summarization (AggreFact), or general logic (LogiQA). Financial-language error may differ — which is why **FinReflectKG-HalluBench matters** and why a *small* financial calibration set is still worth building for the reasoning layer even though the general sets bound the ballpark.
- **Label noise is real:** use cleaned splits (Seo et al.); raw benchmark noise (~15% combined ambiguous+mislabeled on some) otherwise sets a false error floor.
- **These calibrate the verifier, not your exact pipeline end-to-end** — a benchmark tells you `GroundingProvider`'s error on *its* claims; your decomposition + your claims differ. So benchmark calibration gives a **strong prior / ballpark**, and the review-loop's own labels refine it to your actual distribution over time. Benchmarks bootstrap; the review loop tunes.
- **Meta-evaluation ceilings double as targets:** the same datasets tell you the SOTA ceiling (MiniCheck ~77% BAcc on AggreFact), so you calibrate *and* know how close to the achievable frontier you are.

## Net
You do **not** need to hand-label most decision-types from scratch. Groundedness, edge/step validity, AND/OR typing, and attribution all calibrate against existing expert-annotated benchmarks (LLM-AggreFact, EntailmentBank, STREET, RDTE, LogiQA 2.0, Fact-Level Attribution), plus a **financial-domain** grounding set (FinReflectKG-HalluBench). Bespoke labeling shrinks to a small financial completeness/sufficiency set and tiny confirmation samples — and the review loop refines the benchmark-derived priors to your real distribution.

## Sources
- Groundedness/hallucination: [LLM-AggreFact via HalluGuard](https://arxiv.org/pdf/2510.00880) · [FaithLens — cleaned AggreFact + HoVer](https://arxiv.org/pdf/2512.20182) · [TRUE via Granite Guardian](https://arxiv.org/pdf/2412.07724) · [RAGTruth + SummEdits via PrefixNLI](https://arxiv.org/pdf/2511.01359) · [FaithJudge/FaithBench leaderboard](https://aclanthology.org/2025.emnlp-industry.54.pdf)
- Financial: [FinReflectKG-HalluBench](https://arxiv.org/html/2603.20252v1)
- Reasoning chains/steps: [EntailmentBank](https://aclanthology.org/2021.emnlp-main.585.pdf) · [STREET](https://arxiv.org/pdf/2302.06729) · [reasoning-trace evaluation survey (PRM800k etc.)](https://aclanthology.org/2025.findings-emnlp.94.pdf) · [RDTE — relevance/acceptability/sufficiency](https://arxiv.org/pdf/2402.14798) · [LogiQA 2.0 — AND/OR reasoning types](https://frcchang.github.io/pub/An%20Improved%20Dataset%20for%20Logical%20Reasoning%20in%20Natural%20Language%20Understanding.pdf)
- Attribution: [Fact-Level Attribution — recall/precision human set](https://arxiv.org/pdf/2602.11509)
- Frame/completeness: [FRASE/FrameNet](https://arxiv.org/pdf/2503.22144)
