# rq_eval — Detailed Design Overview: tools & exact scoring

What the spec docs leave implicit: **which library/tool implements each step**
(mock offline vs. live on Bedrock) and **the exact math** each score uses. Read
alongside `NAVIGATION.md` (where each lives) and the specs (`../*-design.md`).

Tiers: **T1** pure code · **T2** fixed model, thresholded in code · **T3g**
pinned generation · **T3** judge · **oracle** human-maintained config.

---

## Part 1 — Tool / library inventory (what's actually used)

| Package | Role in the system | Where | Status |
|---|---|---|---|
| **pydantic** | typed contracts + config validation (`Config`, `AtomRecord`, `Claim`, …) | everywhere | used |
| **PyYAML** | reads `config.yaml` + the YAML oracles (only via `config.py`) | `config.py` | used |
| `hashlib` (stdlib) | content-hash ids (atoms/claims/triplets); mock embeddings; mock seeded bit | contracts, mock | used |
| `re` (stdlib) | tokenization, sentence/clause split, citation/number parsing | mock, T1 tools | used |
| `math` (stdlib) | cosine, Wilson CI, conformal quantile | scoring, method_a | used |
| `json` (stdlib) | JSONL atom store, calibration set, prompt files | audit, pipeline | used |
| `sqlite3` (stdlib) | optional atom store backend | `audit/sqlite_atom_store.py` | used |
| `urllib` (stdlib) | live URL/DOI reachability (fabrication + source reachable) | `live/resolver.py` | used (live) |
| **boto3** | Bedrock **Converse** (`ScoringJudge` + read-only `ExplanationJudge` + generator), Titan **InvokeModel** (embeddings), **ApplyGuardrail** (grounding+relevance) | `providers/live/` | live only (lazy) |
| **spaCy** `en_core_web_lg` | live sentence segmentation | `live/nlp.py` | live only (lazy) |
| **coreferee** | live coreference resolution (decontextualization) | `live/nlp.py` | live only (lazy) |
| **torch + fairseq** (RoBERTa-large-MNLI) | optional native 3-way NLI (real Contradiction) | `live/grounding_fairseq.py` | live, optional (lazy) |
| pytest, hypothesis, mypy, ruff | tests + typing + lint | `tests/`, CI | dev |
| ~~numpy, scipy~~ | were in the design's tool list | — | **pruned in R6** — all statistics are stdlib `math` (see the note in `requirements.txt`) |
| ~~ragas~~ | was in the design's live tool list | — | **pruned in R6** — Method A reimplements the RAGAS *method* on Bedrock+Titan (note in `requirements-live.txt`) |

> The three packages named in the design's tool lists were removed from the
> requirements files during R6; the implementation uses stdlib `math` and a
> direct Method-A implementation instead.

### The one verifier, three premises (design §6)
`GroundingProvider.entails(premise, hypothesis) -> {label∈E/N/C, raw_score}` is
the single NLI used by groundedness (premise=context span), source_attribution
(premise=cited chunk), and source_quality's "supports" (premise=source). Backends:

| Backend | Tool | label rule | `models.nli` |
|---|---|---|---|
| mock | token coverage + negation heuristic | `cov=overlap(hyp,prem)`; `cov≥contra_tau ∧ negation-mismatch → C`; `cov≥entail_tau → E`; else `N`; `raw=cov` | `mock` |
| Bedrock | Guardrails contextual-grounding | `score≥entail_tau → E` else `N` (no C) | `bedrock` |
| fairseq | RoBERTa-large-MNLI (torch.hub) | `argmax{C,N,E}`; `raw=P(entail)` | `fairseq` |

`supported = (label == E)` is the boolean every downstream check consumes.

---

## Part 2 — Shared scoring primitives

**Mock deterministic text** (`providers/mock/deterministic_text.py`) — the offline stand-ins:
- `tokens(t)` = lowercase `[a-z0-9]+` minus stopwords.
- `overlap(a,b) = |tokens(a) ∩ tokens(b)| / |tokens(a)|` (directional coverage).
- `jaccard(a,b) = |∩| / |∪|`.
- `embed(t)` = 64-dim vector; each token md5-hashed to an index, counted, then L2-normalized (cosine ≈ shared-token rate).

**Formula registry** (`scoring/formulas.py`) — every score is one of these pure functions over `AtomRecord`s (so it replays):

| formula_id | math |
|---|---|
| `mean` | `Σ verdict / n` |
| `weighted_mean` | `Σ verdict·w / Σ w` |
| `conjunction_weighted_mean` | group atoms by subject; `correct = AND(verdicts)`; `Σ correct·w / Σ w` (w per subject) |
| `relevance_capped_mean` | (legacy) `abstain_relevant→1.0`; else `base=mean(responsive)`; if `on_ask_answer=False` → `min(base, cap)` |
| `relevance_tree_capped_mean` | `abstain_relevant→1.0`; else `base=mean(claim_relevance.weight)` (code-graded: anchor/bg/stranded=1.0, in-tree depth d=`depth_decay**d`, off-topic=0.0); if `on_ask_answer=False` → `min(base, cap)` |
| `task_success_weighted` | `impossible→1.0`; else `Σ outcome·w / Σ w` |
| `achieved_ratio` | `impossible→1.0`; else `mean(outcome verdicts)` |
| `unsupported_rate` | `1 − Σ verdict / n` |

**Statistics** (`scoring/`):
- Wilson 95% CI (`wilson.py`), `z=1.96`:
  `center=(p̂+z²/2n)/(1+z²/n)`, `half=(z/(1+z²/n))·√(p̂(1−p̂)/n+z²/4n²)`, clamp `[0,1]`; `n=0→(0,1)`.
- Bands (`bands.py`): `score≥G→"G"; score≥A→"A"; else "R"` (default `G=0.90, A=0.75`).
- Off-ask cap (`aggregation.py`): `on_ask ? score : min(score, cap)`.
- Min-n abstention: `abstain ⟺ n < min_n`.

---

## Part 3 — Response Quality

### §0.2 shared claim extraction — `pipeline/` (deterministic parse-first)
Runs once, cached, consumed by accuracy/completeness/relevance. **No judge or
generator on the primary path**: segment `[T1]` → verifiable-vs-opinion lexical
filter (`T1Tools.is_verifiable`) `[T1]` → decompose into content-unit clauses
(`NlpProvider.parse_clauses`; abstractive-implied spans flagged, not generated)
`[T1]` → decontextualize (`resolve_coref` + structural self-contained check)
`[T1/T2]`. Optional pinned surface-realizer `[T2]` behind `extraction.realizer_enabled`
(off by default, droppable per the realizer-impact test). Pinned by `extractor_version`;
stability measured.

### §1 accuracy  — `Σ correct·w / Σ w`
Inputs: answer claims + retrieved context + citations. Per claim, four booleans (all **imported**), AND-ed, then weighted-mean.

| Step | Tier | Tool (mock → live) | Computation |
|---|---|---|---|
| grounded? | T2 | entails (overlap → Guardrail/fairseq) | imported from groundedness: claim's triplets all `E` |
| + numeric edge | T1 | `T1Tools` (`re`) | if claim has a number: `|na−nb| ≤ tolerance·max(|na|,|nb|)` vs source (NOT NLI) |
| source-adequate? | T1/T2/T3 | `SourceQualityProvider` | `source_quality ≥ adequacy_threshold` (§3) |
| attributed? | T2 | entails on cited chunk + conformal | `Attributable ∧ confidence ≥ threshold` (§4/§5) |
| responsive? | T1+T2 | relevance (NLI+lexical, no judge) | imported atom from §3 |
| unsourced residual | T3 | judge (Bedrock Claude) | when no context: truth-judge boolean |
| **score** | code | `conjunction_weighted_mean` | `claim_correct = AND(the four)`; `accuracy = Σ correct·w / Σ w`; `w` = importance weight (toggle) |
| CI / band | code | Wilson / BandMapper | Wilson over (correct claims, #claims) |

### §2 completeness — strict vital recall (two-tier, mode-based reference)
Inputs: question + sources. The Tier-1 reference is **mode-selected** and the mode
is stamped on the result (`assurance_mode`); a human recall-sample miss-rate is reported.

| Step | Tier | Tool | Computation |
|---|---|---|---|
| Tier-1 requirements (mode) | oracle / T3g | `ReferenceModeSelector` | `generated` (default, per-question `[T3-gen]`) · `archetype` (`config/question_archetypes.yaml` shapes) · `templated` (`config/requirement_templates.yaml`); mode → `assurance_mode` |
| recall error bar | code | `RecallSample` | `recall_miss_rate = |sampled should-contain facts not surfaced| / |sampled|` |
| Tier-2 units | T3g + T1 | generator + extractive | top-down (from requirement) + bottom-up (source sentences overlapping the requirement) |
| admissibility gate | T1+T2 | `T1Tools` + coref + double-NLI | atomic (`re` split) ∧ self-contained (coref) ∧ decidable (**double-NLI**: entails(answer,unit) vs entails(answer+sources,unit) agree; disagreement → reference-grounded residual judge) → freeze |
| dedupe | T2 | embeddings cosine | drop unit if `cos ≥ dedupe_tau` with a kept unit |
| assign (support) | T2 | entails (answer = premise, unit = hypothesis) | supported ⟺ `label == E` |
| **score** | code | `mean` over vital support atoms | `strict_vital_recall = |vital supported| / |vital|` |
| also reported | code | `two_level_scoring.py` | `requirement_coverage = |reqs with ≥1 supported unit| / |reqs|`; `weighted_recall = Σ recall(r)·w(r)/Σ w(r)`, `w=2` if vital & weighting else 1 |
| CI / abstain | code | Wilson / min-n | Wilson over (vital supported, vital total); abstain if vital total `< min_n` |

### §3 relevance — anchor-and-support tree + orphan resolution
Inputs: question + answer claims. Relevance is a support tree over the whole answer,
not a per-claim filter; the `responsive` atom is still exported to accuracy.

| Step | Tier | Tool | Computation |
|---|---|---|---|
| per-claim on-topic | T2 | relevance | `relevance(question, claim) ≥ relevance_tau` |
| per-claim on-ask | T1+T2 | NLI + lexical (no judge) | `on_ask = entails(claim, ask)==E ∨ key_term_overlap(question, claim) ≥ lexical_min_overlap` (DIVER-QA) |
| responsive atom (exported) | T2 | code | `on_topic ∧ on_ask` → imported by accuracy |
| edges | T1+T2 | markers + entails | `A→B ⟺ entails(A,B).raw ≥ edge_tau ∧ label≠C` (marker = candidate prior only) |
| anchors | T2+code | on-ask seed + centrality | seeds ∪ {in-degree ≥ `anchor_centrality_min`}; recall band via conformal (`anchor_alpha`) |
| tree | code | reachability | depth from anchor (fixpoint, `max_hops`); grade weight `depth_decay ** depth` |
| orphans | T1+T2 | on-topic + orphan→anchor NLI | off-topic (penalize) / stranded-veracity (kept, contradiction routed to `ConsistencyProvider`) / background (kept) |
| abstention | T3 | judge | proper decline to unanswerable → score `1.0` |
| **score** | code | `relevance_tree_capped_mean` | `mean(claim_relevance.weight)`, capped at `off_ask_cap` if answer-level on-ask is False |

### §4 task_success — verifier-routed `Σ achieved·w / Σ w`
Inputs: question + answer (+ artifacts/state).

| Step | Tier | Tool | Computation |
|---|---|---|---|
| infer objective | T3g | generator | intent text |
| classify + template | oracle | `config/task_templates.yaml` (keyword match) | task type → verifier-tagged outcomes + weights |
| decompose outcomes | T3g | generator | instantiate outcomes |
| route each outcome ↓ | — | `VerifierRouter` | one verifier per tag: |
| — artifact-presence | T1 | `re` | any required pattern in answer |
| — executable | T1 | heuristic (sandbox iface) | code signals ∧ run-claim present (real exec when a sandbox is wired) |
| — state | T1 | string/hash match | `expected == answer` or `expected ∈ answer` |
| — constraint | T1 | code | includes ∧ ¬excludes ∧ word-count bounds |
| — coverage | T2 | entails | `entails(answer, requirement) == E` |
| — grounded/responsive | import | entails/relevance | reuse §1/§3 checks |
| — adequacy | T3 | judge | the only judge call in task_success |
| **score** | code | `task_success_weighted` | `Σ achieved·w / Σ w`; well-scoped impossible → `1.0` |

---

## Part 4 — Evidence & Truthfulness

### §0 triplets — `pipeline/triplets.py`
**Parse-first** (§0): each claim clause is parsed into `subject | predicate | object`
by `T1Tools.parse_triplet` over `NlpProvider.parse_clauses` `[T1]`; the generator
(`[[triplets]]` → Bedrock live) is invoked **only for the residual** the parser
can't cleanly triple (nested/abstractive predicates) `[T3-gen]`. Carries claim id +
citation + source pointer. Pinned (`triplet_extractor_version`); stability = `|∩ ids| / |∪ ids|`.

### §1 groundedness — `|E| / |total triplets|`
| Step | Tier | Tool | Computation |
|---|---|---|---|
| similarity pre-filter | T1 | embeddings cosine | pick nearest context span per triplet (premise); **not** the score |
| three-way entailment | T2 | entails (overlap → Guardrail/fairseq) | per triplet → E/N/C |
| **score** | code | `mean` | `groundedness = |E-labeled triplets| / |total|` |
| export | code | — | per-claim `grounded = AND(triplet == E)`; per-triplet `raw_score` → conformal |

### §2 hallucination — unsupported rate + fabrication gate
| Step | Tier | Tool | Computation |
|---|---|---|---|
| unsupported rate | T2/code | reads §1 verdicts | `1 − groundedness`; split `neutral_rate=|N|/tot`, `contradiction_rate=|C|/tot` |
| fabrication gate | T1 | set-membership + `ResolverProvider` | `exists = (cited id ∈ retrieved set) ∨ resolve(url/doi)`; any `¬exists → gate FAIL` (band forced `R`) |
| resolver (mock→live) | T1 | marker-check → `urllib` HEAD / DOI registry | mock: not a fabricated-marker string; live: HTTP 200–399 / DOI |
| **score** | code | `unsupported_rate` | replays from the §1 triplet atoms |

### §3 source_quality — `mean(7 properties)` per source
Per source (internal-corpus chunks with no url/domain pass the metadata checks by construction):

| Property | Tier | Tool | Rule |
|---|---|---|---|
| reachable | T1 | resolver | internal→True; else `resolve(url)` |
| dated & fresh | T1 | `str`/`date` | `date` present ∧ `date ≤ as_of_date` |
| authored | T1 | code | `author` present |
| reputable domain | T1 | `reliability_list.yaml` | deny→False; allow→True; unknown (allow-list set)→False |
| corroborated | T1 | entails + count | `|distinct domain/author among sources that entail the claim| ≥ corroboration_min` |
| supports the claim | T2 | entails | `entails(source, claim) == E` |
| disinterested | T1 (rule) · T3 (residual) | `CoiRule` + sampled judge | `¬(denylisted ∨ affiliation_conflict)` where decisive; only the ambiguous remainder samples a judge at `disinterest_sample_rate` |
| **score / adequate** | code | `mean` | `source_quality = mean(properties)`; `source-adequate? = score ≥ adequacy_threshold` |

### §4 source_attribution — ALCE citation precision
| Step | Tier | Tool | Computation |
|---|---|---|---|
| per-citation support | T2 | entails on **cited** chunk | label → AttrScore 3-way (`labels: three`) or CAQA 4-way (`four`); Attributable ⟺ `E` |
| ALCE recall + precision | code | `alce.py` | `precision = |attributable citations| / |citations|`; `recall = |statements whose cite set supports them| / |cited statements|` |
| no-citation handling | code | — | claims without a citation excluded here (routed to accuracy's unsourced residual; no double-count) |
| **score / attributed** | code | `mean` + conformal | `source_attribution = citation precision`; `attributed? = Attributable ∧ (confidence ≥ conformal threshold)` |

### §5 conformal factuality — `scoring/conformal.py`
| Step | Tier | Tool | Computation |
|---|---|---|---|
| confidence | T2 | entails `raw_score` | per calibration example |
| calibrate | code | pure `math` | factual-only `νᵢ = 1 − confidenceᵢ`; `k = ⌈(1−α)(n+1)⌉` (clamp n); `τ̂ = k-th smallest ν`; `threshold = 1 − τ̂` |
| retain + guarantee | code | — | retain iff `confidence ≥ threshold`; band `[1−α, min(1, 1−α + 1/(n+1))]`; abstain if `n < min_calibration_n` |
| per-stratum | code | `ConformalStratifier` | same, grouped by stratum (config `per_stratum`); calibration set: `CalibrationStore` + `config/calibration/*.jsonl` |

---

## Part 5 — Determinism & replay (why the math lives in code)

Every model output is a boolean/label/text stamped into an `AtomRecord`; every
score above is a pure formula over those atoms. `audit/replay.py::ReplayVerifier`
recomputes each `DimensionResult.score` from its `atom_ids` + `formula_id` with
**no model call**: T1/code atoms replay bit-for-bit, T2 atoms replay from the
stamped label, T3 atoms replay from the logged verdict (model+version stamped so
drift is detectable). In `mode: mock` the T2/T3 tools are the lexical stand-ins
above; flipping to live swaps them for boto3/spaCy/fairseq with **no change to
any formula**.
