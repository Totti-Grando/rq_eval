# rq_eval — Full-Capacity Migration Guide (mock → live on AWS)

This is the hand-over runbook for taking `rq_eval` from the offline `mock`
development mode to **full live capacity** on the AWS/Bedrock target machine. It
supersedes the shorter runbook in `GUIDE.md §7` and adds everything the
claim-graph / support-set series (G1–G9) introduced.

**What "full capacity" means here** — every provider live, native three-way NLI
(`fairseq`), the fabrication resolver live, **both Layer-2 flags on** (accuracy
DAG derivation-rescue + relevance support-tree) *after* the edge-recall harness
clears the bar, live metadata for source_quality, and the pinned oracles +
calibration set replaced with real, human-validated content. Going live is a
config flip; going *trustworthy* also needs the human data in §7.

The core promise holds throughout: **mock → live swaps providers with no change to
any formula**; every score still replays from the atom log; the AI still emits
only booleans/labels/text and code computes every number.

---

## 0. Prerequisites

- **Python 3.11** on the target (the spaCy / coreferee / fairseq wheels target 3.11).
- **AWS account with Amazon Bedrock access**, network egress, and credentials
  (SSO, instance role, profile, or keys).
- Bedrock **model access enabled** for your Claude model and Titan Text Embeddings V2.
- ~a few GB disk for `en_core_web_lg` + (optional) the torch/fairseq NLI model.

---

## 1. Install

```bash
git clone <this repo>            # or copy the rq_eval/ folder over
cd rq_eval
PYTHON=python3.11 ./install.sh
```

`install.sh` runs, in order:
1. `pip install -r requirements.txt` — the **offline core** (pydantic, PyYAML,
   **networkx** ← the shared `ClaimGraph`, which pulls numpy transitively).
2. `pip install -e .` — the package itself.
3. `pip install -r requirements-live.txt` — **boto3**, **spaCy**, **coreferee**.
4. `spacy download en_core_web_lg` — the model used for sentence segmentation,
   **dependency-parse claim decomposition** (§0.2), and coreference.
5. `coreferee install en` — coreference resolution for decontextualization.
6. **(optional but recommended at full capacity)** `torch` + `fairseq` — native
   three-way NLI. If the build fails, `install.sh` skips it and you fall back to
   `models.nli: bedrock` (see §3 for the trade-off).
7. Freezes the resolved versions to `requirements.lock`.

**Optional — graph visualization (§0.3 diagnostic):** the resolved-graph render
(`pipeline/graph_viz.py`) writes a JSON node-link artifact with **no extra deps**;
for the force-directed PNG, additionally `pip install matplotlib`. It is a lazy
import — absence is a silent no-op, never a run failure.

> networkx is a **runtime** dependency now (the one shared claim graph). It is
> pure-Python and offline-safe; it is already in `requirements.txt`, so both mock
> and live installs get it.

---

## 2. AWS one-time setup

1. **Enable Bedrock model access** for your Claude model (used by `ScoringJudge`,
   `ExplanationJudge`, and `GeneratorProvider`) and **Titan Text Embeddings V2**
   (`EmbeddingProvider`).
2. **Create a Guardrail** with the **contextual-grounding policy** enabled — it
   powers both the grounding NLI (`models.nli: bedrock`) *and* the Method-B
   relevance score. Note its **guardrail id + version**. *(You can skip this if
   you run `models.nli: fairseq` and `relevance.method: A` — see §3/§6.)*
3. Grant the role/profile: `bedrock:Converse`, `bedrock:InvokeModel`,
   `bedrock:ApplyGuardrail`.

```bash
cp .env.example .env      # fill AWS_PROFILE or keys (or rely on SSO / instance role)
```

---

## 3. Choose the NLI backend (the biggest full-capacity lever)

The single `GroundingProvider` is the workhorse: it builds the support set `S`
(groundedness §1), confirms the claim-graph edges (§0.3), and feeds the conformal
layer (§5). Its backend is chosen by `models.nli`:

| `models.nli` | Labels | Full-capacity fit |
|---|---|---|
| `fairseq` | native **E / N / C** (RoBERTa-large-MNLI) | **Recommended.** Real **Contradiction** labels ⇒ hallucination's N-vs-C split is meaningful, stranded-contradiction orphans are detectable, and edge confirmation is strongest. Local, no per-call Bedrock cost. |
| `bedrock` | **E / N only** (Guardrails grounding) | Works, but cannot emit C — hallucination's contradiction_rate stays 0 and contradiction-based routing is blunted. Fine for a first live bring-up. |
| `mock` | lexical | dev only — do **not** use live. |

**At full capacity, prefer `fairseq`.** If it wouldn't build during `install.sh`,
either fix the torch/fairseq wheels or accept the `bedrock` limitation above.

---

## 4. `config.yaml` — the full-capacity block

`config.yaml` is the **only** file you edit. Annotated full-capacity values:

```yaml
providers:
  mode: live                     # flip everything to the live providers

aws:
  region: <your-region>
  profile: <your-profile>        # or omit and use SSO / instance role

models:
  judge_id: <your Bedrock Claude model id>   # ScoringJudge + ExplanationJudge + Generator
  embed_id: amazon.titan-embed-text-v2:0
  guardrail_id: <from §2>        # needed only if nli: bedrock or relevance uses Guardrail
  guardrail_version: <e.g. 1>
  nli: fairseq                   # full capacity: native 3-way (see §3)

groundedness:
  groundedness_k: 3              # top-k chunks entailed per triplet to build S;
                                 # raise for higher recall at more NLI calls (§8)

graph:                           # §0.3 shared claim-graph edge detection
  edge_tau: 0.5                  # entails(AND parents, claim) >= this confirms an edge
  topical_min: 0.3              # candidate-parent gate; raise to prune, lower for recall
  numeric_tolerance: 0.0         # exact arithmetic provenance (profit = revenue - costs)

accuracy:
  numeric_tolerance: 0.0
  dag_rescue_enabled: true       # Layer 2 — TURN ON ONLY AFTER §5 (edge-recall gate)

relevance:
  method: B                      # B = Guardrail gate (needs the guardrail); A = reverse-Q + Titan
  tree_enabled: true             # Layer 2 — TURN ON ONLY AFTER §5 (edge-recall gate)
  # anchor/tree knobs: edge_tau lives under graph.*; anchor_alpha, max_hops, depth_decay tunable

source_quality:
  as_of_date: "<the point-in-time you evaluate against>"   # freshness binds to this
  live_metadata_fetch: false     # set true only if your retrieval layer drops date/author
                                 # metadata and you want a live fetch to recover them
  disinterest_sample_rate: 0.1   # >0 to actually sample the disinterest residual judge

hallucination:
  resolver: live                 # urllib HEAD / DOI existence check for the fabrication gate
  doi_registry_enabled: true     # if you cite DOIs

conformal:
  calibration_path: config/calibration/<your-real-labeled-set>.jsonl   # see §7
```

Everything else (thresholds, bands, seeds, `pins.*`) has a sane default and is tunable.

---

## 5. GATE — validate edge detection BEFORE enabling Layer 2

The two Layer-2 features (accuracy DAG-rescue, relevance tree) rest on
**edge-detection recall**, which the design is explicit is the weakest link. Ship
them **off** until you have measured that recall on your own data:

1. Assemble a small **human-linked** fixture set: claims + the gold
   `(parent_id, child_id)` support edges a person marked.
2. Run the harness (`src/rq_eval/validation/edge_recall.py`):
   ```python
   from rq_eval.validation.edge_recall import EdgeCase, EdgeRecallHarness
   from rq_eval.pipeline.edge_detection import EdgeDetector
   from rq_eval.graders.t1 import T1Tools
   from rq_eval.config import load_config
   from rq_eval.providers.factory import ProviderFactory

   cfg = load_config()
   det = EdgeDetector(ProviderFactory(cfg).build().grounding, T1Tools(),
                      cfg.graph.edge_tau, cfg.graph.topical_min, cfg.graph.numeric_tolerance)
   report = EdgeRecallHarness(det).measure([EdgeCase(claims, gold_edges), ...])
   print(report.recall, report.precision)
   ```
3. **Only if recall clears your bar** (report it as the system's honest error bar),
   set `accuracy.dag_rescue_enabled: true` and/or `relevance.tree_enabled: true`.
   If it doesn't, leave the flags off — the **protected cores** (accuracy Layer 1
   axiom-truth, relevance direct on-topic/on-ask) fully score the system and
   degrade gracefully, never mis-scoring on weak edges.

With the flags **off**, the shared graph is still built (typed nodes) but stays
inert; with them on, edges are detected and the two projections read them.

---

## 6. Verify BEFORE any real evaluation

```bash
python smoke_test.py     # probes judge, explanation, generator, embedding,
                         # grounding, relevance, nlp, resolver, consistency —
                         # ALL must PASS (these are real AWS/model calls)
python -m rq_eval.runner # runs the planted fixtures end-to-end; eyeball the report
pytest -q                # optional: full suite in live mode (costs tokens)
```

The `audit/replay.py::ReplayVerifier` recomputes every score from the atom log
with **no model call** and asserts equality — run it (it's exercised by the
suite) to confirm determinism holds on the live path. Do not run production
evaluations until `smoke_test.py` is all-green.

---

## 7. Human / domain data required for trustworthy scores

Going live is a config flip; **trustworthy scores also need real content in the
pinned oracles + calibration.** These ship with starter/synthetic data — replace
them, then **bump the matching `pins.*_version`** (the reproducibility fence):

| File / knob | Ships with | Must become | Drives |
|---|---|---|---|
| `config/calibration/calibration-v1.jsonl` | synthetic toy pairs | **human-labeled** `{claim, context, label, stratum}` | the conformal guarantee (§5); `min_calibration_n` gates it |
| `config/reliability_list.yaml` | a few example domains | curated allow/deny domain list (MBFC-style) | source_quality "reputable domain" |
| `config/coi_denylist.yaml` | starter entries | real conflict-of-interest domains/authors | source_quality disinterest (COI rule) |
| `config/requirement_templates.yaml` | drivers/comparison/default | per-question-type facet oracle | completeness `templated` coverage |
| `config/question_archetypes.yaml` | ~8 generic shapes | tuned/added shapes if needed | completeness `archetype` mode |
| `config/completeness_recall_sample.jsonl` | a few rows | real "should-contain" sample | completeness `recall_miss_rate` error bar |
| `config/task_templates.yaml` | fix/explain/compare/… | domain outcomes + verifier tags | task_success routing/scoring |
| `config/prompts/claim-extractor-v1.json` (+ inline gen prompts) | minimal instructions | production prompts | **live model quality depends on these** (mock ignores them) |

Also set `completeness.reference_mode` to `templated` for question-types where you
have invested in a real checklist (strongest guarantee), `archetype` for the
middle tier, or leave `generated` (default, open-domain).

---

## 8. What each live backend replaces (updated for G1–G9)

| Interface / step | Mock (dev) | Live (full capacity) | Selected by |
|---|---|---|---|
| `ScoringJudge` (5 residuals only) | seeded-hash `[[tag]]` verdicts | Bedrock Claude Converse (reference-grounded YES/NO) | `providers.mode: live` |
| `ExplanationJudge` (read-only) | templated stub | Bedrock Claude run summary | `providers.mode: live` |
| `GeneratorProvider` (reference build) | `[[tag]]` splitters | Bedrock Claude text gen | `providers.mode: live` |
| `EmbeddingProvider` | hashed token vectors | Titan Text Embeddings V2 | `providers.mode: live` |
| `GroundingProvider` (the one NLI) | token-overlap + negation | **fairseq E/N/C** or Bedrock Guardrails E/N | `models.nli` |
| `RelevanceProvider` | token Jaccard | Bedrock Guardrails relevance | `providers.mode: live` |
| `NlpProvider` — segmentation, **dependency-parse decomposition (§0.2)**, coref | regex + clause splitter | **spaCy `en_core_web_lg` + coreferee** (ClausIE/PredPatt-style parse) | `providers.mode: live` |
| `ResolverProvider` (fabrication gate) | fabricated-marker check | urllib HEAD + optional DOI registry | `hallucination.resolver: live` |
| `ConsistencyProvider` (relevance routing) | stub (`edge_sound → True`) | still a stub — the Reasoning category isn't built; swaps in later with no relevance change | forward-declared |

**What gets *materially better* live and matters most now:**
- **Semantic support set `S`** (groundedness §1) → drives source_quality supports/
  corroboration and attribution `C∩S` — the mock can't see that "Los Blancos lifted
  the trophy" supports "Real Madrid won".
- **Claim-graph edges** (§0.3) → the mock's lexical NLI can't synthesize the
  premise→conclusion edges the tree/DAG need; live NLI (fairseq) is what makes
  Layer 2 worth enabling.
- **Deterministic extraction** (§0.2) → live spaCy dependency parse produces real
  clause/predicate-argument decomposition vs the mock's conjunction splitter.

---

## 9. Known live limitations & knobs (be honest about these)

- **Indexical binding uses a regex filler finder** (`T1Tools.find_filler`), not
  full spaCy NER, on both mock and live. Unbindable indexicals are flagged
  `context-incomplete` and routed out of grounding (reported, not guessed).
  Upgrading to spaCy-NER binding is a live-only enhancement behind the same node
  typing — no formula change.
- **Bedrock Guardrails NLI has no Contradiction label** — use `fairseq` if the
  N-vs-C split or contradiction routing matters (§3).
- **The `ConsistencyProvider` is still a stub** — relevance never penalizes on
  reasoning soundness; stranded contradictions are routed but downstream scoring
  by a Reasoning dimension isn't built yet.
- **Execution sandbox** (task_success `executable` outcomes) is still a heuristic
  behind `task_success.execution_sandbox` — do not enable without a real sandbox
  (it would run model-produced code).
- **Cost / latency** scales with model calls: groundedness runs
  `groundedness_k` NLI calls **per triplet**; edge detection (when a Layer-2 flag
  is on) is roughly **O(n²)** entailment calls over the answer's claims. Tune
  `groundedness_k`, `graph.topical_min` (prunes candidate parents), and
  `disinterest_sample_rate` to trade recall for cost. `fairseq` NLI is local
  (no per-call Bedrock charge); Bedrock NLI/judge/generator/embeddings are billed.

---

## 10. One-page checklist

- [ ] `python3.11 ./install.sh` (fairseq installed for full-capacity NLI)
- [ ] Bedrock model access + Guardrail created; IAM allows Converse/InvokeModel/ApplyGuardrail
- [ ] `.env` filled (or SSO/instance role)
- [ ] `config.yaml`: `providers.mode: live`, `models.*`, `models.nli: fairseq`, `hallucination.resolver: live`, `source_quality.as_of_date`
- [ ] Real `config/calibration/*.jsonl` + oracle files in place; `pins.*_version` bumped
- [ ] **Edge-recall harness run**; Layer-2 flags flipped on **only** if recall clears the bar
- [ ] `python smoke_test.py` all-green
- [ ] `python -m rq_eval.runner` fixtures look right; ReplayVerifier passes
- [ ] (optional) `pip install matplotlib` for the graph-viz PNG
