# rq_eval — Navigation & Migration Guide

How the codebase is laid out, what is implemented, and **exactly** what to do to
take it live on the AWS (Bedrock) work machine — including every "lexical" mock
default that must be replaced with a real model.

Companion docs: `ARCHITECTURE.md` (layer diagram + section→folder table),
`README.md` (quick start), the design specs one level up
(`../response-quality-design.md`, `../evidence-truthfulness-design.md`), and a
`README.md` inside every `src/` subfolder (with the formulas written out).

---

## 1. Mental model (read this first)

- **One rule everywhere:** *AI emits booleans/labels only; code computes every
  number.* Models produce yes/no verdicts, three-way NLI labels (E/N/C), or
  pinned reference **text** (claims, units, triplets, outcomes). Every score is
  arithmetic in `scoring/` over those booleans.
- **Every verdict is an `AtomRecord`** (append-only log) stamped with
  model+version. **Every score replays** from the logged atoms + a formula id
  with no model call (`audit/replay.py`).
- **One config file.** `config.yaml` is the *only* place with model ids,
  regions, thresholds, seeds, paths, versions. `config.py` is the only module
  that reads it (a test enforces this). Moving machines = edit `config.yaml`
  (+ `.env`), run `install.sh`, run `smoke_test.py`.
- **Offline-first.** In `providers.mode: mock` the whole thing runs with **no
  network** using deterministic lexical stand-ins. Going live is a config flip
  plus real credentials/data — no code changes.
- **Two categories, eight dimensions.** Response Quality = accuracy,
  completeness, relevance, task_success. Evidence & Truthfulness = groundedness,
  hallucination, source_quality, source_attribution. Accuracy *imports*
  grounded/source-adequate/attributed/responsive from the others.

---

## 2. Repo map (where everything lives)

```
rq_eval/
  config.yaml              # THE one config (all knobs)
  .env.example             # AWS secrets template -> copy to .env on target
  requirements.txt         # offline core (installs on 3.11 AND 3.14)
  requirements-live.txt    # live-only deps (target, Python 3.11)
  install.sh               # one-time target setup
  smoke_test.py            # probes each provider; run before evaluating
  ARCHITECTURE.md          # layer diagram + section->folder table (nav test)
  config/                  # data oracles (human-editable, pinned)
    prompts/claim-extractor-v1.json     # §0 extraction prompts
    requirement_templates.yaml          # completeness Tier-1 oracle
    task_templates.yaml                 # task_success verifier-typed outcomes
    reliability_list.yaml               # source_quality domain allow/deny
    calibration/calibration-v1.jsonl    # conformal calibration set (SYNTHETIC)
  src/rq_eval/
    config.py              # pydantic config schema; sole reader; load_config()/load_yaml()
    contracts.py           # Claim, Triplet, AtomRecord, DimensionResult, EvalInput, ContextChunk, CalibrationExample
    providers/             # every external dep behind an interface + mock + live
      base.py              #   the interfaces (Judge/Generator/Embedding/Grounding/Relevance/Nlp/Resolver/SourceQuality/Attribution)
      factory.py           #   ProviderFactory — the ONLY constructor (mock|live by config)
      model_stamp.py       #   (model, version) stamps for atoms
      mock/  live/         #   the two implementations of each interface
    pipeline/              # §0: claims (segment→verifiable→claimify→decontext→pin/stability) + triplets.py
    graders/               # tier adapters: T1Tools, JudgeGrader, GroundingGrader (3-way), RelevanceGrader
    scoring/               # pure math: formulas registry, Wilson CI, bands, off-ask cap, conformal
    audit/                 # atom stores (jsonl/sqlite), AtomLogger, ReplayVerifier, CalibrationStore
    dimensions/
      base.py              #   Dimension ABC: evaluate(EvalInput) -> DimensionResult
      responsiveness.py    #   §3->§1 export (relevance publishes responsive)
      relevance/  accuracy/  completeness/  task_success/     # Response Quality
      groundedness/  hallucination/  source_quality/  source_attribution/  # Evidence & Truthfulness
    runner.py              # Evaluator.evaluate() — orchestrates all 8 + conformal; evaluate() wrapper; CLI
    report.py              # human-readable run report
    fixtures.py            # planted Q/A/context cases (RQ + E&T)
  tests/                   # ~110 tests, all offline
```

**How to find the code for a spec step:** the folder mirrors the design section
(e.g. design §2 step 3 "unit admissibility gate" →
`dimensions/completeness/admissibility_gate.py`), and each class docstring cites
its `§`/step + tier. Open the folder's `README.md` for the formulas.

---

## 3. Non-negotiable invariants (don't break these)

1. **Booleans/labels only from models.** `JudgeProvider.binary -> bool`;
   `GroundingProvider.entails -> {E|N|C}`; generation returns text, never a
   number. Float→boolean thresholding happens in the **grader/dimension** layer
   from config, never in the provider. `scoring/` imports no model code (tested).
2. **One config reader.** Only `config.py` reads `config.yaml`/env/YAML
   (`load_yaml`). Grep test forbids magic values elsewhere.
3. **Every check is an atom; every score replays.** If you add a verdict, log an
   `AtomRecord`; if you add a score, register a pure `Formula` so it replays.
4. **Construction only via `ProviderFactory`.** Nothing instantiates a provider
   directly.
5. **Style:** one class per file, `< ~300` lines, `mypy --strict` + `ruff`
   clean, per-folder `README.md`, `ARCHITECTURE.md` table kept in sync (a nav
   test verifies it).

---

## 4. One evaluation, end-to-end (`runner.py` `Evaluator.evaluate`)

Order matters because of the imports:

```
pipeline.run(answer)         -> claims (+ §0 stability)
triplets = extract_all(claims)   (RefChecker S-P-O units)
conformal = calibrate(calibration set)     # global threshold + guarantee band

relevance      -> exports per-claim RESPONSIVE atom  (ResponsivenessExport)
groundedness   -> exports per-claim GROUNDED atom + triplet confidences (GroundednessExport)
accuracy       -> imports grounded + responsive; calls SourceQualityProvider (source-adequate)
                  and AttributionProvider (attributed, gated by the conformal threshold);
                  composes claim_correct = grounded ∧ source_adequate ∧ attributed ∧ responsive
completeness   -> two-tier nugget recall
task_success   -> verifier-routed outcomes
hallucination  -> unsupported rate (N/C split) + T1 fabrication gate   (reads GroundednessExport)
source_quality -> 7 property checks per source
source_attribution -> ALCE citation precision/recall; conformal band stamped
```

Result: `EvaluationResult{ results: {8 DimensionResults}, claims, stability,
store, conformal, atoms }`. The replay verifier can recompute every score from
`store` with no model call.

---

## 5. Implementation status — DONE

Everything in the two build orders (B1–B10 and E1–E9) is implemented, tested
(~110 tests), `mypy --strict` + `ruff` clean, and runs offline. Highlights:

- Config + provider portability layer; contracts + append-only atom log + replay
  verifier + formula registry.
- §0 pipeline (claims) and claim-triplets, both pinned + stability-measured.
- All 8 dimensions, with the three accuracy imports **wired to the real
  providers** (no stubs left for source_quality/attribution; grounded imported
  from groundedness).
- Three-way grounding contract (E/N/C) universal.
- Fabrication gate (T1), ALCE recall/precision, conformal factuality (marginal +
  per-stratum), calibration store.
- Runner (8 dims + conformal), report, fixtures, smoke test.

---

## 6. The lexical/mock defaults and their live replacements  ← the important part

In `providers.mode: mock` **all semantic judgment is lexical/deterministic** so
tests are reproducible offline. These are the stand-ins you replace by going
live. Each has a real implementation already written behind the same interface —
**flip config, no code changes.**

| Provider (interface) | Mock (now) — lexical | Live replacement | Selected by |
|---|---|---|---|
| `JudgeProvider` | seeded-hash / `[[tag]]` token-overlap verdicts | **Bedrock Claude** (Converse, strict YES/NO) — `live/judge.py` | `providers.mode: live` |
| `GeneratorProvider` | `[[triplets]]/[[sentences]]/[[repeat]]` parse splitters | **Bedrock Claude** text gen — `live/generator.py` | `providers.mode: live` |
| `EmbeddingProvider` | hashed bag-of-tokens vectors | **Titan Text Embeddings v2** — `live/embedding.py` | `providers.mode: live` + `models.embed_id` |
| `GroundingProvider` (E/N/C) | token coverage ≥ `entail_tau` → E; negation-mismatch → C | **Bedrock Guardrails** contextual grounding (E/N) *or* **fairseq RoBERTa-MNLI** (native E/N/C) | `models.nli: bedrock \| fairseq` |
| `RelevanceProvider` | token Jaccard | **Bedrock Guardrails** relevance score — `live/relevance_guardrail.py` | `providers.mode: live` |
| `NlpProvider` | regex sentence split + leading-pronoun coref | **spaCy `en_core_web_lg` + coreferee** — `live/nlp.py` | `providers.mode: live` |
| `ResolverProvider` | "exists unless the string looks fabricated" | **urllib HEAD + optional DOI registry** — `live/resolver.py` | `hallucination.resolver: live` |

**What this means concretely:** offline, "grounded", "relevant", "responsive",
"attributed", "supports", triplet extraction, unit drafting, task-outcome
judging, and coreference are all **token-overlap / parse heuristics**. They are
*structurally* correct (the scoring, composition, gates, and replay are the
real thing) but **semantically blind** — they can't tell that "Real Madrid won"
is entailed by "Los Blancos lifted the trophy". That is exactly what the live
models fix. Nothing else in the pipeline changes.

Two provider selectors are independent of `providers.mode`:
- `models.nli` picks the grounding/NLI backend (can stay `mock` for a partial-live run).
- `hallucination.resolver` picks the URL/DOI resolver.

For a **fully live** run set: `providers.mode: live`, `models.nli: bedrock`
(recommended) or `fairseq`, and `hallucination.resolver: live`.

---

## 7. Migration runbook — taking it live on the AWS machine

Target machine assumptions: **Python 3.11** (spaCy/coreferee/fairseq wheels),
AWS credentials with Bedrock access, network egress.

### 7.1 Get the code + install
```bash
git clone <this repo>            # or copy the rq_eval/ folder over
cd rq_eval
PYTHON=python3.11 ./install.sh
```
`install.sh` does, in order: `pip install -r requirements.txt` → `pip install -e .`
→ `pip install -r requirements-live.txt` → `spacy download en_core_web_lg` →
`coreferee install en` → (optional) `torch`+`fairseq` → freezes `requirements.lock`.
If fairseq fails to build, **skip it** and use `models.nli: bedrock`.

### 7.2 AWS one-time setup (console/CLI)
1. **Enable Bedrock model access** for your Claude model (judge/generator) and
   **Titan Text Embeddings V2**.
2. **Create a Guardrail** with the **contextual-grounding policy** enabled (this
   powers both grounding *and* relevance). Note its **guardrail id + version**.
3. Ensure the role/profile can call `bedrock:Converse`, `bedrock:InvokeModel`,
   and `bedrock:ApplyGuardrail`.

### 7.3 Credentials
```bash
cp .env.example .env      # fill AWS_PROFILE or keys (or rely on SSO/instance role)
```

### 7.4 Flip `config.yaml` (the only file you edit)
```yaml
providers: { mode: live }
aws:      { region: <your-region>, profile: <your-profile> }
models:
  judge_id: <your Bedrock Claude model id>
  embed_id: amazon.titan-embed-text-v2:0
  guardrail_id: <from step 7.2>
  guardrail_version: <e.g. 1 or DRAFT>
  nli: bedrock            # or fairseq if you installed it
hallucination: { resolver: live, doi_registry_enabled: true|false }
source_quality: { as_of_date: "<the point-in-time you evaluate against>" }
conformal: { calibration_path: config/calibration/<your real set>.jsonl }
```
Everything else (thresholds, bands, seeds, weighting toggles) is tunable but has
a sane default.

### 7.5 Verify BEFORE evaluating
```bash
python smoke_test.py     # probes judge, generator, embedding, grounding,
                         # relevance, nlp, resolver — must all PASS (real AWS calls)
pytest -q                # optional: full suite in live mode (costs tokens)
python -m rq_eval.runner # runs the fixtures end-to-end; eyeball the report
```
Do not run real evaluations until `smoke_test.py` is all-green.

---

## 8. Human / domain work required before certification

Going live is a config flip; **producing trustworthy scores also needs real
data in the pinned oracles.** These ship with starter/synthetic content:

| File | Ships with | Must become | Why |
|---|---|---|---|
| `config/calibration/calibration-v1.jsonl` | **synthetic** finance/sports toy pairs | a **human-labeled** `{claim, context, label, stratum}` set | the conformal guarantee is only meaningful over real labels; `min_calibration_n` gates it |
| `config/reliability_list.yaml` | a few example domains | your curated financial-source allow/deny list (MBFC-style) | drives source_quality's "reputable domain" check |
| `config/requirement_templates.yaml` | drivers/comparison/default | your per-question-type facet oracle | completeness Tier-1 coverage guarantee rests on this fixed scaffold |
| `config/task_templates.yaml` | fix/explain/compare/… with cue words | domain-tuned outcomes + verifier tags | task_success routing/scoring |
| `config/prompts/claim-extractor-v1.json` + inline triplet/reverse-question prompts | minimal instructions | production-grade prompts | mock ignores them; **live model quality depends on them** — author + review, then bump the `pins.*_version` |
| `source_quality.as_of_date` | `2026-07-31` | the real point-in-time window | freshness check binds to it |
| `source_quality.disinterest_sample_rate` | `0.0` (judge never runs) | e.g. `0.1` | to actually sample the disinterest judge on target |

When you change any pinned reference, **bump its `pins.*_version`** — that is the
reproducibility fence.

---

## 9. Known deviations, stubs, and optional pieces (be aware)

- **Method A (RAGAS) is reimplemented directly**, not via the `ragas` library:
  `dimensions/relevance/method_a.py` generates reverse-questions (GeneratorProvider)
  + Titan cosine. `ragas` in `requirements-live.txt` is therefore **unused** —
  remove it or wire the library if you prefer. Relevance defaults to Method B
  (guardrail), so Method A only runs if `relevance.method: A|both`.
- **`InferenceValidityStub`** (`dimensions/accuracy/stubs.py`) still returns
  `True` — it belongs to the out-of-scope `logical_consistency` category. Accuracy's
  "inferred" flag is audit-only; wire the real module when that category is built.
- **Execution sandbox is an interface, not an implementation.** task_success
  `executable` outcomes use a text heuristic (`execution_sandbox: false`). To get
  the design's "run the code" determinism, implement `ExecutionSandbox`
  (`verifiers/execution.py`) with a real sandbox and set the toggle. **Do not**
  enable it without a proper sandbox — it would run model-produced code.
- **fairseq is optional.** Bedrock Guardrails can't distinguish Neutral from
  Contradiction (it returns E/N only); use `models.nli: fairseq` if you need
  native three-way (real Contradiction labels for hallucination's N/C split).
- **4-way attribution / per-stratum conformal** are built but off by default:
  `source_attribution.labels: four`, `conformal.per_stratum: true`.
- **`accuracy.residual_policy`** is currently vestigial (was the stub's profile
  knob); the Nexa default now lives in `source_quality.source_adequate_default`.
- **Nexa vs RavenPack profile:** with internal-corpus chunks (no url/domain)
  source_quality's metadata checks pass by construction; open-web sources
  (RavenPack) exercise all seven checks — populate the reliability list first.

---

## 10. Command cheat-sheet

```bash
# offline dev (this machine, mock)
pytest -q                       # full suite, no network
mypy --strict src/ ; ruff check .
python smoke_test.py            # mock providers pass
python -m rq_eval.runner        # fixtures end-to-end -> report

# programmatic use
python -c "from rq_eval.runner import evaluate; \
  r = evaluate('Who won?', 'Real Madrid won [chunk-1].', ['Real Madrid won in 2024.']); \
  print({k: v.score for k, v in r.results.items()})"
```

Inputs to `Evaluator.evaluate` are an `EvalInput{question, answer,
context: [ContextChunk], citations, profile}`; `ContextChunk` carries optional
`url/date/author/domain` metadata that source_quality reads.

---

## 11. How to extend

- **Add a dimension:** subclass `Dimension`, put it in `dimensions/<name>/`, log
  atoms, register a pure `Formula` in `scoring/formulas.py`, wire it into
  `runner.Evaluator.evaluate`, add a `README.md` + an `ARCHITECTURE.md` table row
  (the nav test enforces both).
- **Swap a model backend:** add a sibling class under `providers/live/` (lazy
  imports), select it in `ProviderFactory`, expose the choice as a config key.
- **Change a threshold/weight/version:** edit `config.yaml` only.
```
