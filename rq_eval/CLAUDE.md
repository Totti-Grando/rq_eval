# rq_eval — agent entrypoint

Certification-grade LLM answer evaluator. **Two categories, eight dimensions:**
Response Quality (accuracy, completeness, relevance, task_success) + Evidence &
Truthfulness (groundedness, hallucination, source_quality, source_attribution).

Status: fully implemented (B1–B10 + E1–E9) and design-synced (U1–U7: deterministic
parse-first extraction, relevance anchor-tree + orphan resolution, completeness
reference modes, parse-first triplets, forward-declared `ConsistencyProvider`).
~145 tests green, `mypy --strict` + `ruff` clean, runs **offline** in `providers.mode: mock`.

## Read in this order
1. `GUIDE.md` — orientation, repo map, mock→live table, AWS migration runbook, **known deviations (§9)**.
2. `ARCHITECTURE.md` — layer diagram + design-section → folder table.
3. `config.yaml` — every knob/default (the single source of truth).
4. Specs (canonical; **design wins over build orders**): `../response-quality-design.md`, `../evidence-truthfulness-design.md`, `../build-order-addendum-style-docs.md`.
5. The `README.md` inside each `src/rq_eval/**` folder — formulas + tiers for that module.

## Invariants — do NOT break these (they are tested)
- **AI emits booleans/labels/text only; code computes every number.** `scoring/` imports no model/provider code.
- Float→boolean thresholding happens in the **grader/dimension** layer from config, never in a provider.
- **Every verdict is an `AtomRecord`; every score replays** from the atom log via a registered `Formula` (no model call). Add a check → log an atom; add a score → register a pure formula.
- **`config.py` is the only reader of `config.yaml`/env/YAML.** No model ids/regions/thresholds/seeds/paths anywhere else (grep test enforces it).
- **Providers are built only via `ProviderFactory`.** One class per file, `<~300` lines, docstrings cite the design `§`/step + tier.
- Touch a folder → update its `README.md`; add a folder → add an `ARCHITECTURE.md` table row (a nav test checks both).

## Mental model
- `providers.mode: mock` = deterministic **lexical** stand-ins (token overlap/hash/regex) so everything runs with no network. Going live is a **config flip** (`mode: live`, `models.nli: bedrock`, `hallucination.resolver: live`, real ids/creds) — no code change. See `GUIDE.md` §6.
- Tiers: **T1** = pure code · **T2** = fixed model (NLI/grounding/embeddings), thresholded in code · **T3-gen** = pinned frozen text (units/outcomes/reverse-questions; the residual triplet + optional realizer) · **T3** = judge (the only non-replayable step, cornered into narrow residues).
- **Claim decomposition is deterministic parsing** (§0.2), not generation: `pipeline/` uses `NlpProvider.parse_clauses` + `T1Tools` (no judge/generator on the primary path); the optional surface-realizer is off by default. Triplets are parse-first too.
- **Relevance is an anchor-and-support tree** (§3): edges (entailment-confirmed) → anchors (on-ask + centrality + conformal) → tree (depth-graded) → orphans (off-topic / stranded-veracity / background). Stranded contradictions route to the forward-declared `ConsistencyProvider` stub. The `responsive` atom is still exported to accuracy.
- **Completeness has three reference modes** (§2): `generated` (default), `archetype`, `templated`; the mode is stamped as `DimensionResult.assurance_mode` and a human recall-sample `recall_miss_rate` is reported.
- Accuracy *imports* grounded (§1-E&T), source-adequate (§3), attributed (§4), responsive (§3-RQ) — it never recomputes them. Run order lives in `runner.py::Evaluator.evaluate`.

## Dev loop (this machine, Python 3.14, mock)
```bash
pytest -q                     # full suite, offline
mypy --strict src/ ; ruff check .
python smoke_test.py          # probes every provider
python -m rq_eval.runner      # fixtures end-to-end → report
```
Use the venv interpreter: `../.venv/Scripts/python.exe` (Windows).

## Where things live
`providers/` (interfaces + mock/ + live/ + factory) · `pipeline/` (§0 claims + triplets) ·
`graders/` (T1 tools, T2/T3 adapters) · `scoring/` (pure formulas, Wilson, bands, conformal) ·
`audit/` (atom stores, replay, calibration) · `contracts.py` (typed records) ·
`dimensions/<name>/` (one per dimension) · `runner.py` / `report.py` / `fixtures.py`.

## Before you "fix" something
Intentional deviations (stubs, deterministic-vs-judge choices, execution-sandbox
interface-only, ragas reimplemented, task_success verifier-routed §4) are listed
in `GUIDE.md` §9 and the per-dimension READMEs. Check there first.

## To extend
Add a dimension: subclass `dimensions/base.Dimension`, log atoms, register a
`scoring/formulas.py` formula, wire into `runner.Evaluator.evaluate`, add README +
ARCHITECTURE row. Swap a backend: add a `providers/live/` sibling (lazy imports),
select it in `ProviderFactory`, expose the choice as a config key. Change a
threshold/version: edit `config.yaml` only.
