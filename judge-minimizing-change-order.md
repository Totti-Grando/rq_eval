# Claude Code change order — judge-minimizing reforms (retrofit into `rq_eval`)

**Goal:** apply the judge-minimizing reforms to the **existing** `rq_eval` codebase so it matches the updated `@response-quality-design.md` and `@evidence-truthfulness-design.md`. This is a **refactor of a working system**, not a new build: keep every existing test green, change providers/atoms in place, and where a step moves from judge → fixed, prove the old judge path is gone. Where the docs and this order differ, **the design docs win**.

**The five reforms (all backed in the docs' inline citations):**
1. relevance **on-ask**: generative judge → **fixed NLI + lexical flag** (DIVER-QA).
2. completeness **admissibility**: judge checks → **deterministic** (typed-atomic, coref self-contained, double-NLI decidability).
3. source_quality **disinterest**: sampled judge → **`[T1]` COI rule** + shrunken sampled residual.
4. **judge split**: one `JudgeProvider` → **`ScoringJudge`** (irreducible booleans) + **`ExplanationJudge`** (read-only summary).
5. every remaining `ScoringJudge` call is **reference-grounded** and **conformal-wrappable**.

**Unchanged constraints:** no AWS on the dev machine — all new behavior behind the existing provider interfaces with **mock** implementations, everything green offline in `mode: mock`; all new knobs in the **single `config.yaml`**; style addendum (`@build-order-addendum-style-docs.md`) applies to every phase (OOP, one class/file, per-folder README with the calculations, `mypy --strict`, ARCHITECTURE.md).

## How to run this

- **Read first:** the two updated design docs (esp. RQ §0.5 judge quarantine, §2 admissibility, §3 on-ask; Evidence §3 disinterest), the reform rationale in `@minimizing-judge-deterministic-reform.md`, and the current `providers/`, `dimensions/relevance/`, `dimensions/completeness/`, `dimensions/source_quality/` code.
- **Plan mode.** Propose a phased plan for R1–R7, wait for approval, one phase at a time, tests green after each.
- **Non-negotiable:** after this order, **no score-affecting step calls a generative judge except the named irreducible residuals** (accuracy unsourced, task_success adequacy, relevance abstention, decidability residual, disinterest residual); the `ExplanationJudge` touches **no** number. Booleans-only and replay stay intact.

---

## R1. Split the judge — `ScoringJudge` + `ExplanationJudge`

The keystone; do it first so later phases retarget onto the right role.
- Rename the existing `JudgeProvider` → **`ScoringJudge`**, signature `binary(question, context, reference=None) → {verdict: bool, reason}` (add the optional `reference`); it stays booleans-only (no numeric output). Update all current call sites to `ScoringJudge`.
- Add **`ExplanationJudge`** `summarize(dimension_results, atom_records) → str` — mock: templated stub; live: Bedrock Converse. It is **read-only**: it receives finished `DimensionResult`s + `AtomRecord`s and returns prose. Enforce structurally: it has **no `verdict` field**, writes **no AtomRecord**, and **no `formula_id` may import it** (add a test that asserts no scoring path references `ExplanationJudge`).
- `ProviderFactory` constructs both from config; both mock by default.
**Accept:** `JudgeProvider` no longer exists (grep-proven); every scoring judge call is a `ScoringJudge`; a test proves `ExplanationJudge` output never reaches a formula; all existing tests green.

## R2. relevance on-ask → fixed NLI + lexical flag (RQ §3)

Replace the live on-ask judge with the DIVER-QA recipe using existing tools:
- `T1Tools.ask_hypothesis(question) → str` `[T1]`: template the question's specific ask into a declarative hypothesis (no generation).
- on-ask NLI `[T2]`: `GroundingProvider.entails(premise=claim, hypothesis=ask)`, `on_ask_nli = (label == E)` at `entail_tau`.
- lexical flag `[T1]`: key-term/entity overlap on the question's content words → `on_ask_lex`.
- combine: `on_ask = on_ask_nli ∨ on_ask_lex` (rule + `entail_tau` from `config`).
- keep the mock as `overlap(question, claim) ≥ 0.5`; Method B and answer-level relevance unchanged.
- optional: allow the on-ask decision to be conformal-wrapped (reuse Evidence §5 machinery) behind a config flag.
**Accept:** no `ScoringJudge` call remains in the per-claim on-ask path; `responsive = on_topic ∧ on_ask` exported unchanged in shape; accuracy's `responsive` atom now traces to NLI+lexical AtomRecords (test: flipping the NLI label flips responsive and hence accuracy); tested.

## R3. completeness admissibility → deterministic (RQ §2)

Replace the one-time admission judge with three fixed checks in the gate:
- **atomic** `[T1]`: unit fills exactly one predicate / one 5W1H slot — `T1Tools` structural/parse check (Warrant Gap), replacing conjunction-split-then-judge.
- **self-contained** `[T1]`: no unresolved mention after coreferee — `∄ pronoun/mention unresolved` (Molecular Facts/FactCoref).
- **entailment-decidable** `[T2]`: **double-NLI agreement** — `GroundingProvider.entails(unit)` with context-only vs context+corpus; `decidable = (verdict_context == verdict_context_plus_corpus)`; only disagreements fall to a **`ScoringJudge`** residual admission.
- keep units at one-proposition granularity (don't over-decompose — Atomic-SNLI).
**Accept:** the per-unit admission judge is gone except the disagreement residual; a fixture unit needing world-knowledge (verdict flips with corpus) is correctly rejected/deferred; frozen-set behavior + τ validation unchanged; tested.

## R4. source_quality disinterest → COI rule + residual (Evidence §3)

- `[T1]` COI rule: `config/coi_denylist.yaml` (domains/authors) + **affiliation-vs-claim-subject match** (source's org == the claim's subject entity → conflicted). `disinterested = ¬(denylisted ∨ affiliation_conflict)` when the rule is decisive.
- only the **ambiguous remainder** (rule not decisive) is sampled to a `ScoringJudge` at `disinterest_sample_rate`.
- `source_quality = mean(7 properties)` unchanged; `adequate = score ≥ adequacy_threshold` unchanged.
**Accept:** disinterest is decided by the `[T1]` rule wherever it applies; judge fires only on the residual; COI list is config, not code; tested with a self-citation fixture (company release cited for a claim about itself → conflicted).

## R5. Reference-ground every remaining ScoringJudge call (RQ §0.5)

For each irreducible `ScoringJudge` atom, pass the available reference so it isn't a no-reference verdict (judges over-credit without one):
- accuracy **unsourced residual** → pass the retrieved corpus as `reference`.
- task_success **adequacy** → pass the pinned outcome template as `reference`.
- relevance **abstention**, admissibility **decidability residual**, disinterest **residual** → pass whatever reference exists (question/context).
**Accept:** no `ScoringJudge.binary()` is called with `reference=None` where a reference exists; tested.

## R6. Config, docs, determinism sweep

- Add config keys: on-ask (`entail_tau`, lexical rule, conformal flag), admissibility (double-NLI on/off, residual policy), `coi_denylist` path + affiliation rule, `disinterest_sample_rate`.
- Update per-folder READMEs with the **new calculations** (on-ask combine, double-NLI decidability, COI rule) and the **determinism line** (which parts now replay bit-for-bit that previously called a judge).
- Update `ARCHITECTURE.md`: the two judge roles + the explanation layer sitting after scoring.
- **Determinism ledger test:** assert the set of score-affecting `[T3]` atoms == exactly {unsourced_residual, task_success_adequacy, relevance_abstention, decidability_residual, disinterest_residual} — so no new judge silently creeps onto the scoring path later.
**Accept:** all new knobs in `config.yaml` only; READMEs/ARCHITECTURE current; the ledger test passes.

## R7. End-to-end + fixtures

- Extend fixtures: an off-ask answer caught by NLI-not-lexical (and vice-versa); a world-knowledge unit rejected by double-NLI; a self-citation flagged by the COI rule; a run where `ExplanationJudge` produces a summary and the ReplayVerifier still reproduces every score with no model call.
- Run the full suite (all prior + new) offline in `mode: mock`.
**Accept:** full offline run green; each reform's fixture shows the expected outcome; ReplayVerifier passes over a whole run including the explanation step.

---

## Definition of done

- `JudgeProvider` replaced by `ScoringJudge` + `ExplanationJudge`; the explanation layer is read-only and provably absent from every scoring path.
- on-ask, admissibility (atomic/self-contained/decidable), and disinterest are **fixed/deterministic-first**; the only score-affecting judge calls are the five named residuals, each reference-grounded.
- All new behavior offline-green in mock; all config in `config.yaml`; providers swap mock→live with no formula change.
- Determinism-ledger test locks the T3 surface; style addendum honored; all tests green; ReplayVerifier passes.
