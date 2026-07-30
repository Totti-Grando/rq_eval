# Addendum — code style, navigability & self-documentation (append to the Response Quality build order)

These requirements apply to **every phase (B1–B10)** and are part of each phase's acceptance criteria, not a cleanup pass at the end.

## 1. OOP architecture

- **Every dimension is a class** implementing a common abstract base: `Dimension` with `evaluate(EvalInput) -> DimensionResult`. One class per file: `AccuracyDimension`, `CompletenessDimension`, `RelevanceDimension`, `TaskSuccessDimension`. No free-floating scoring functions at module level — logic lives in classes.
- **Every grader is a class** behind its abstract interface (`JudgeProvider`, `GroundingProvider`, `EmbeddingProvider`, `RelevanceProvider`); mock and live are sibling subclasses. Construction only via a single `ProviderFactory` that reads config — nothing instantiates a provider directly.
- **Contracts are typed classes** (pydantic models or dataclasses): `Claim`, `AtomRecord`, `DimensionResult`, `EvalInput`. No raw dicts crossing module boundaries.
- **Composition over inheritance** beyond the base interfaces; dependency injection everywhere (a dimension receives its providers in `__init__` — it never creates them), which is what keeps everything mock-testable.
- **Single responsibility per class and file.** If a class does extraction *and* scoring, split it. Target: any file readable top-to-bottom in one sitting (< ~300 lines; hard cap 400 — refactor rather than exceed).
- **Naming is navigational:** the class name says what it is (`WilsonInterval`, `UnitAdmissibilityGate`, `ClaimExtractor`, `AtomLog`), the file name matches the class, the folder matches the design-doc section. Someone holding `response-quality-design.md` §2 step 3 must be able to guess the file: `dimensions/completeness/admissibility_gate.py`.

## 2. Navigability rules

- **Folder structure mirrors the design doc:** `pipeline/` = §0, `contracts.py` + `audit/` = §0.5, `dimensions/accuracy/` = §1, `dimensions/completeness/` = §2, `dimensions/relevance/` = §3, `dimensions/task_success/` = §4, `scoring/` = the formulas, `providers/` = the tier adapters. Each numbered build step in the design maps to an identifiable class or method — put the step number in the docstring (`"""§2 step 3 — unit admissibility gate."""`).
- **Every public class and method has a docstring** stating: what it does, its tier ([T1]/[T2]/[T3]), inputs → outputs, and — for anything that computes — the formula.
- **Type hints everywhere;** `mypy --strict` clean.
- **No magic values:** every threshold, weight, and constant is named and sourced from config; a reader can trace any number in an output back to a config key or a formula.
- **A top-level `ARCHITECTURE.md`** at repo root: one page — the layer diagram (contracts → providers → graders → pipeline → dimensions → scoring → audit), the data flow of one evaluation end-to-end, and a table mapping design-doc sections → folders.

## 3. Per-folder README.md (required in every subfolder of src/)

Each `README.md` follows this template and is written **when the folder's code lands, in the same phase**:

```
# <folder name>
**Design ref:** <which section/steps of response-quality-design.md this implements>
**Purpose:** 2–4 sentences — what this folder does and where it sits in the flow.
**Classes:** one line per class — name, role, tier.
**Calculations:** every formula computed here, written out explicitly
  (e.g. accuracy = Σ correct·w / Σ w where correct = grounded ∧ attributed ∧ responsive;
   Wilson CI formula; off-ask cap rule; strict_vital_recall = |vital supported|/|vital|).
  If the folder computes nothing, state "No calculations — <what it does instead>."
**Determinism:** which parts replay bit-for-bit, which are pinned-generative, which are judge.
**How to extend:** 1–3 bullets (e.g. "add a new task-type template by …").
```

- The **Calculations** section is mandatory and literal — the actual formulas with variable definitions, matching the design doc, so you can audit the math from the README without reading code.
- READMEs are **acceptance-gated**: a phase is not done if a touched folder's README is missing or stale. Add a CI/test check that every `src/` subfolder containing `.py` files has a `README.md`.

## 4. Acceptance additions (apply to every B-phase)

- `mypy --strict` and lint clean; file-length cap respected.
- Every touched subfolder has a current README.md per the template, with formulas present where anything is computed.
- `ARCHITECTURE.md` updated whenever a folder is added.
- A "navigation test" as documentation: ARCHITECTURE.md's section→folder table is verified against the actual tree by a small test, so the map can't rot.
