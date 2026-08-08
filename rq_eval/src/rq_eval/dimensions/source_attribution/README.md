# dimensions/source_attribution

**Design ref:** Evidence & Truthfulness §4 source_attribution — MAJOR, ALCE
citation recall/precision; **what accuracy imports as `attributed?`** (replaces
the plain-grounding placeholder).

**Purpose:** verify each cited claim is credited to a source that *actually*
supports it — a **set operation over the §1 support set `S`**, not a second NLI
pass. Resolve the cited set `C` (explicit + implicit scope), then
`attributed ⟺ C∩S≠∅`. Precision-favoring; no-citation claims are N/A (route out).

**Classes:**
- `resolve_explicit` / `ScopePropagator` (`citations.py`) — [T1] cited-set resolution: explicit regex + implicit scope confirmed in `S`.
- `AlceScorer` — [code] ALCE citation recall + precision.
- `AttributionProviderImpl` — `attributed(claim_id, cited) -> {bool, confidence}` = `C∩S≠∅ ∧ conformal` (accuracy import).
- `AttributionExport` — per-claim attribution confidence (for the conformal layer §5).
- `SourceAttributionDimension` — set-op over `S` → ALCE precision (recall + `explicit`/`implicit-confirmed` split reported).

**Calculations:**
- per cited claim: `attributed ⟺ C ∩ S ≠ ∅` (`S` = the claim's support chunk-ids from §1); no NLI here.
- `citation_precision = |attributed| / |cited claims|` (score; `mean` over per-claim atoms).
- `citation_recall = |claims with C∩S≠∅| / |cited claims|`.
- diagnostics: `mis_cited = |C−S|` (cited-but-unsupported), `uncited_supported = |S−C|`.
- `attributed? = (C∩S≠∅) ∧ confidence ≥ threshold` (E8 uses the conformal threshold); `attributed ⊆ grounded` by construction.
- Wilson 95% CI over (attributed, cited claims).

**Determinism:** attribution is pure set-ops + code over §1's logged support set
(atoms are **T1** now, no fresh NLI); ALCE recall/precision are pure code.

**How to extend:** tune `precision_threshold`; adjust implicit-scope rules in
`citations.py`; multi-citation statements are handled by the set-ops as-is.
