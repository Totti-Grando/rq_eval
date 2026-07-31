# dimensions/source_attribution

**Design ref:** Evidence & Truthfulness §4 source_attribution — MAJOR, ALCE
citation recall/precision; **what accuracy imports as `attributed?`** (replaces
the plain-grounding placeholder).

**Purpose:** verify each cited claim is credited to the chunk that *actually*
supports it (a faithfulness gap distinct from correctness). Precision-favoring;
no-citation claims are excluded (they route to accuracy's unsourced residual).

**Classes:**
- `AttributionLabeler` — [T2] map E/N/C → AttrScore 3-way (default) or CAQA 4-way.
- `AlceScorer` — [code] ALCE citation recall + precision.
- `AttributionProviderImpl` — `attributed(claim, cited_chunk) -> {bool, confidence}` (accuracy import).
- `AttributionExport` — per-claim attribution confidence (for the conformal layer §5).
- `SourceAttributionDimension` — per-cited-claim verdict → ALCE precision (recall reported).

**Calculations:**
- per cited claim: entail the **cited** chunk vs the claim → Attributable iff label == E.
- `citation_precision = |attributable citations| / |citations|` (score; `mean` over per-citation atoms).
- `citation_recall = |statements whose citation set supports them| / |cited statements|`.
- `attributed? = Attributable ∧ confidence ≥ precision_threshold` (E8 replaces the confidence gate with the conformal threshold).
- Wilson 95% CI over (attributable, cited claims).

**Determinism:** the cited-chunk entailment is T2 (replays from the stamped
label); ALCE recall/precision are pure code. No generation, no judge.

**How to extend:** switch `source_attribution.labels: three|four`; tune
`precision_threshold`; multi-citation statements are handled by `AlceScorer` as-is.
