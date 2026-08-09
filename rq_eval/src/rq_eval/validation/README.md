# validation

**Design ref:** §0.3 — the measured error bars the design is honest about. Not
scorers and not on any dimension's scoring path; harnesses that quantify how well
a bounded-but-imperfect step performs, reported alongside the scores.

**Purpose:** edge detection is the claim graph's weakest link — a missed edge
makes a valid derived claim look like a failed orphan. This measures detection
recall/precision against a human-linked sample so the number is *reported*, not
assumed. It is the gate for enabling the Layer-2 flags (accuracy DAG-rescue,
relevance tree).

**Classes:**
- `EdgeCase` — one labelled case: claims + gold `(parent_id, child_id)` edges.
- `EdgeRecallHarness` — runs `EdgeDetector` over the cases; returns an `EdgeRecallReport`.
- `EdgeRecallReport` — `recall`, `precision`, `true_positives`, `detected`, `gold`.

**Calculations:**
- `recall = |detected ∩ gold| / |gold|` (1.0 if no gold edges).
- `precision = |detected ∩ gold| / |detected|` (1.0 if nothing detected).

**Determinism:** pure code over the deterministic `EdgeDetector`; no model calls
beyond the detector's own fixed NLI; no scoring side effects.

**How to extend:** add labelled `EdgeCase`s (single-parent, convergent, arithmetic,
divergent); raise the bar the report must clear before flipping a Layer-2 flag on.
