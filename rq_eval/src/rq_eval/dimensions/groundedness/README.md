# dimensions/groundedness

**Design ref:** Evidence & Truthfulness §1 groundedness — MAJOR, **source
faithfulness**: is each claim entailed by the retrieved context. The reproducible
T2 core the whole category rests on.

**Purpose:** classify every claim-triplet against its nearest context span as
Entailment/Neutral/Contradiction and score the supported fraction. Exports the
per-claim `grounded?` (all its triplets E) for accuracy to import, and per-triplet
confidences for the conformal layer (§5).

**Classes:**
- `SimilarityPreFilter` — [T1] pick the nearest context span per triplet (embeddings; not the score).
- `GroundednessDimension` — orchestrate pre-filter → entailment → score.
- `GroundednessExport` — the §1→accuracy hand-off (per-claim grounded atom + triplet confidences).

**Calculations:**
- `groundedness = |E-labeled triplets| / |total triplets|` (`mean` over triplet support atoms; RAGAS-faithfulness `|V|/|S|`).
- per-claim `grounded? = AND(triplet.label == E)`.
- Wilson 95% CI over (supported, total).

**Determinism:** the pre-filter (embedding cosine) is deterministic and **not**
part of the score; entailment is T2 (replays from the stamped label); all math is
code. Flipping one triplet's label changes both the score and the imported
per-claim grounded (hence accuracy).

**How to extend:** swap the entailment backend via `models.nli`; tune
`entail_tau`/`contra_tau`; the per-claim grounded atom is the single import
surface for accuracy — do not recompute it there.
