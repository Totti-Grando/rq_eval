# dimensions/groundedness

**Design ref:** Evidence & Truthfulness §1 groundedness — MAJOR, **source
faithfulness**: is each claim entailed by the retrieved context. The reproducible
T2 core the whole category rests on.

**Purpose:** run one **per-chunk support pass** — for each triplet, entail the
top-`groundedness_k` chunks and collect the **support set** `S = {chunk : E}`
(grouped by document). Score the supported fraction; **export `S`** as the single
artifact the whole Evidence category derives from (source_quality reads
supports/corroboration off it; source_attribution intersects the cited set `C`
with it — no further NLI). Per-triplet confidences feed the conformal layer (§5).

**Scope statement:** groundedness is **direct, per-claim source presence** — the
axiom-*builder* for the claim graph (§RQ 0.3), **not** the answer's headline
factuality number. A validly-*derived* claim scores "not directly grounded" here;
its transitive truth is accuracy's DAG resolution (§RQ 1).

**Classes:**
- `SimilarityPreFilter` — [T1] `select_k`: rank chunks, keep top-`groundedness_k` per triplet (embeddings; not the score).
- `GroundednessDimension` — per kept chunk `entails(chunk, triplet)`; build + log `S`; score `|S≠∅|/|total|`.
- `GroundednessExport` — the support-set hand-off: per-claim `S` (chunk-ids + distinct docs) + answer-wide aggregate + grounded atom + confidences.

**Calculations:**
- `S(triplet) = {chunk : entails(chunk, triplet) == E}`; `supported ⟺ S ≠ ∅`.
- `groundedness = |triplets with S≠∅| / |total triplets|` (`mean` over triplet support atoms).
- per-claim `grounded? = AND(triplet supported)`; corroboration doc-count = `|distinct docs in S|` (read by §3).
- Wilson 95% CI over (supported, total).

**Determinism:** the pre-filter (embedding cosine) is deterministic and **not**
part of the score; entailment is T2 (replays from the stamped label); all math is
code. Flipping one triplet's label changes both the score and the imported
per-claim grounded (hence accuracy).

**How to extend:** swap the entailment backend via `models.nli`; tune
`entail_tau`/`contra_tau`; the per-claim grounded atom is the single import
surface for accuracy — do not recompute it there.
