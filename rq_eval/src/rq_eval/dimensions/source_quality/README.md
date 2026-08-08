# dimensions/source_quality

**Design ref:** Evidence & Truthfulness §3 source_quality — MAJOR, the bridge to
world factuality; **what accuracy imports as `source-adequate?`** (replaces the
old stub).

**Purpose:** decide whether the sources a claim relies on are *trustworthy* via
seven signals — mostly deterministic, one judged. `SourceQualityProviderImpl` is
what accuracy calls; `SourceQualityDimension` reports the category score.

**Classes:**
- `ReliabilityList` — [oracle] pinned domain allow/deny YAML.
- `CoiRule` — [T1] conflict-of-interest oracle (`config/coi_denylist.yaml` + affiliation match).
- `SourceQualityScorer` — the seven property checks (logs one atom each).
- `SourceQualityProviderImpl` — `adequate = score ≥ adequacy_threshold` (accuracy import).
- `SourceQualityDimension` — per-source score, averaged across the answer's sources.

**Calculations:**
- properties: reachable [T1], dated&fresh [T1] (`date ≤ as_of_date`, metadata-only
  unless `live_metadata_fetch`), authored [T1], reputable-domain [T1] (reliability
  list), **corroborated [T1]** (`|distinct docs in S| ≥ corroboration_min`, read off
  the §1 support set — **no NLI**), **supports-claim [T1]** (`S ≠ ∅`, imported from §1
  — **no NLI**), **disinterested [T1 COI rule]** (`¬(denylisted ∨ affiliation_conflict)`
  when decisive; ambiguous remainder samples a residual judge at `disinterest_sample_rate`).
- `source_quality = mean(the seven property booleans)` per source.
- `source-adequate? = source_quality ≥ adequacy_threshold`.
- internal-corpus sources (no url/domain) satisfy the metadata checks by
  construction (Nexa profile); no cited source → `source_adequate_default`.

**Determinism:** disinterest is now a **T1 COI rule** wherever it applies (the
old sampled judge is gone except for the genuinely ambiguous remainder), so with
`disinterest_sample_rate: 0.0` the whole dimension is deterministic; a self-
citation (source org == claim subject) is flagged conflicted with no judge.

**How to extend:** edit `config/reliability_list.yaml` (bump `reliability_version`)
and `config/coi_denylist.yaml` (bump `coi_version`); tune `adequacy_threshold`/
`corroboration_min`/`as_of_date`; raise `disinterest_sample_rate` to sample the
residual judge on the ambiguous remainder.
