# dimensions/source_quality

**Design ref:** Evidence & Truthfulness §3 source_quality — MAJOR, the bridge to
world factuality; **what accuracy imports as `source-adequate?`** (replaces the
old stub).

**Purpose:** decide whether the sources a claim relies on are *trustworthy* via
seven signals — mostly deterministic, one judged. `SourceQualityProviderImpl` is
what accuracy calls; `SourceQualityDimension` reports the category score.

**Classes:**
- `ReliabilityList` — [oracle] pinned domain allow/deny YAML.
- `SourceQualityScorer` — the seven property checks (logs one atom each).
- `SourceQualityProviderImpl` — `adequate = score ≥ adequacy_threshold` (accuracy import).
- `SourceQualityDimension` — per-source score, averaged across the answer's sources.

**Calculations:**
- properties: reachable [T1], dated&fresh [T1] (`date ≤ as_of_date`), authored [T1],
  reputable-domain [T1] (reliability list), corroborated [T1] (`|distinct supporting
  domains/authors| ≥ corroboration_min`), supports-claim [T2] (`entails == E`),
  disinterested [T3, sampled at `disinterest_sample_rate`, else assumed true].
- `source_quality = mean(the seven property booleans)` per source.
- `source-adequate? = source_quality ≥ adequacy_threshold`.
- internal-corpus sources (no url/domain) satisfy the metadata checks by
  construction (Nexa profile); no cited source → `source_adequate_default`.

**Determinism:** six of seven properties are T1/T2 and replay; only the sampled
disinterest check is a judge, and at `disinterest_sample_rate: 0.0` (offline
default) it is assumed true → fully deterministic.

**How to extend:** edit `config/reliability_list.yaml` (bump `reliability_version`);
tune `adequacy_threshold`/`corroboration_min`/`as_of_date`; raise
`disinterest_sample_rate` on the target to actually sample the judge.
