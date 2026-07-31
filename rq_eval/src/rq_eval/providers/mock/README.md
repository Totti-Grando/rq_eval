# providers/mock

**Design ref:** build order B2 — deterministic mock implementations.

**Purpose:** offline, seeded, network-free stand-ins for every provider, good
enough to exercise every downstream code path and to make tests deterministic.

**Classes:**
- `DeterministicText` — shared seeded token-overlap + hashed-embedding helper (not a provider).
- `MockJudgeProvider` — [T3] tag-dispatched boolean verdicts (`[[affirm]]`/`[[deny]]`/`[[overlap[:tau]]]`/seeded).
- `MockGeneratorProvider` — [T3-gen] tag-dispatched text (`[[echo]]`/`[[sentences]]`/`[[repeat]]`).
- `MockEmbeddingProvider` — [T2] hashed bag-of-tokens vectors.
- `MockGroundingProvider` — [T2] three-way entailment: coverage ≥ entail_tau → E, negation-mismatch → C, else N.
- `MockRelevanceProvider` — [T2] token-Jaccard relevance score.
- `MockNlpProvider` — [T1/T2] regex segmentation + leading-pronoun coref.
- `MockResolverProvider` — [T1] reference exists unless it contains a fabricated-marker token.
- `MockGeneratorProvider` `[[triplets]]` — deterministic S|P|O parse splitter (E&T §0).

**Calculations (mock heuristics, not real scores):**
- `overlap(a, b) = |tokens(a) ∩ tokens(b)| / |tokens(a)|` (grounding coverage, judge `[[overlap]]`).
- `jaccard(a, b) = |tokens(a) ∩ tokens(b)| / |tokens(a) ∪ tokens(b)|` (relevance).
- `embed(t)` = L2-normalized counts of hashed content tokens; cosine rises with shared tokens.
- seeded bit = `sha256(seed | question | context)[0] & 1` (judge default).

**Determinism:** fully deterministic given the config seeds; identical inputs
always yield identical outputs.

**How to extend:** add a `[[tag]]` handler in `MockJudgeProvider` /
`MockGeneratorProvider` when a new phase needs a specific deterministic verdict.
