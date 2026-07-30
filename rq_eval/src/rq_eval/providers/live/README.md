# providers/live

**Design ref:** build order B2 (live path) + migration notes in
`response-quality-design.md`.

**Purpose:** real backends for the target Bedrock machine. All heavy imports
(boto3, spaCy, coreferee, torch/fairseq) are lazy — inside methods — so these
classes are import- and construct-safe without the dependencies installed; the
import fires only when a method is actually called.

**Classes:**
- `BedrockSession` — lazy boto3 client factory (region/profile from config).
- `BedrockJudgeProvider` — [T3] Converse API, strict YES/NO → boolean.
- `BedrockGeneratorProvider` — [T3-gen] Converse API, temperature 0, seed-stamped.
- `TitanEmbeddingProvider` — [T2] Titan InvokeModel embeddings.
- `GuardrailGroundingProvider` — [T2] Guardrails contextual-grounding GROUNDING score.
- `GuardrailRelevanceProvider` — [T2] Guardrails contextual-grounding RELEVANCE score (Method B).
- `FairseqGroundingProvider` — [T2] torch.hub RoBERTa-large-MNLI entailment prob (optional; `models.nli: fairseq`).
- `SpacyNlpProvider` — [T1/T2] spaCy `en_core_web_lg` segmentation + coreferee coref.

**Calculations:** none of our own — these return raw model outputs (scores /
text / booleans). All thresholding and scoring happen upstream in our code.

**Determinism:** non-replayable (real model calls). temperature 0 reduces
variance; every atom stamps model+version so drift is detectable, not silent.

**How to extend:** add a live backend as a sibling class with lazy imports and
wire it into `ProviderFactory`. Verify with `python smoke_test.py` before use.
