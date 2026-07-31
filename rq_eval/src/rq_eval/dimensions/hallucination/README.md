# dimensions/hallucination

**Design ref:** Evidence & Truthfulness §2 hallucination — MAJOR. Two distinct
failures verified differently: the unsupported-claim rate (a T2 read of §1) and
the deterministic fabricated-citation **gate** (T1).

**Purpose:** report how much of the answer is unsupported (Neutral vs
Contradiction split), and hard-**gate** the run if any citation is fabricated.

**Classes:**
- `FabricationGate` (+ `FabricationResult`) — [T1] per-citation existence: chunk-id set-membership or URL/DOI resolve; any fabrication → gate FAIL.
- `HallucinationDimension` — unsupported rate + N/C split + gate → `DimensionResult`.

**Calculations:**
- `unsupported_rate = 1 − groundedness = 1 − |E| / |total triplets|` (score; `unsupported_rate` formula replays from the groundedness triplet atoms).
- `neutral_rate = |N| / total`, `contradiction_rate = |C| / total` (severe sub-case reported separately).
- fabrication gate: `exists(citation) = (chunk-id ∈ retrieved set) ∨ resolver.resolve(url/doi)`; `gate_failed = any(¬exists)`.
- band = R if `gate_failed` else `BandMapper(1 − unsupported_rate)`.

**Determinism:** the score replays bit-for-bit from §1's triplet verdicts; the
gate is pure T1 (set-membership + resolver, mock offline). No judge, no
generation.

**How to extend:** enable live URL/DOI checks via `hallucination.resolver: live`
and `doi_registry_enabled`; tune the E/N/C split via `entail_tau`/`contra_tau`.
