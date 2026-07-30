# audit

**Design ref:** §0.5 contracts, records & audit — the append-only atom log and
the replay verifier (this is where determinism becomes *verifiable*).

**Purpose:** persist one `AtomRecord` per yes/no check to an append-only store,
and recompute any `DimensionResult` score from its logged atoms + formula id
with no model call. `AtomLogger` is what graders/dimensions call to record a
check; `ReplayVerifier` is what proves the score is reproducible (and catches
tampering).

**Classes:**
- `Clock` / `SystemClock` / `FixedClock` — ISO timestamps (metadata only; never in a score or the atom hash).
- `AtomStore` — append-only interface (`append`, `all`, `get`, `by_ids`).
- `JsonlAtomStore` — one JSON object per line.
- `SqliteAtomStore` — `atoms(seq, id, data)` table, insertion-ordered.
- `AtomStoreFactory` — config-selected store (`paths.atom_log_backend`).
- `AtomLogger` — creates (via `AtomRecord.create`) + appends + returns an atom.
- `ReplayVerifier` — recompute + compare stored scores.

**Calculations:** none of its own — it *delegates* recomputation to `scoring/`'s
formula registry. Replay check: `registry.compute(result.formula_id,
store.by_ids(result.atom_ids)) == result.score` (exact equality).

**Determinism:** atom content hash excludes the timestamp, so logs from a
`FixedClock` are reproducible; T1/T2/code atoms replay bit-for-bit; T3 atoms
replay from their logged verdict with model+version stamped for drift detection.

**How to extend:** add a new backend by subclassing `AtomStore` and wiring it
into `AtomStoreFactory`. Never mutate a stored atom — the log is append-only.
