# rq_eval — Response Quality evaluation

Scores four dimensions — **accuracy, completeness, relevance, task_success** —
under one rule: **AI extracts and judges yes/no; code computes every number.**
Every boolean is an audited atom; every score replays from the atoms with no
model call. Builds, runs, and tests **fully offline** (`providers.mode: mock`);
moves to a Bedrock machine by editing one file (`config.yaml`).

Implements `../response-quality-design.md` (spec v6) per
`../response-quality-build-order.md` and `../build-order-addendum-style-docs.md`.

## Quick start (offline / mock)

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
pytest                     # full suite, no network
python smoke_test.py       # probes mock providers
```

## Moving to the Bedrock machine (Python 3.11)

See `install.sh` and the migration notes at the end of
`../response-quality-design.md`. In short: `./install.sh`, fill `.env`, flip
`config.yaml` (`providers.mode: live`, model ids, guardrail id/version,
`models.nli: bedrock`), then `python smoke_test.py` must pass before evaluating.

## Layout

See `ARCHITECTURE.md` for the layer diagram, one end-to-end data flow, and the
design-section → folder map. Each `src/` subfolder has its own `README.md`.
