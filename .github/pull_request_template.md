## Change

Describe the concrete pipeline behavior that changed.

## Evidence

- [ ] `python -m ruff check src tests`
- [ ] `python -m pytest`
- [ ] `amazon-er smoke --config configs/smoke.yaml`
- [ ] Candidate recall was measured if retrieval changed.
- [ ] OOF entity-level F0.5 was measured if ranking or decoding changed.

## Reproducibility

List the configuration, input fingerprints, model revisions, and generated artifacts needed to
reproduce the result. Do not attach private competition data or credentials.
