# Contributing

Use grouped entity folds, add deterministic tests for metric or decoding changes, and keep model
downloads out of unit tests. Do not commit competition data, outputs, credentials or generated
embeddings. Run `ruff check src tests` and `pytest` before opening a pull request.

Performance claims must identify the split, candidate population, configuration and artifact
manifest. Smoke-test scores are correctness checks and must never be presented as benchmark scores.

