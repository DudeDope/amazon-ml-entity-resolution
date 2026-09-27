# Reproduction checklist

1. Record SHA256 hashes or size/mtime fingerprints for all seven input TSVs.
2. Pin model revisions and package versions.
3. Run `amazon-er doctor` and save its JSON output.
4. Run the smoke test before a paid session.
5. Keep work/cache paths on local NVMe; sync checkpoints to durable storage between stages.
6. Measure candidate recall before training a more expensive matcher.
7. Save OOF predictions, fold models and decoder configuration.
8. Run the official validator with `--check-ids`.
9. Record hashes and row counts for both final TSV files.

Set immutable Hugging Face revisions in the production configuration before a final reproducible
run. Authentication tokens must come from environment variables or the platform secret store.
