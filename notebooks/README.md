# Notebooks

Keep notebooks as thin clients over the package. A notebook may select a configuration,
launch a stage, inspect reports, and download artifacts; model logic belongs in `src/amazon_er`.
This keeps Colab, SageMaker, local Linux, and CI behavior consistent.

Recommended notebooks:

1. `01_data_audit.ipynb` — schema, missingness, languages, scripts and cardinality.
2. `02_candidate_recall.ipynb` — retrieval-source recall and blocking misses.
3. `03_model_errors.ipynb` — OOF slices, hard negatives and calibration.
4. `04_submission.ipynb` — run manifests, validator output and artifact hashes.

