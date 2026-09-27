# Architecture

The system separates **candidate recall**, **pair ranking**, and **set decoding**. Mixing these
objectives makes failures difficult to diagnose: a model cannot recover a true pair absent from
the candidate set, and a well-calibrated pair classifier is not automatically optimal for macro
F0.5 or empty entities.

## Stage contracts

| Stage | Required input | Main output | Acceptance gate |
|---|---|---|---|
| Normalize | Source TSV files | normalized Parquet records | row and ID counts preserved |
| Retrieve | normalized records | `(q, t, retrieval_source, rank, score)` | candidate recall by slice |
| Fuse | retrieval outputs | unique bounded candidate pairs | positives are never injected |
| Featurize | records and candidates | pair-feature Parquet shards | stable schema and fingerprints |
| GBDT | grouped train folds | OOF/test pair probabilities | untouched-fold macro F0.5 |
| Hard negatives | OOF scores and labels | difficult negative pairs | source/query diversity |
| Rerank | ambiguous top candidates | neural pair probabilities | OOF gain after cost accounting |
| Graph | pair scores and records | collective features | no label leakage across folds |
| Calibrate | OOF predictions | calibrated probabilities | reliability and log loss |
| Decode | calibrated scores | per-query match sets | exact macro F0.5 |
| Submit | decoded sets and candidates | two TSV files | official validator passes |

## Leakage controls

- Source-1 entities, rather than pairs, define folds.
- Candidate generation is label independent. Ground truth is used only to measure recall.
- Any feature based on model probabilities uses OOF probabilities during training.
- Target competition sees the same graph scope in validation and test.
- Model and decoder selection use OOF/calibration folds; the final holdout is not searched.
- Full-data refits inherit decisions fixed by OOF evidence.

## Candidate layer

Each retriever emits a common schema and provenance. Native Unicode and transliterated views are
kept separately. Reciprocal-rank fusion prevents one score scale from dominating, while exact
rare-token and numeric-address evidence may receive explicit bonuses. Per-source caps preserve
retrieval diversity before the final global cap.

## Model layer

The GBDT handles structured lexical, numeric, rank and graph signals. The neural reranker is
restricted to uncertain queries because applying a four-billion-parameter model to every pair is
expensive and usually redundant. Hard negatives come from cross-fitted mistakes and competing
queries for the same target.

## Decoder

The output metric is defined on each Source-1 match set. The decoder therefore models whether a
query has any match, estimates likely cardinality, selects a top-k set by expected F0.5, and can
apply deterministic target exclusivity. Global thresholding remains an ablation, not the default.

