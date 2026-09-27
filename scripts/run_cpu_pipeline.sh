#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/production.yaml}"
DATA_DIR="${2:-dataset}"
ARTIFACTS="${3:-artifacts}"
OUTPUT="${4:-output}"

amazon-er prepare-data --data-dir "$DATA_DIR" --output "$ARTIFACTS/data" --split both

for SPLIT in train test; do
  amazon-er generate-candidates \
    --queries "$ARTIFACTS/data/$SPLIT/queries.parquet" \
    --targets "$ARTIFACTS/data/$SPLIT/targets.parquet" \
    --output "$ARTIFACTS/candidates/$SPLIT" \
    --config "$CONFIG"
  amazon-er build-features \
    --pairs "$ARTIFACTS/candidates/$SPLIT/fused.parquet" \
    --queries "$ARTIFACTS/data/$SPLIT/queries.parquet" \
    --targets "$ARTIFACTS/data/$SPLIT/targets.parquet" \
    --output "$ARTIFACTS/features/$SPLIT.parquet"
done

amazon-er label-features \
  --features "$ARTIFACTS/features/train.parquet" \
  --truth "$ARTIFACTS/data/train/truth.parquet" \
  --output "$ARTIFACTS/features/train_labeled.parquet"
amazon-er train-ensemble \
  --features "$ARTIFACTS/features/train_labeled.parquet" \
  --output "$ARTIFACTS/models/gbdt" \
  --config "$CONFIG"
amazon-er score-ensemble \
  --features "$ARTIFACTS/features/test.parquet" \
  --model-dir "$ARTIFACTS/models/gbdt" \
  --output "$ARTIFACTS/scores/test.parquet" \
  --full-model
amazon-er decode \
  --scores "$ARTIFACTS/scores/test.parquet" \
  --queries "$ARTIFACTS/data/test/queries.parquet" \
  --output "$OUTPUT" \
  --strategy expected_f_exclusive
amazon-er validate \
  --matching "$OUTPUT/matching_results.tsv" \
  --candidate "$OUTPUT/candidate_pairs.tsv" \
  --test-dir "$DATA_DIR/test"
