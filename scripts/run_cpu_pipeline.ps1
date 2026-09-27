param(
    [string]$Config = "configs/production.yaml",
    [string]$DataDir = "dataset",
    [string]$Artifacts = "artifacts",
    [string]$Output = "output"
)
$ErrorActionPreference = "Stop"

amazon-er prepare-data --data-dir $DataDir --output "$Artifacts/data" --split both
foreach ($Split in @("train", "test")) {
    amazon-er generate-candidates --queries "$Artifacts/data/$Split/queries.parquet" --targets "$Artifacts/data/$Split/targets.parquet" --output "$Artifacts/candidates/$Split" --config $Config
    amazon-er build-features --pairs "$Artifacts/candidates/$Split/fused.parquet" --queries "$Artifacts/data/$Split/queries.parquet" --targets "$Artifacts/data/$Split/targets.parquet" --output "$Artifacts/features/$Split.parquet"
}
amazon-er label-features --features "$Artifacts/features/train.parquet" --truth "$Artifacts/data/train/truth.parquet" --output "$Artifacts/features/train_labeled.parquet"
amazon-er train-ensemble --features "$Artifacts/features/train_labeled.parquet" --output "$Artifacts/models/gbdt" --config $Config
amazon-er score-ensemble --features "$Artifacts/features/test.parquet" --model-dir "$Artifacts/models/gbdt" --output "$Artifacts/scores/test.parquet" --full-model
amazon-er decode --scores "$Artifacts/scores/test.parquet" --queries "$Artifacts/data/test/queries.parquet" --output $Output --strategy expected_f_exclusive
amazon-er validate --matching "$Output/matching_results.tsv" --candidate "$Output/candidate_pairs.tsv" --test-dir "$DataDir/test"
