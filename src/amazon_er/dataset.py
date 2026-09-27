"""Competition TSV adapters and canonical Parquet preparation."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .advanced_features import normalize_records


def read_frame(path: str | Path) -> pl.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pl.read_parquet(source)
    if source.suffix.lower() in {".tsv", ".txt"}:
        return pl.read_csv(source, separator="\t", infer_schema_length=10_000, null_values=[])
    if source.suffix.lower() == ".csv":
        return pl.read_csv(source, infer_schema_length=10_000, null_values=[])
    raise ValueError(f"Unsupported table format: {source}")


def truth_to_pairs(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"source1_entity_id", "matched_entity_ids"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Truth file is missing {sorted(required - set(frame.columns))}")
    return (
        frame.select(
            pl.col("source1_entity_id").alias("q"),
            pl.col("matched_entity_ids").fill_null("").str.split(",").alias("t"),
        )
        .explode("t")
        .filter(pl.col("t") != "")
        .unique(["q", "t"])
    )


def prepare_split(data_dir: str | Path, output_dir: str | Path, split: str) -> dict[str, Path]:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    root = Path(data_dir) / split
    destination = Path(output_dir) / split
    destination.mkdir(parents=True, exist_ok=True)
    queries = read_frame(root / f"{split}_source1.tsv").rename({"entity_id": "q"})
    targets = pl.concat(
        [
            read_frame(root / f"{split}_source2.tsv"),
            read_frame(root / f"{split}_source3.tsv"),
        ],
        how="diagonal_relaxed",
    ).rename({"entity_id": "t"})
    queries = normalize_records(queries, "q")
    targets = normalize_records(targets, "t")
    paths = {"queries": destination / "queries.parquet", "targets": destination / "targets.parquet"}
    queries.write_parquet(paths["queries"])
    targets.write_parquet(paths["targets"])
    if split == "train":
        truth = truth_to_pairs(read_frame(root / "train_ground_truth.tsv"))
        paths["truth"] = destination / "truth.parquet"
        truth.write_parquet(paths["truth"])
    return paths


def attach_labels(features: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    if not {"q", "t"}.issubset(features.columns) or not {"q", "t"}.issubset(truth.columns):
        raise ValueError("features and truth must both contain q and t")
    positive = truth.select("q", "t").unique().with_columns(pl.lit(1).alias("label"))
    return features.join(positive, on=["q", "t"], how="left").with_columns(
        pl.col("label").fill_null(0).cast(pl.Int8)
    )
