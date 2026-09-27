"""Cross-fitted hard-negative mining without positive injection."""

from __future__ import annotations

import polars as pl

REQUIRED = {"q", "t", "label", "score"}


def mine_hard_negatives(
    pairs: pl.DataFrame,
    per_query: int = 10,
    per_target: int = 3,
    score_floor: float = 0.01,
) -> pl.DataFrame:
    missing = REQUIRED - set(pairs.columns)
    if missing:
        raise ValueError(f"Missing hard-negative columns: {sorted(missing)}")
    negatives = pairs.filter((pl.col("label") == 0) & (pl.col("score") >= score_floor))
    by_query = (
        negatives.sort(["q", "score", "t"], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("q").alias("_rank"))
        .filter(pl.col("_rank") <= per_query)
        .with_columns(pl.lit("query_confusion").alias("negative_source"))
        .drop("_rank")
    )
    by_target = (
        negatives.sort(["t", "score", "q"], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("t").alias("_rank"))
        .filter(pl.col("_rank") <= per_target)
        .with_columns(pl.lit("target_competition").alias("negative_source"))
        .drop("_rank")
    )
    keys = [column for column in ("q", "t", "fold") if column in pairs.columns]
    return (
        pl.concat([by_query, by_target], how="vertical_relaxed")
        .sort("score", descending=True)
        .unique(keys, keep="first")
    )

