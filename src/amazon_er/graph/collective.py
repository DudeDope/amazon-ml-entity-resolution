"""Leakage-safe collective features over a scored candidate graph."""

from __future__ import annotations

import polars as pl


def add_collective_features(frame: pl.DataFrame, score_column: str = "score") -> pl.DataFrame:
    for column in ("q", "t", score_column):
        if column not in frame.columns:
            raise ValueError(f"Missing collective-feature column: {column}")
    q_context = frame.group_by("q").agg(
        pl.col(score_column).max().alias("q_score_max"),
        pl.col(score_column).sum().alias("q_score_sum"),
        pl.len().alias("q_candidate_count"),
    )
    t_context = frame.group_by("t").agg(
        pl.col(score_column).max().alias("t_score_max"),
        pl.col(score_column).sort(descending=True).slice(1, 1).first().fill_null(0.0).alias("t_score_second"),
        pl.len().alias("t_query_count"),
    )
    result = frame.join(q_context, on="q", how="left").join(t_context, on="t", how="left")
    return result.with_columns(
        (pl.col("q_score_max") - pl.col(score_column)).alias("q_score_gap"),
        (pl.col("t_score_max") - pl.col(score_column)).alias("t_score_gap"),
        pl.when(pl.col(score_column) == pl.col("t_score_max"))
        .then(pl.col(score_column) - pl.col("t_score_second"))
        .otherwise(pl.col(score_column) - pl.col("t_score_max"))
        .alias("t_competition_margin"),
        pl.col(score_column).rank("ordinal", descending=True).over("q").alias("q_score_rank"),
    )

