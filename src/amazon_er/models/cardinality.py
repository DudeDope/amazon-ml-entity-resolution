"""Per-query score aggregates for match-existence and cardinality models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

AGGREGATE_FEATURES = [
    "p_max",
    "p_second",
    "p_third",
    "p_sum",
    "p_mean",
    "margin",
    "n_candidates",
    "n_p50",
    "n_p20",
    "n_p05",
]


def query_aggregates(scores: pl.DataFrame) -> pl.DataFrame:
    for name in ("q", "score"):
        if name not in scores.columns:
            raise ValueError(f"Missing score column: {name}")
    frame = scores.group_by("q").agg(
        pl.col("score").max().alias("p_max"),
        pl.col("score").sort(descending=True).slice(1, 1).first().fill_null(0.0).alias("p_second"),
        pl.col("score").sort(descending=True).slice(2, 1).first().fill_null(0.0).alias("p_third"),
        pl.col("score").sum().alias("p_sum"),
        pl.col("score").mean().alias("p_mean"),
        pl.len().alias("n_candidates"),
        (pl.col("score") >= 0.5).sum().alias("n_p50"),
        (pl.col("score") >= 0.2).sum().alias("n_p20"),
        (pl.col("score") >= 0.05).sum().alias("n_p05"),
    )
    return frame.with_columns((pl.col("p_max") - pl.col("p_second")).alias("margin"))


@dataclass
class QueryModels:
    random_state: int = 42

    def fit(self, aggregates: pl.DataFrame, truth_counts: pl.DataFrame) -> "QueryModels":
        data = aggregates.join(truth_counts.select("q", "true_count"), on="q", how="inner")
        matrix = data.select(AGGREGATE_FEATURES).fill_null(0).to_numpy()
        count = data["true_count"].to_numpy().astype(np.int32)
        self.existence_model_ = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=self.random_state,
        ).fit(matrix, count > 0)
        self.count_model_ = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=self.random_state,
        ).fit(matrix, count)
        return self

    def predict(self, aggregates: pl.DataFrame) -> pl.DataFrame:
        matrix = aggregates.select(AGGREGATE_FEATURES).fill_null(0).to_numpy()
        exists = self.existence_model_.predict_proba(matrix)[:, 1]
        count = np.clip(self.count_model_.predict(matrix), 0.0, None)
        return aggregates.select("q").with_columns(
            pl.Series("existence_probability", exists),
            pl.Series("expected_count", count),
        )

