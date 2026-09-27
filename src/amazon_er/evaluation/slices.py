"""Country, script, cardinality and error-table reporting."""

from __future__ import annotations

import polars as pl

from .metrics import entity_fbeta


def entity_error_table(
    truth: pl.DataFrame,
    prediction: pl.DataFrame,
    metadata: pl.DataFrame | None = None,
) -> pl.DataFrame:
    true_sets = truth.group_by("q").agg(pl.col("t").unique().sort().alias("truth"))
    pred_sets = prediction.group_by("q").agg(pl.col("t").unique().sort().alias("prediction"))
    result = true_sets.join(pred_sets, on="q", how="full", coalesce=True).with_columns(
        pl.col("truth").fill_null(pl.lit([]).cast(pl.List(pl.String))),
        pl.col("prediction").fill_null(pl.lit([]).cast(pl.List(pl.String))),
    )
    result = result.with_columns(
        pl.struct("truth", "prediction")
        .map_elements(lambda x: entity_fbeta(set(x["truth"]), set(x["prediction"])), return_dtype=pl.Float64)
        .alias("f05"),
        pl.col("truth").list.len().alias("true_count"),
        pl.col("prediction").list.len().alias("predicted_count"),
    )
    if metadata is not None:
        result = result.join(metadata, on="q", how="left", validate="1:1")
    return result.sort(["f05", "q"])


def slice_report(errors: pl.DataFrame, columns: tuple[str, ...] = ("country", "script", "true_count")) -> pl.DataFrame:
    parts = []
    for column in columns:
        if column not in errors.columns:
            continue
        parts.append(
            errors.group_by(column)
            .agg(pl.len().alias("queries"), pl.col("f05").mean().alias("macro_f05"))
            .with_columns(pl.lit(column).alias("slice"), pl.col(column).cast(pl.String).alias("value"))
            .select("slice", "value", "queries", "macro_f05")
        )
    return pl.concat(parts, how="vertical_relaxed") if parts else pl.DataFrame()
