"""Candidate schema validation and reciprocal-rank fusion."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import polars as pl

REQUIRED_COLUMNS = {"q", "t", "retrieval_source", "rank"}


class CandidateSchemaError(ValueError):
    pass


def validate_candidates(frame: pl.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise CandidateSchemaError(f"Missing candidate columns: {sorted(missing)}")
    if frame.select(
        pl.any_horizontal(pl.col("q").is_null(), pl.col("t").is_null()).any()
    ).item():
        raise CandidateSchemaError("Candidate q/t IDs may not be null")
    if frame.filter(pl.col("rank") <= 0).height:
        raise CandidateSchemaError("Candidate ranks must start at one")


def fuse_candidates(
    frames: Iterable[pl.DataFrame],
    top_k: int,
    rrf_k: float = 60.0,
    source_weights: Mapping[str, float] | None = None,
    per_source_cap: int | None = None,
) -> pl.DataFrame:
    if top_k <= 0 or rrf_k <= 0:
        raise ValueError("top_k and rrf_k must be positive")
    prepared = []
    for frame in frames:
        validate_candidates(frame)
        current = frame
        if per_source_cap is not None:
            current = current.filter(pl.col("rank") <= per_source_cap)
        if "retrieval_score" not in current.columns:
            current = current.with_columns(pl.lit(0.0).alias("retrieval_score"))
        prepared.append(current.select("q", "t", "retrieval_source", "rank", "retrieval_score"))
    if not prepared:
        return pl.DataFrame(
            schema={"q": pl.String, "t": pl.String, "fused_score": pl.Float64,
                    "best_rank": pl.Int32, "source_count": pl.UInt32, "sources": pl.List(pl.String)}
        )
    combined = pl.concat(prepared, how="vertical_relaxed")
    weights = source_weights or {}
    weight_table = pl.DataFrame(
        {"retrieval_source": list(weights), "_weight": list(weights.values())},
        schema={"retrieval_source": pl.String, "_weight": pl.Float64},
    )
    if weight_table.height:
        combined = combined.join(weight_table, on="retrieval_source", how="left")
    else:
        combined = combined.with_columns(pl.lit(1.0).alias("_weight"))
    combined = combined.with_columns(pl.col("_weight").fill_null(1.0))
    fused = (
        combined.with_columns((pl.col("_weight") / (rrf_k + pl.col("rank"))).alias("_rrf"))
        .group_by("q", "t")
        .agg(
            pl.col("_rrf").sum().alias("fused_score"),
            pl.col("rank").min().alias("best_rank"),
            pl.col("retrieval_score").max().alias("max_retrieval_score"),
            pl.col("retrieval_source").n_unique().alias("source_count"),
            pl.col("retrieval_source").unique().sort().alias("sources"),
        )
        .sort(["q", "fused_score", "best_rank", "t"], descending=[False, True, False, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("q").alias("fused_rank"))
        .filter(pl.col("fused_rank") <= top_k)
    )
    return fused
