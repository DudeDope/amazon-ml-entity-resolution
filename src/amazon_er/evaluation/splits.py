"""Leakage-safe entity grouping and country holdout helpers."""

from __future__ import annotations

import numpy as np
import polars as pl

from ..data_io import stable_hash


def assign_group_folds(frame: pl.DataFrame, q_column: str = "q", folds: int = 5, seed: int = 42) -> pl.DataFrame:
    if folds < 2:
        raise ValueError("folds must be at least two")
    if q_column not in frame.columns:
        raise ValueError(f"Missing group column: {q_column}")
    values = frame[q_column].cast(pl.String).to_list()
    fold = np.asarray([stable_hash(f"{seed}:{value}") % folds for value in values], dtype=np.int16)
    return frame.with_columns(pl.Series("fold", fold))


def country_holdout(frame: pl.DataFrame, country: str, country_column: str = "country") -> tuple[pl.DataFrame, pl.DataFrame]:
    if country_column not in frame.columns:
        raise ValueError(f"Missing country column: {country_column}")
    return (
        frame.filter(pl.col(country_column) != country),
        frame.filter(pl.col(country_column) == country),
    )
