"""Turn pair probabilities into deterministic per-query match sets."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import polars as pl

from ..graph.matching import target_exclusive_mask
from .expected_f import expected_f_select, pi_select


def decode_pairs(
    scores: pl.DataFrame,
    strategy: str = "expected_f_exclusive",
    threshold: float = 0.7,
    beta: float = 0.5,
    k_max: int = 25,
    extra_miss: float = 0.0,
    existence_by_q: Mapping[object, float] | None = None,
) -> pl.DataFrame:
    for column in ("q", "t", "score"):
        if column not in scores.columns:
            raise ValueError(f"Missing decoder column: {column}")
    q = scores["q"].to_numpy()
    t = scores["t"].to_numpy()
    probability = np.clip(scores["score"].to_numpy().astype(np.float64), 0.0, 1.0)
    exclusive = target_exclusive_mask(q, t, probability)
    constrained = np.where(exclusive, probability, 0.0)
    if strategy == "threshold":
        selected = probability >= threshold
    elif strategy == "threshold_exclusive":
        selected = (probability >= threshold) & exclusive
    elif strategy == "expected_f":
        selected = expected_f_select(q, probability, extra_miss, beta, k_max)
    elif strategy == "expected_f_exclusive":
        selected = expected_f_select(q, constrained, extra_miss, beta, k_max)
    elif strategy == "pi_exclusive":
        if existence_by_q is None:
            raise ValueError("pi_exclusive requires existence_by_q")
        selected = pi_select(q, constrained, existence_by_q, extra_miss, beta, k_max)
    else:
        raise ValueError(f"Unknown decoder strategy: {strategy}")
    return (
        scores.with_columns(pl.Series("selected", selected))
        .filter(pl.col("selected"))
        .sort(["q", "score", "t"], descending=[False, True, False])
    )

