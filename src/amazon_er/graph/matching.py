"""Many-query to one-target competition used by validation and inference."""

from __future__ import annotations

import numpy as np
import polars as pl


def target_exclusive_mask(q: np.ndarray, t: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Keep the highest-scoring query for each target.

    Ties are broken by the lexical/numeric order of `q`, making output independent of process or
    input ordering. A query may retain multiple targets; a target is assigned to at most one query.
    """
    if not (len(q) == len(t) == len(score)):
        raise ValueError("q, t and score lengths differ")
    if len(q) == 0:
        return np.zeros(0, dtype=bool)
    frame = pl.DataFrame({"_i": np.arange(len(q)), "q": q, "t": t, "score": score})
    best = (
        frame.sort(["t", "score", "q"], descending=[False, True, False])
        .group_by("t", maintain_order=True)
        .first()
    )
    mask = np.zeros(len(q), dtype=bool)
    mask[best["_i"].to_numpy()] = True
    return mask


def apply_target_exclusivity(frame: pl.DataFrame, score_column: str = "score") -> pl.DataFrame:
    mask = target_exclusive_mask(
        frame["q"].to_numpy(), frame["t"].to_numpy(), frame[score_column].to_numpy()
    )
    return frame.filter(pl.Series("_exclusive", mask))
