"""Submission creation from modular-pipeline score tables."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .submission import SubmissionWriter


def write_submission(
    queries: pl.DataFrame,
    candidates: pl.DataFrame,
    selected: pl.DataFrame,
    output_dir: str | Path,
    query_column: str = "q",
) -> tuple[Path, Path]:
    if query_column not in queries.columns:
        raise ValueError(f"Missing query column: {query_column}")
    for name, frame in (("candidates", candidates), ("selected", selected)):
        if not {"q", "t"}.issubset(frame.columns):
            raise ValueError(f"{name} must contain q and t")
    candidate_map = {
        str(q): list(map(str, values))
        for q, values in candidates.group_by("q").agg(pl.col("t").unique()).iter_rows()
    }
    match_map = {
        str(q): list(map(str, values))
        for q, values in selected.group_by("q").agg(pl.col("t").unique()).iter_rows()
    }
    root = Path(output_dir)
    with SubmissionWriter(root) as writer:
        for query in queries[query_column].cast(pl.String).to_list():
            writer.write(query, match_map.get(query, []), candidate_map.get(query, []))
    return root / "matching_results.tsv", root / "candidate_pairs.tsv"
