"""High-recall exact, sparse and optional dense candidate workflow."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..advanced_features import normalize_records
from .dense import DenseRetriever
from .exact import build_signature_index, exact_candidates
from .fusion import fuse_candidates
from .sparse import SparseRetriever


def generate_candidates(
    queries: pl.DataFrame,
    targets: pl.DataFrame,
    output_dir: str | Path | None = None,
    top_k: int = 200,
    sparse_top_k: int = 120,
    rrf_k: float = 60.0,
    max_features: int = 400_000,
    ngram_range: tuple[int, int] = (2, 5),
    dense_model: str | None = None,
    dense_revision: str | None = None,
    dense_top_k: int = 100,
    device: str = "cuda",
    batch_size: int = 256,
) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    q_frame = queries if "combined_norm" in queries.columns else normalize_records(queries, "q")
    t_frame = targets if "combined_norm" in targets.columns else normalize_records(targets, "t")
    frames: dict[str, pl.DataFrame] = {}
    for field in ("name_norm", "address_norm"):
        if field in q_frame.columns and field in t_frame.columns:
            index = build_signature_index(t_frame, [field], id_column="t")
            frames[f"exact_{field}"] = exact_candidates(q_frame, index, [field], id_column="q")
    for view in ("combined_norm", "combined_roman"):
        if view not in q_frame.columns or view not in t_frame.columns:
            continue
        retriever = SparseRetriever(ngram_range=ngram_range, max_features=max_features)
        retriever.fit(t_frame["t"].to_list(), t_frame[view].to_list())
        frames[f"sparse_{view}"] = retriever.search(
            q_frame["q"].to_list(), q_frame[view].to_list(), sparse_top_k, source=f"sparse_{view}"
        )
    if dense_model:
        dense = DenseRetriever(dense_model, dense_revision, device, batch_size)
        dense.fit(t_frame["t"].to_list(), t_frame["combined_roman"].to_list())
        frames["dense"] = dense.search(
            q_frame["q"].to_list(), q_frame["combined_roman"].to_list(), dense_top_k
        )
    fused = fuse_candidates(frames.values(), top_k=top_k, rrf_k=rrf_k)
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            frame.write_parquet(root / f"{name}.parquet")
        fused.write_parquet(root / "fused.parquet")
    return fused, frames
