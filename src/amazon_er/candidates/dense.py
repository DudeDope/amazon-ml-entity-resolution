"""Optional SentenceTransformer + FAISS dense retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl


def _optional_dependencies():
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised only with optional extras
        raise RuntimeError("Dense retrieval requires `pip install -e .[dense]`") from exc
    return faiss, SentenceTransformer


@dataclass
class DenseRetriever:
    model_name: str
    revision: str | None = None
    device: str = "cuda"
    batch_size: int = 256
    instruction: str = "Retrieve records describing the same real-world business entity."

    def _model(self):
        _, SentenceTransformer = _optional_dependencies()
        if not hasattr(self, "model_"):
            kwargs = {"device": self.device, "trust_remote_code": True}
            if self.revision:
                kwargs["revision"] = self.revision
            self.model_ = SentenceTransformer(self.model_name, **kwargs)
        return self.model_

    def encode(self, texts: Iterable[str], query: bool = False) -> np.ndarray:
        values = list(texts)
        if query:
            values = [f"Instruct: {self.instruction}\nQuery: {text}" for text in values]
        result = self._model().encode(
            values,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(values) > self.batch_size,
        )
        return np.asarray(result, dtype=np.float32)

    def fit(self, target_ids: Iterable[str], texts: Iterable[str]) -> "DenseRetriever":
        faiss, _ = _optional_dependencies()
        self.target_ids_ = np.asarray(list(target_ids), dtype=object)
        embeddings = self.encode(texts, query=False)
        if len(embeddings) != len(self.target_ids_):
            raise ValueError("target_ids and texts have different lengths")
        self.index_ = faiss.IndexFlatIP(embeddings.shape[1])
        self.index_.add(embeddings)
        return self

    def search(self, query_ids: Iterable[str], texts: Iterable[str], top_k: int) -> pl.DataFrame:
        query_ids = np.asarray(list(query_ids), dtype=object)
        embeddings = self.encode(texts, query=True)
        scores, indices = self.index_.search(embeddings, min(top_k, len(self.target_ids_)))
        width = indices.shape[1]
        return pl.DataFrame(
            {
                "q": np.repeat(query_ids, width),
                "t": self.target_ids_[indices.reshape(-1)],
                "retrieval_source": np.repeat("dense", len(query_ids) * width),
                "rank": np.tile(np.arange(1, width + 1, dtype=np.int32), len(query_ids)),
                "retrieval_score": scores.reshape(-1).astype(np.float32),
            }
        )

    def save(self, directory: str | Path) -> None:
        faiss, _ = _optional_dependencies()
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index_, str(root / "index.faiss"))
        np.save(root / "target_ids.npy", self.target_ids_)
        (root / "metadata.json").write_text(
            json.dumps({"model": self.model_name, "revision": self.revision}, indent=2), encoding="utf-8"
        )

