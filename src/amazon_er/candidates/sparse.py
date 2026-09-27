"""Character n-gram retrieval with a scikit-learn sparse index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


@dataclass
class SparseRetriever:
    ngram_range: tuple[int, int] = (3, 5)
    max_features: int = 400_000
    min_df: int = 2
    n_jobs: int = -1

    def fit(self, target_ids: Iterable[str], texts: Iterable[str]) -> "SparseRetriever":
        self.target_ids_ = np.asarray(list(target_ids), dtype=object)
        corpus = list(texts)
        if len(corpus) != len(self.target_ids_):
            raise ValueError("target_ids and texts have different lengths")
        self.vectorizer_ = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            dtype=np.float32,
            norm="l2",
        )
        matrix = self.vectorizer_.fit_transform(corpus)
        self.index_ = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=self.n_jobs)
        self.index_.fit(matrix)
        return self

    def search(
        self,
        query_ids: Iterable[str],
        texts: Iterable[str],
        top_k: int,
        source: str = "sparse_char",
        batch_size: int = 10_000,
    ) -> pl.DataFrame:
        query_ids = list(query_ids)
        texts = list(texts)
        if len(query_ids) != len(texts):
            raise ValueError("query_ids and texts have different lengths")
        top_k = min(top_k, len(self.target_ids_))
        parts = []
        for start in range(0, len(texts), batch_size):
            stop = min(len(texts), start + batch_size)
            query_matrix = self.vectorizer_.transform(texts[start:stop])
            distance, index = self.index_.kneighbors(query_matrix, n_neighbors=top_k)
            count = stop - start
            parts.append(
                pl.DataFrame(
                    {
                        "q": np.repeat(np.asarray(query_ids[start:stop], dtype=object), top_k),
                        "t": self.target_ids_[index.reshape(-1)],
                        "retrieval_source": np.repeat(source, count * top_k),
                        "rank": np.tile(np.arange(1, top_k + 1, dtype=np.int32), count),
                        "retrieval_score": (1.0 - distance.reshape(-1)).astype(np.float32),
                    }
                )
            )
        return pl.concat(parts) if parts else pl.DataFrame()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> "SparseRetriever":
        return joblib.load(path)

