"""High-precision exact and signature candidate blocks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import polars as pl


def build_signature_index(
    targets: pl.DataFrame,
    fields: Iterable[str],
    id_column: str = "entity_id",
    max_block_size: int = 500,
) -> dict[tuple[str, str], list[str]]:
    index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in targets.select(id_column, *fields).iter_rows(named=True):
        target = str(row[id_column])
        for field in fields:
            value = str(row[field] or "").strip()
            if value:
                index[(field, value)].append(target)
    return {key: values for key, values in index.items() if len(values) <= max_block_size}


def exact_candidates(
    queries: pl.DataFrame,
    index: dict[tuple[str, str], list[str]],
    fields: Iterable[str],
    id_column: str = "entity_id",
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for query in queries.select(id_column, *fields).iter_rows(named=True):
        qid = str(query[id_column])
        seen: set[str] = set()
        rank = 0
        for field in fields:
            value = str(query[field] or "").strip()
            for target in index.get((field, value), []):
                if target in seen:
                    continue
                seen.add(target)
                rank += 1
                rows.append(
                    {
                        "q": qid,
                        "t": target,
                        "retrieval_source": f"exact_{field}",
                        "rank": rank,
                        "retrieval_score": 1.0,
                    }
                )
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"q": pl.String, "t": pl.String, "retrieval_source": pl.String,
                "rank": pl.Int32, "retrieval_score": pl.Float32}
    )

