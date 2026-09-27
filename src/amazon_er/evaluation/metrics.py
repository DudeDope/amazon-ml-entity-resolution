"""Entity-level F-beta and candidate-recall metrics."""

from __future__ import annotations

from collections.abc import Mapping, Set

import numpy as np
import polars as pl


def entity_fbeta(truth: set[str], prediction: set[str], beta: float = 0.5) -> float:
    true_positive = len(truth & prediction)
    false_positive = len(prediction - truth)
    false_negative = len(truth - prediction)
    if not truth and not prediction:
        return 1.0
    beta_squared = beta * beta
    denominator = (1 + beta_squared) * true_positive + false_positive + beta_squared * false_negative
    return (1 + beta_squared) * true_positive / denominator if denominator else 0.0


def evaluate_sets(
    truth: Mapping[str, Set[str]],
    prediction: Mapping[str, Set[str]],
    candidates: Mapping[str, Set[str]] | None = None,
    beta: float = 0.5,
) -> dict[str, float]:
    query_ids = sorted(truth)
    scores = [entity_fbeta(set(truth[q]), set(prediction.get(q, set())), beta) for q in query_ids]
    recalls = []
    if candidates is not None:
        for q in query_ids:
            expected = set(truth[q])
            recalls.append(1.0 if not expected else len(expected & set(candidates.get(q, set()))) / len(expected))
    return {
        "macro_fbeta": float(np.mean(scores)) if scores else 0.0,
        "candidate_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "queries": float(len(query_ids)),
        "predicted_links": float(sum(len(prediction.get(q, set())) for q in query_ids)),
    }


def pairs_to_sets(frame: pl.DataFrame, q: str = "q", t: str = "t") -> dict[str, set[str]]:
    if not {q, t}.issubset(frame.columns):
        raise ValueError(f"Frame must contain {q!r} and {t!r}")
    return {str(row[0]): set(map(str, row[1])) for row in frame.group_by(q).agg(pl.col(t)).iter_rows()}
