import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .metric import evaluate

EXCLUDE = {"qid", "cid", "label", "probability", "split", "country"}


def feature_columns(frame):
    return [x for x in frame if x not in EXCLUDE]


def fit_model(frame, cfg, columns=None):
    columns = columns or feature_columns(frame)
    if frame.label.nunique() != 2:
        raise ValueError("Training requires both positive and negative pairs")
    model = LGBMClassifier(
        **cfg["model"],
        random_state=cfg["seed"],
        n_jobs=cfg["threads"],
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(frame[columns], frame.label)
    return model, columns


def predict_sets(frame, probabilities, threshold):
    selected = frame.loc[np.asarray(probabilities) >= threshold, ["qid", "cid"]]
    return {q: set(g.cid) for q, g in selected.groupby("qid", sort=False)}


def candidate_sets(frame):
    return {q: set(g.cid) for q, g in frame.groupby("qid", sort=False)}


def tune_threshold(frame, probabilities, truth, grid):
    result = []
    for threshold in grid:
        result.append(
            {"threshold": threshold, **evaluate(truth, predict_sets(frame, probabilities, threshold))}
        )
    table = pd.DataFrame(result)
    # For identical scores prefer the more conservative threshold.
    best = table.sort_values(["macro_f05", "threshold"], ascending=[False, False]).iloc[0]
    return float(best.threshold), table
