"""Official singleton-aware entity macro F0.5, independent of pair classifier."""

import numpy as np


def entity_f05(truth: set, prediction: set) -> float:
    if not truth:
        return float(not prediction)
    return 1.25 * len(truth & prediction) / (0.25 * len(truth) + len(prediction))


def evaluate(truth: dict, predictions: dict, candidates: dict | None = None) -> dict:
    if set(predictions) - set(truth):
        raise ValueError("Prediction includes an unknown S1")
    scores, singles, nonsingles = [], [], []
    tp = fp = fn = count = recovered = 0
    for entity, actual in truth.items():
        predicted = predictions.get(entity, set())
        score = entity_f05(actual, predicted)
        scores.append(score)
        (nonsingles if actual else singles).append(score)
        tp += len(actual & predicted)
        fp += len(predicted - actual)
        fn += len(actual - predicted)
        count += len(predicted)
        if candidates is not None:
            recovered += len(actual & candidates.get(entity, set()))
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    result = dict(
        entities=len(truth),
        macro_f05=float(np.mean(scores)) if scores else 0.0,
        precision=p,
        recall=r,
        pair_f1=2 * p * r / (p + r) if p + r else 0.0,
        singleton_accuracy=float(np.mean(singles)) if singles else None,
        non_singleton_macro_f05=float(np.mean(nonsingles)) if nonsingles else None,
        predicted_links=count,
        predicted_matches_per_entity=count / max(1, len(truth)),
        false_positive_matches_per_entity=fp / max(1, len(truth)),
        tp=tp,
        fp=fp,
        fn=fn,
    )
    if candidates is not None:
        result["candidate_recall"] = recovered / max(1, tp + fn)
    return result
