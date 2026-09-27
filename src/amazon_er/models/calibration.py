"""Global or segmented probability calibration fitted on OOF predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibrator:
    method: str = "platt"
    min_segment_rows: int = 10_000

    def _new_model(self):
        if self.method == "platt":
            return LogisticRegression(C=1.0, max_iter=500, class_weight=None)
        if self.method == "isotonic":
            return IsotonicRegression(out_of_bounds="clip")
        raise ValueError(f"Unknown calibration method: {self.method}")

    @staticmethod
    def _fit(model, score: np.ndarray, label: np.ndarray):
        if isinstance(model, LogisticRegression):
            model.fit(score.reshape(-1, 1), label)
        else:
            model.fit(score, label)

    @staticmethod
    def _predict(model, score: np.ndarray) -> np.ndarray:
        if isinstance(model, LogisticRegression):
            return model.predict_proba(score.reshape(-1, 1))[:, 1]
        return model.predict(score)

    def fit(
        self,
        scores: Iterable[float],
        labels: Iterable[int],
        segments: Iterable[str] | None = None,
    ) -> "ProbabilityCalibrator":
        score = np.asarray(list(scores), dtype=np.float64)
        label = np.asarray(list(labels), dtype=np.int8)
        self.global_model_ = self._new_model()
        self._fit(self.global_model_, score, label)
        self.segment_models_ = {}
        if segments is not None:
            segment = np.asarray(list(segments), dtype=object)
            if len(segment) != len(score):
                raise ValueError("segments and scores have different lengths")
            for value in np.unique(segment):
                mask = segment == value
                if mask.sum() < self.min_segment_rows or len(np.unique(label[mask])) < 2:
                    continue
                model = self._new_model()
                self._fit(model, score[mask], label[mask])
                self.segment_models_[str(value)] = model
        return self

    def predict(self, scores: Iterable[float], segments: Iterable[str] | None = None) -> np.ndarray:
        score = np.asarray(list(scores), dtype=np.float64)
        result = self._predict(self.global_model_, score)
        if segments is None:
            return np.clip(result, 0.0, 1.0)
        segment = np.asarray(list(segments), dtype=object)
        for value, model in self.segment_models_.items():
            mask = segment == value
            if mask.any():
                result[mask] = self._predict(model, score[mask])
        return np.clip(result, 0.0, 1.0)

