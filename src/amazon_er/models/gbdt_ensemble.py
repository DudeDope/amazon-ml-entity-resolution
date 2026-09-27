"""Grouped cross-fitted LightGBM ensemble and all-data refit."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import polars as pl

NON_FEATURES = {"q", "t", "label", "y", "fold", "country", "script"}


@dataclass
class EnsembleResult:
    feature_names: list[str]
    model_paths: list[Path]
    oof: pl.DataFrame
    full_model_path: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


def default_params(seed: int, threads: int) -> dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 255,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "num_threads": threads,
        "verbosity": -1,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
    }


def _matrix(frame: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    return frame.select(features).fill_null(0).to_numpy().astype(np.float32, copy=False)


def train_cross_fitted(
    frame: pl.DataFrame,
    output_dir: str | Path,
    folds: Sequence[int] | None = None,
    feature_names: Sequence[str] | None = None,
    params: dict[str, Any] | None = None,
    rounds: int = 600,
    seed: int = 42,
    threads: int = 32,
    refit_all: bool = True,
) -> EnsembleResult:
    label = "label" if "label" in frame.columns else "y"
    for required in ("q", "t", label, "fold"):
        if required not in frame.columns:
            raise ValueError(f"Missing training column: {required}")
    features = list(
        feature_names
        or [
            name
            for name, dtype in frame.schema.items()
            if name not in NON_FEATURES and dtype.is_numeric()
        ]
    )
    if not features:
        raise ValueError("No model features were selected")
    fold_values = list(folds or sorted(frame["fold"].unique().to_list()))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    model_params = {**default_params(seed, threads), **(params or {})}
    predictions = []
    model_paths = []
    for fold in fold_values:
        train = frame.filter(pl.col("fold") != fold)
        holdout = frame.filter(pl.col("fold") == fold)
        dataset = lgb.Dataset(
            _matrix(train, features),
            label=train[label].to_numpy(),
            feature_name=features,
            free_raw_data=True,
        )
        model = lgb.train(model_params, dataset, num_boost_round=rounds)
        path = root / f"fold_{fold}.txt"
        model.save_model(str(path))
        model_paths.append(path)
        score = model.predict(_matrix(holdout, features)).astype(np.float32)
        predictions.append(
            holdout.select("q", "t", label, "fold").with_columns(pl.Series("score", score))
        )
    oof = pl.concat(predictions, how="vertical_relaxed").sort(["fold", "q", "t"])
    oof.write_parquet(root / "oof_predictions.parquet")
    full_path = None
    if refit_all:
        model = lgb.train(
            model_params,
            lgb.Dataset(_matrix(frame, features), label=frame[label].to_numpy(), feature_name=features),
            num_boost_round=rounds,
        )
        full_path = root / "full_model.txt"
        model.save_model(str(full_path))
    metadata = {
        "features": features,
        "folds": fold_values,
        "rounds": rounds,
        "params": model_params,
        "models": [path.name for path in model_paths],
        "full_model": full_path.name if full_path else None,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return EnsembleResult(features, model_paths, oof, full_path)


def predict_ensemble(
    frame: pl.DataFrame,
    model_dir: str | Path,
    use_full_model: bool = False,
) -> np.ndarray:
    root = Path(model_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    features = metadata["features"]
    names = [metadata["full_model"]] if use_full_model else metadata["models"]
    if any(name is None for name in names):
        raise ValueError("Requested full model was not saved")
    matrix = _matrix(frame, features)
    prediction = np.zeros(frame.height, dtype=np.float64)
    for name in names:
        prediction += lgb.Booster(model_file=str(root / name)).predict(matrix) / len(names)
    return prediction.astype(np.float32)
