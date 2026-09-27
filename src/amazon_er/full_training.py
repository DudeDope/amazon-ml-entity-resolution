"""Out-of-core full-corpus training for the classical entity matcher.

Candidate features are generated in restartable Parquet shards.  Every retrieved
positive is retained, while the fit split keeps a configurable number of the
hardest retrieved negatives per S1 entity.  A small, deterministic calibration
and holdout sample retains every candidate so the competition metric and
candidate recall remain measurable without materializing the full pair table.
"""

import gc
import json
import os
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from .blocking import INDEX_VERSION, Blocker, build_index
from .data_io import chunks, dump_json, fingerprint, id_set, rows
from .error_analysis import analyze
from .features import FeatureComputer, fit_vectorizers
from .metric import evaluate
from .model import candidate_sets, feature_columns, fit_model, predict_sets, tune_threshold
from .normalization import prepare
from .validation import sampled, split_for

FULL_PIPELINE_VERSION = 1


def evaluation_role(entity_id, population, cfg):
    """Return calibration/holdout only for the deterministic evaluation sample."""
    if not sampled(entity_id, population, cfg["full_eval_entities"], cfg["seed"]):
        return "fit"
    split = split_for(entity_id, cfg["seed"])
    return split if split in {"calibration", "holdout"} else "fit"


def select_training_rows(frame, hard_negatives):
    """Keep all evaluation candidates and bounded hard negatives for model fitting."""
    if frame.empty:
        return frame
    evaluation = frame[frame.split != "fit"]
    fit = frame[frame.split == "fit"]
    positives = fit[fit.label == 1]
    negatives = fit[fit.label == 0].copy()
    if hard_negatives <= 0:
        negatives = negatives.iloc[0:0]
    elif len(negatives):
        score_columns = [
            c
            for c in [
                "combined_tfidf",
                "name_tfidf",
                "address_tfidf",
                "retrieval_combined",
                "retrieval_name",
                "retrieval_address",
                "name_ratio",
                "address_ratio",
            ]
            if c in negatives
        ]
        negatives["_hard_score"] = negatives[score_columns].max(axis=1)
        negatives = (
            negatives.sort_values(
                ["qid", "_hard_score", "cid"], ascending=[True, False, True]
            )
            .groupby("qid", sort=False)
            .head(hard_negatives)
            .drop(columns="_hard_score")
        )
    return pd.concat([positives, negatives, evaluation], ignore_index=True)


def _load_truth(path):
    truth = {}
    for row in rows(path):
        qid = row["source1_entity_id"]
        if qid in truth:
            raise ValueError("Duplicate ground truth entity: " + qid)
        # Keep the compact source string until its S1 batch is processed.
        truth[qid] = row["matched_entity_ids"]
    return truth


def _cache_key(cfg, paths):
    settings = {
        k: cfg[k]
        for k in [
            "seed",
            "block_limit",
            "composite_blocks",
            "top_k",
            "prefilter_k",
            "tfidf_sample",
            "tfidf_max_features",
            "ngram_range",
            "full_training_batch_size",
            "full_eval_entities",
            "hard_negatives_per_entity",
        ]
    }
    settings.update(
        full_pipeline_version=FULL_PIPELINE_VERSION,
        index_version=INDEX_VERSION,
    )
    return fingerprint(paths, settings)


def _merge_counts(total, current):
    for key, value in current.items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value


def _read_model_frame(shards, columns, split=None):
    frames = []
    for shard in shards:
        frame = pd.read_parquet(
            shard,
            columns=columns,
            filters=None if split is None else [("split", "==", split)],
        )
        if len(frame):
            frames.append(frame)
    if not frames:
        raise ValueError("No model rows found" + (" for split " + split if split else ""))
    result = pd.concat(frames, ignore_index=True)
    del frames
    return result


def _save_report(path, title, data):
    path.write_text(
        "# " + title + "\n\n```json\n" + json.dumps(data, indent=2, ensure_ascii=False) + "\n```\n",
        encoding="utf-8",
    )


def train_full(cfg):
    """Train the V1 matcher on candidates generated from every training S1 row."""
    required = [
        "full_training_batch_size",
        "full_eval_entities",
        "hard_negatives_per_entity",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError("Missing full-training configuration: " + ", ".join(missing))

    started = time.perf_counter()
    np.random.seed(cfg["seed"])
    work = Path(cfg["work_dir"])
    for folder in ["artifacts", "reports", "experiments"]:
        (work / folder).mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = work / "artifacts" / run_id
    reports = out / "reports"
    out.mkdir()
    reports.mkdir()
    dump_json(out / "config.json", cfg)
    dump_json(
        out / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": psutil.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total,
            "available_ram_bytes": psutil.virtual_memory().available,
            "workers": cfg.get("workers", 1),
            "tree_threads": cfg["threads"],
            "full_pipeline_version": FULL_PIPELINE_VERSION,
        },
    )

    data = Path(cfg["data_dir"]) / "train"
    source = data / "train_source1.tsv"
    truth_path = data / "train_ground_truth.tsv"
    target_paths = [data / "train_source2.tsv", data / "train_source3.tsv"]
    input_paths = [source, truth_path, *target_paths]
    population = sum(len(frame) for frame in chunks(source, cfg["chunk_size"]))
    truth_text = _load_truth(truth_path)
    if len(truth_text) != population:
        raise ValueError(
            f"Ground truth/S1 row-count mismatch: {len(truth_text):,} vs {population:,}"
        )

    index = build_index(cfg, "train")
    blocker = Blocker(index, cfg)
    vectorizers = fit_vectorizers(blocker, cfg)
    joblib.dump(vectorizers, out / "vectorizers.joblib", compress=3)

    key = _cache_key(cfg, input_paths)
    cache = work / "artifacts" / ("full_features_" + key[:16])
    cache.mkdir(exist_ok=True)
    metadata_path = cache / "metadata.json"
    expected_metadata = {
        "fingerprint": key,
        "population": population,
        "full_pipeline_version": FULL_PIPELINE_VERSION,
    }
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != expected_metadata:
            raise ValueError("Full-training cache metadata mismatch; use a fresh work_dir")
    else:
        dump_json(metadata_path, expected_metadata)

    totals = {}
    eval_truth = {}
    eval_entities = {}
    shard_paths = []
    size = cfg["full_training_batch_size"]
    with FeatureComputer(blocker, vectorizers, cfg) as computer:
        for shard_index, raw in enumerate(chunks(source, size)):
            shard = cache / f"features_{shard_index:06d}.parquet"
            summary_path = cache / f"features_{shard_index:06d}.json"
            shard_paths.append(shard)
            queries = [prepare(row) for row in raw.to_dict("records")]
            batch_truth = {}
            roles = {}
            for query in queries:
                qid = query["entity_id"]
                if qid not in truth_text:
                    raise ValueError("S1 entity missing from ground truth: " + qid)
                targets = id_set(truth_text[qid])
                batch_truth[qid] = targets
                role = evaluation_role(qid, population, cfg)
                roles[qid] = role
                if role != "fit":
                    eval_truth[qid] = targets
                    eval_entities[qid] = query

            if shard.exists() and summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                _merge_counts(totals, summary)
                print(
                    f"FULL FEATURES {min((shard_index + 1) * size, population):,}/{population:,} "
                    "entities (resumed)",
                    flush=True,
                )
                continue

            features, stats = computer.compute(queries)
            if len(features):
                features["label"] = np.fromiter(
                    (int(cid in batch_truth[qid]) for qid, cid in zip(features.qid, features.cid)),
                    dtype=np.int8,
                    count=len(features),
                )
                features["split"] = features.qid.map(roles)
                features["country"] = features.qid.map(
                    {query["entity_id"]: query["country"] for query in queries}
                )
                selected = select_training_rows(features, cfg["hard_negatives_per_entity"])
            else:
                selected = features.copy()
                selected["label"] = pd.Series(dtype=np.int8)
                selected["split"] = pd.Series(dtype=str)
                selected["country"] = pd.Series(dtype=str)

            candidate_counts = features.groupby("qid").size() if len(features) else pd.Series(dtype=int)
            positive_counts = (
                features[features.label == 1].groupby("qid").size()
                if len(features)
                else pd.Series(dtype=int)
            )
            summary = {
                "entities": len(queries),
                "entities_with_candidates": int(len(candidate_counts)),
                "entities_with_retrieved_positive": int(len(positive_counts)),
                "truth_links": int(sum(map(len, batch_truth.values()))),
                "retrieved_positive_links": int(features.label.sum()) if len(features) else 0,
                "candidate_rows": int(len(features)),
                "selected_rows": int(len(selected)),
                "selected_positive_rows": int(selected.label.sum()) if len(selected) else 0,
                "selected_negative_rows": int((selected.label == 0).sum()) if len(selected) else 0,
                "calibration_entities": int(sum(role == "calibration" for role in roles.values())),
                "holdout_entities": int(sum(role == "holdout" for role in roles.values())),
                "pool_size_sum": int(stats.pool_size.sum()) if len(stats) else 0,
                "truncated_blocks": int(stats.truncated_blocks.sum()) if len(stats) else 0,
            }
            temporary = shard.with_suffix(".parquet.partial")
            selected.to_parquet(temporary, index=False, compression="zstd")
            os.replace(temporary, shard)
            dump_json(summary_path, summary)
            _merge_counts(totals, summary)
            print(
                f"FULL FEATURES {min((shard_index + 1) * size, population):,}/{population:,} "
                f"entities; {totals.get('selected_rows', 0):,} selected rows",
                flush=True,
            )

    if totals.get("entities") != population:
        raise ValueError(
            f"Feature shards cover {totals.get('entities', 0):,} of {population:,} S1 entities"
        )
    if not shard_paths or any(not path.exists() for path in shard_paths):
        raise ValueError("Full feature shard set is incomplete")
    totals["candidate_recall_all_training"] = totals.get("retrieved_positive_links", 0) / max(
        1, totals.get("truth_links", 0)
    )
    totals["entity_positive_retrieval_rate"] = totals.get(
        "entities_with_retrieved_positive", 0
    ) / max(1, population)
    totals["mean_candidates_per_s1"] = totals.get("candidate_rows", 0) / max(1, population)
    totals["mean_selected_rows_per_s1"] = totals.get("selected_rows", 0) / max(1, population)
    dump_json(out / "full_feature_summary.json", totals)

    schema_names = pq.read_schema(shard_paths[0]).names
    columns = feature_columns(pd.DataFrame(columns=schema_names))
    numeric = ["label", *columns]
    print("Loading bounded full-corpus fit rows", flush=True)
    fit = _read_model_frame(shard_paths, numeric, "fit")
    validation_model, columns = fit_model(fit, cfg, columns)
    fit_positive_pairs = int(fit.label.sum())
    fit_negative_pairs = int((fit.label == 0).sum())
    del fit
    gc.collect()

    eval_columns = ["qid", "cid", "label", "split", "country", *columns]
    calibration = _read_model_frame(shard_paths, eval_columns, "calibration")
    holdout = _read_model_frame(shard_paths, eval_columns, "holdout")
    calibration_truth = {
        qid: targets for qid, targets in eval_truth.items() if split_for(qid, cfg["seed"]) == "calibration"
    }
    holdout_truth = {
        qid: targets for qid, targets in eval_truth.items() if split_for(qid, cfg["seed"]) == "holdout"
    }
    calibration_probability = validation_model.predict_proba(calibration[columns])[:, 1]
    threshold, threshold_table = tune_threshold(
        calibration, calibration_probability, calibration_truth, cfg["threshold_grid"]
    )
    threshold_table.to_csv(reports / "threshold_curve.csv", index=False)
    holdout["probability"] = validation_model.predict_proba(holdout[columns])[:, 1]
    candidates = candidate_sets(holdout)
    predictions = predict_sets(holdout, holdout.probability, threshold)
    metrics = evaluate(holdout_truth, predictions, candidates)
    metrics["mean_candidates_s1"] = sum(
        len(candidates.get(qid, set())) for qid in holdout_truth
    ) / max(1, len(holdout_truth))
    metrics["by_country"] = {
        country: evaluate(
            {qid: targets for qid, targets in holdout_truth.items() if eval_entities[qid]["country"] == country},
            {qid: values for qid, values in predictions.items() if eval_entities[qid]["country"] == country},
            candidates,
        )
        for country in sorted({eval_entities[qid]["country"] for qid in holdout_truth})
    }
    metrics["by_source"] = {
        source_prefix: evaluate(
            {qid: {target for target in targets if target.startswith(source_prefix)} for qid, targets in holdout_truth.items()},
            {qid: {target for target in values if target.startswith(source_prefix)} for qid, values in predictions.items()},
            {
                qid: {target for target in candidates.get(qid, set()) if target.startswith(source_prefix)}
                for qid in holdout_truth
            },
        )
        for source_prefix in ["S2-", "S3-"]
    }
    holdout.to_parquet(out / "holdout_predictions.parquet", index=False, compression="zstd")
    joblib.dump(
        {"model": validation_model, "columns": columns, "threshold": threshold},
        out / "validation_model.joblib",
        compress=3,
    )
    pd.DataFrame(
        {
            "feature": columns,
            "gain": validation_model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False).to_csv(reports / "feature_importance.csv", index=False)
    metrics["errors"] = analyze(
        eval_entities,
        holdout_truth,
        predictions,
        candidates,
        holdout,
        blocker,
        reports,
    )

    details = {
        "run_id": run_id,
        "training_mode": "full_s1_out_of_core",
        "threshold": threshold,
        "holdout": metrics,
        "training_entities_processed": population,
        "fit_positive_pairs": fit_positive_pairs,
        "fit_hard_negatives": fit_negative_pairs,
        "calibration_entities": len(calibration_truth),
        "holdout_entities": len(holdout_truth),
        "features": len(columns),
        "full_target_pool": blocker.con.execute("SELECT count(*) FROM records").fetchone()[0],
        "feature_generation": totals,
        "note": (
            "Validation model uses every non-evaluation S1 with retrieved candidates. "
            "All retrieved positives and bounded hard negatives are retained. The final model "
            "refits on selected rows from every S1, including calibration and holdout."
        ),
    }
    print("FULL V1 VALIDATION " + json.dumps(details, indent=2), flush=True)

    del calibration, holdout, validation_model
    gc.collect()
    print("Loading all selected rows for final refit", flush=True)
    final_frame = _read_model_frame(shard_paths, numeric)
    final_model, columns = fit_model(final_frame, cfg, columns)
    final_positive_pairs = int(final_frame.label.sum())
    final_negative_pairs = int((final_frame.label == 0).sum())
    del final_frame
    gc.collect()

    final_model.booster_.save_model(str(out / "final_model.txt"))
    bundle = {
        "model": final_model,
        "columns": columns,
        "threshold": threshold,
        "vectorizers": vectorizers,
        "config": cfg,
        "training_fingerprint": key,
        "run_id": run_id,
        "training_mode": "full_s1_out_of_core",
    }
    joblib.dump(bundle, out / "model.joblib", compress=3)
    details.update(
        final_positive_pairs=final_positive_pairs,
        final_negative_pairs=final_negative_pairs,
        model_tree_count=final_model.booster_.num_trees(),
        tree_leaf_count=sum(
            tree["num_leaves"] for tree in final_model.booster_.dump_model()["tree_info"]
        ),
        training_wall_seconds=time.perf_counter() - started,
    )
    _save_report(reports / "v1_full_results.md", "Full-corpus V1 validation results", details)
    dump_json(out / "metrics.json", details)
    (work / "artifacts/latest_run.txt").write_text(str(out.resolve()), encoding="utf-8")
    for path in reports.iterdir():
        shutil.copy2(path, work / "reports" / path.name)
    blocker.close()
    return out
