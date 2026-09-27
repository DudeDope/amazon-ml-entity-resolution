"""End-to-end audited V0/V1 experiments. Run `python -m src.pipeline --help`."""

import argparse
import csv
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .audit import run_audit
from .blocking import INDEX_VERSION, Blocker, build_index
from .data_io import config, dump_json, fingerprint
from .error_analysis import analyze
from .features import fit_vectorizers, select_candidates
from .metric import evaluate
from .model import candidate_sets, fit_model, predict_sets, tune_threshold
from .pair_dataset import build_pairs, sample_entities


def subset_truth(truth, ids):
    return {q: truth[q] for q in ids}


def save_report(path, title, data):
    path.write_text(
        "# " + title + "\n\n```json\n" + json.dumps(data, indent=2, ensure_ascii=False) + "\n```\n",
        encoding="utf-8",
    )


def record_experiment(cfg, run_id, exp_id, split, metrics, threshold, notes, extra=None):
    extra = extra or {}
    path = Path(cfg["work_dir"]) / "experiments/experiments.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    row = dict(
        run_id=run_id,
        experiment_id=exp_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=commit,
        seed=cfg["seed"],
        validation_split=split,
        candidate_method="signature+fuzzy-prefilter+char-tfidf-union",
        candidate_parameters=json.dumps(
            {
                **{k: cfg[k] for k in ["block_limit", "prefilter_k", "top_k", "ngram_range"]},
                "top_k": extra.get("k", cfg["top_k"]),
                "composite_blocks": cfg.get("composite_blocks", False),
            },
            sort_keys=True,
        ),
        features="lexical,tfidf,numeric,country-equality,source,retrieval",
        model=(
            "retrieval_only"
            if metrics.get("macro_f05") is None
            else ("baseline" if exp_id in ["E000", "E001"] else "LightGBM")
        ),
        model_parameters=json.dumps(cfg["model"], sort_keys=True),
        threshold=threshold,
        candidate_recall=metrics.get("candidate_recall"),
        mean_candidates_s1=metrics.get("mean_candidates_s1"),
        macro_f05=metrics.get("macro_f05"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        singleton_accuracy=metrics.get("singleton_accuracy"),
        notes=notes,
        extra=json.dumps(extra or {}, sort_keys=True),
    )
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def exact_statistics(entities, truth, blocker):
    names = [
        "raw_name",
        "normalized_name",
        "raw_address",
        "normalized_address",
        "raw_name_address",
        "normalized_name_address",
        "country_name",
        "country_address",
        "digits",
    ]
    hits = dict.fromkeys(names, 0)
    links = 0
    country_mismatch = 0
    for qid, targets in truth.items():
        q = entities[qid]
        for cid in targets:
            r = blocker.get(cid)
            if r is None:
                raise ValueError("Invalid ground truth target: " + cid)
            links += 1
            rn = bool(q["business_name"]) and q["business_name"] == r["business_name"]
            ra = bool(q["business_address"]) and q["business_address"] == r["business_address"]
            nn = bool(q["name"]) and q["name"] == r["name"]
            na = bool(q["address"]) and q["address"] == r["address"]
            cc = q["country"] == r["country"]
            country_mismatch += not cc
            values = [
                rn,
                nn,
                ra,
                na,
                rn and ra,
                nn and na,
                cc and nn,
                cc and na,
                bool(q["digits"]) and q["digits"] == r["digits"],
            ]
            for name, value in zip(names, values):
                hits[name] += int(value)
    return {
        "entities": len(entities),
        "true_links": links,
        "cross_country_true_links": country_mismatch,
        "positive_link_coverage": {k: v / max(1, links) for k, v in hits.items()},
        "note": "Positive-link coverage on the deterministic entity sample; not exact-block precision or a full-file estimate.",
    }


def retrieval_benchmark(frame, truth, entities, target_count):
    result = []
    for view in ["name_tfidf", "address_tfidf", "combined_tfidf", "union"]:
        for k in [1, 3, 5, 10, 15]:
            selected = select_candidates(
                frame, k, None if view == "union" else [view], include_exact=view == "union"
            )
            cs = candidate_sets(selected)
            counts = np.array([len(cs.get(q, set())) for q in truth])
            score = evaluate(truth, {}, cs)
            row = {
                "view": view,
                "k": k,
                "candidate_recall": score["candidate_recall"],
                "mean_candidates": float(counts.mean()),
                "reduction_ratio": 1 - len(selected) / max(1, len(truth) * target_count),
            }
            for quant in [0.5, 0.9, 0.95, 0.99, 1]:
                row[f"count_q{quant}"] = float(np.quantile(counts, quant))
            for country in sorted({e["country"] for e in entities.values()}):
                ct = {q: t for q, t in truth.items() if entities[q]["country"] == country}
                row[f"recall_{country}"] = evaluate(ct, {}, cs)["candidate_recall"]
            result.append(row)
    return pd.DataFrame(result)


def train(cfg, ood=False, ablations=False):
    import platform
    import time

    import psutil

    started = time.perf_counter()
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    work = Path(cfg["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    for folder in ["artifacts", "reports", "experiments"]:
        (work / folder).mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = work / "artifacts" / run_id
    out.mkdir()
    reports = out / "reports"
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
        },
    )
    index = build_index(cfg, "train")
    blocker = Blocker(index, cfg)
    paths = list((Path(cfg["data_dir"]) / "train").glob("*.tsv"))
    cache_cfg = {
        k: v
        for k, v in cfg.items()
        if k not in ["output_dir", "work_dir", "model", "threshold_grid", "threads", "workers"]
    }
    cache_cfg["pipeline_version"] = 3
    cache_cfg["index_version"] = INDEX_VERSION
    cache_key = fingerprint(paths, cache_cfg)
    cache = work / "artifacts" / ("features_" + cache_key[:16] + ".joblib")
    if cache.exists():
        print("Loading fingerprinted feature cache", flush=True)
        entities, truth, splits, vectorizers, broad, stats = joblib.load(cache)
    else:
        entities, truth, splits = sample_entities(cfg)
        print(f"Sampled {len(entities):,} S1; full target corpus retained", flush=True)
        vectorizers = fit_vectorizers(blocker, cfg)
        # Preserve top-15 per view for K ablations; final V1 uses configured K.
        broad, stats = build_pairs(entities, truth, blocker, vectorizers, {**cfg, "top_k": 15})
        joblib.dump((entities, truth, splits, vectorizers, broad, stats), cache, compress=3)
    joblib.dump(vectorizers, out / "vectorizers.joblib", compress=3)
    dump_json(out / "split_ids.json", splits)
    dump_json(out / "sample_truth.json", {q: sorted(t) for q, t in truth.items()})
    stats.to_csv(reports / "blocking_pool_stats.csv", index=False)
    exact = exact_statistics(entities, truth, blocker)
    save_report(reports / "exact_match_statistics.md", "Exact-match coverage", exact)
    # Retrieval budgets are diagnosed on calibration, preserving the final holdout for final metrics.
    cal_ids = {q for q, s in splits.items() if s == "calibration"}
    calibration_truth = subset_truth(truth, cal_ids)
    cal_broad = broad[broad.qid.isin(cal_ids)]
    total_targets = blocker.con.execute("SELECT count(*) FROM records").fetchone()[0]
    retrieval = retrieval_benchmark(
        cal_broad, calibration_truth, {q: entities[q] for q in cal_ids}, total_targets
    )
    retrieval.to_csv(reports / "candidate_recall_vs_k.csv", index=False)
    for row in retrieval.to_dict("records"):
        record_experiment(
            cfg,
            run_id,
            "E002",
            "calibration",
            {"candidate_recall": row["candidate_recall"], "mean_candidates_s1": row["mean_candidates"]},
            None,
            "Candidate budget ablation",
            row,
        )
    print(retrieval[retrieval.view == "union"].to_string(index=False), flush=True)
    frame = select_candidates(broad, cfg["top_k"])
    frame["split"] = frame.qid.map(splits)
    frame["country"] = frame.qid.map({q: r["country"] for q, r in entities.items()})
    frame.to_parquet(out / "candidate_features.parquet", index=False)
    cs = candidate_sets(frame)
    hold_ids = {q for q, s in splits.items() if s == "holdout"}
    hold_truth = subset_truth(truth, hold_ids)
    hold = frame[frame.split == "holdout"].copy()
    fit = frame[frame.split == "fit"]
    cal = frame[frame.split == "calibration"]
    v0 = frame[(frame.name_exact == 1) & (frame.address_exact == 1) & (frame.same_country == 1)]
    v0pred = candidate_sets(v0[v0.split == "holdout"])
    empty = evaluate(hold_truth, {}, cs)
    exact_metrics = evaluate(hold_truth, v0pred, cs)
    save_report(
        reports / "v0_results.md",
        "V0 holdout baselines",
        {"all_empty": empty, "normalized_name_address_country": exact_metrics},
    )
    record_experiment(cfg, run_id, "E000", "holdout", empty, None, "Predict all empty")
    record_experiment(
        cfg,
        run_id,
        "E001",
        "holdout",
        exact_metrics,
        None,
        "Conservative normalized name+address+country equality",
    )
    print("V0", json.dumps({"empty": empty["macro_f05"], "exact": exact_metrics["macro_f05"]}), flush=True)
    model, columns = fit_model(fit, cfg)
    cal_prob = model.predict_proba(cal[columns])[:, 1]
    threshold, table = tune_threshold(cal, cal_prob, calibration_truth, cfg["threshold_grid"])
    table.to_csv(reports / "threshold_curve.csv", index=False)
    hold["probability"] = model.predict_proba(hold[columns])[:, 1]
    prediction = predict_sets(hold, hold.probability, threshold)
    metrics = evaluate(hold_truth, prediction, cs)
    metrics["mean_candidates_s1"] = sum(len(cs.get(q, set())) for q in hold_truth) / len(hold_truth)
    details = {
        "run_id": run_id,
        "threshold": threshold,
        "holdout": metrics,
        "fit_entities": sum(s == "fit" for s in splits.values()),
        "calibration_entities": len(cal_ids),
        "fit_positive_pairs": int(fit.label.sum()),
        "fit_hard_negatives": int((fit.label == 0).sum()),
        "features": len(columns),
        "sample_entities": len(entities),
        "full_target_pool": total_targets,
        "blocked_positive_links_not_in_training": sum(
            len(truth[q] - cs.get(q, set())) for q, s in splits.items() if s == "fit"
        ),
        "note": "Validation matcher fits only fit entities; threshold uses calibration. Final model refits all sampled S1 entities. No scores are leaderboard estimates.",
    }
    details["by_country"] = {
        c: evaluate(
            {q: t for q, t in hold_truth.items() if entities[q]["country"] == c},
            {q: p for q, p in prediction.items() if entities[q]["country"] == c},
            cs,
        )
        for c in sorted(hold.country.unique())
    }
    details["by_source"] = {
        s: evaluate(
            {q: {t for t in ts if t.startswith(s)} for q, ts in hold_truth.items()},
            {q: {t for t in ts if t.startswith(s)} for q, ts in prediction.items()},
            {q: {t for t in cs.get(q, set()) if t.startswith(s)} for q in hold_truth},
        )
        for s in ["S2-", "S3-"]
    }
    hold.to_parquet(out / "holdout_predictions.parquet", index=False)
    joblib.dump(
        {"model": model, "columns": columns, "threshold": threshold},
        out / "validation_model.joblib",
        compress=3,
    )
    pd.DataFrame(
        {"feature": columns, "gain": model.booster_.feature_importance(importance_type="gain")}
    ).sort_values("gain", ascending=False).to_csv(reports / "feature_importance.csv", index=False)
    details["errors"] = analyze(entities, hold_truth, prediction, cs, hold, blocker, reports)
    for exp in ["E003", "E004"]:
        met = (
            evaluate(hold_truth, predict_sets(hold, hold.probability, 0.5), cs) if exp == "E003" else metrics
        )
        record_experiment(
            cfg,
            run_id,
            exp,
            "holdout",
            met,
            0.5 if exp == "E003" else threshold,
            "Hard negatives; entity-separated calibration",
        )
    if ood:
        ood_results = []
        for country in sorted(fit.country.unique()):
            ofit = fit[fit.country == country]
            ocal = cal[cal.country == country]
            otest = hold[hold.country != country]
            ocalt = {q: t for q, t in calibration_truth.items() if entities[q]["country"] == country}
            ot = {q: t for q, t in hold_truth.items() if entities[q]["country"] != country}
            om, oc = fit_model(ofit, cfg)
            op = om.predict_proba(ocal[oc])[:, 1]
            threshold_ood, _ = tune_threshold(ocal, op, ocalt, cfg["threshold_grid"])
            metrics_ood = evaluate(
                ot, predict_sets(otest, om.predict_proba(otest[oc])[:, 1], threshold_ood), cs
            )
            ood_results.append(
                {
                    "trained_country": country,
                    "calibration_country": country,
                    "threshold": threshold_ood,
                    **metrics_ood,
                }
            )
            record_experiment(
                cfg,
                run_id,
                "E007",
                "country_holdout",
                metrics_ood,
                threshold_ood,
                "Train/calibrate " + country + "; test other country",
            )
        pd.DataFrame(ood_results).to_csv(reports / "country_ood.csv", index=False)
        details["ood"] = ood_results
    if ablations:
        results = []
        for label, prefixes in [
            ("no_address", ("address_", "numeric_", "postal_")),
            ("no_name", ("name_", "core_", "acronym_", "unicode_name")),
            ("no_numeric", ("numeric_", "postal_")),
            ("no_retrieval", ("retrieval_", "rank_", "block_")),
        ]:
            ac = [c for c in columns if not c.startswith(prefixes)]
            am, ac = fit_model(fit, cfg, ac)
            at, _ = tune_threshold(
                cal, am.predict_proba(cal[ac])[:, 1], calibration_truth, cfg["threshold_grid"]
            )
            met = evaluate(hold_truth, predict_sets(hold, am.predict_proba(hold[ac])[:, 1], at), cs)
            results.append({"ablation": label, "threshold": at, **met})
            record_experiment(cfg, run_id, "E003A", "holdout", met, at, "Feature-family ablation: " + label)
        pd.DataFrame(results).to_csv(reports / "ablations.csv", index=False)
        details["ablation_note"] = (
            "Feature-family ablations are diagnostic; overlapping cross-field/retrieval signals remain. No automatic model selection on holdout."
        )
    print("V1", json.dumps(details, indent=2), flush=True)
    # Refit on all labeled sampled entities only after freezing validation policy.
    final, columns = fit_model(frame, cfg, columns)
    final.booster_.save_model(str(out / "final_model.txt"))
    model_bundle = {
        "model": final,
        "columns": columns,
        "threshold": threshold,
        "vectorizers": vectorizers,
        "config": cfg,
        "training_fingerprint": cache_key,
        "run_id": run_id,
    }
    joblib.dump(model_bundle, out / "model.joblib", compress=3)
    details["model_tree_count"] = final.booster_.num_trees()
    details["tree_leaf_count"] = sum(t["num_leaves"] for t in final.booster_.dump_model()["tree_info"])
    details["training_wall_seconds"] = time.perf_counter() - started
    save_report(reports / "v1_results.md", "V1 validation results", details)
    dump_json(out / "metrics.json", details)
    (work / "artifacts/latest_run.txt").write_text(str(out.resolve()), encoding="utf-8")
    # Reports copied for convenience; original immutable run retains all evidence.
    import shutil

    for path in reports.iterdir():
        shutil.copy2(path, work / "reports" / path.name)
    blocker.close()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["audit", "index", "train", "train-full", "predict", "all"]
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--model")
    parser.add_argument("--ood", action="store_true")
    parser.add_argument("--ablations", action="store_true")
    args = parser.parse_args(argv)
    cfg = config(args.config)
    if args.command in ["audit", "all"]:
        run_audit(cfg)
    if args.command == "index":
        build_index(cfg, args.split)
    if args.command in ["train", "all"]:
        train(cfg, args.ood, args.ablations)
    if args.command == "train-full":
        from .full_training import train_full

        train_full(cfg)
    if args.command in ["predict", "all"]:
        from .inference import predict

        model = args.model or str(
            Path((Path(cfg["work_dir"]) / "artifacts/latest_run.txt").read_text(encoding="utf-8").strip())
            / "model.joblib"
        )
        predict(cfg, model)


if __name__ == "__main__":
    main()
