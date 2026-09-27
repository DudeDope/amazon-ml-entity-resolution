"""Small offline end-to-end run used by CI and new environments."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl

from .advanced_features import build_pair_features, normalize_records, numeric_feature_columns
from .advanced_submission import write_submission
from .artifacts import record_outputs, stage_manifest
from .candidates.exact import build_signature_index, exact_candidates
from .candidates.fusion import fuse_candidates
from .candidates.sparse import SparseRetriever
from .decision.decoder import decode_pairs
from .evaluation.metrics import evaluate_sets, pairs_to_sets
from .evaluation.splits import assign_group_folds
from .models.gbdt_ensemble import predict_ensemble, train_cross_fitted
from .settings import Settings


def _synthetic_records() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    names = [
        ("Shree Ganesh Traders", "श्री गणेश ट्रेडर्स"),
        ("Kaveri Medical Store", "ಕಾವೇರಿ ಮೆಡಿಕಲ್ ಸ್ಟೋರ್"),
        ("Annapurna Foods", "অন্নপূর্ণা ফুডস"),
        ("New Star Electronics", "New Star Electronic"),
        ("Mahalaxmi Textiles", "மகாலட்சுமி டெக்ஸ்டைல்ஸ்"),
        ("Royal Auto Works", "Royal Auto Work"),
        ("Sai Krupa Pharmacy", "साई कृपा फार्मसी"),
        ("National Hardware", "National Hardwares"),
        ("Balaji Sweets", "బాలాజీ స్వీట్స్"),
        ("City Mobile Centre", "City Mobile Center"),
        ("Asha Beauty Parlour", "આશા બ્યુટી પાર્લર"),
        ("Modern Book House", "Modern Books House"),
    ]
    query_rows, target_rows, truth_rows = [], [], []
    for index, (latin, variant) in enumerate(names):
        q, target = f"S1-{index:04d}", f"S2-{index:04d}"
        query_rows.append(
            {"q": q, "name": variant, "address": f"{index + 10} MG Road", "phone": f"900000{index:04d}", "country": "IN"}
        )
        target_rows.append(
            {"t": target, "name": latin, "address": f"{index + 10}, M G Road", "phone": f"900000{index:04d}", "country": "IN"}
        )
        truth_rows.append({"q": q, "t": target})
        target_rows.append(
            {"t": f"S3-{index:04d}", "name": f"Unrelated Shop {index}", "address": f"Sector {99-index}", "phone": f"800000{index:04d}", "country": "IN"}
        )
    return pl.DataFrame(query_rows), pl.DataFrame(target_rows), pl.DataFrame(truth_rows)


def run_smoke(settings: Settings) -> dict[str, object]:
    root = settings.path("work_dir")
    output = settings.path("output_dir")
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "smoke.manifest.json"
    with stage_manifest(manifest_path, "smoke", settings.digest()) as manifest:
        queries, targets, truth = _synthetic_records()
        q_norm = normalize_records(queries, "q")
        t_norm = normalize_records(targets, "t")
        exact_index = build_signature_index(t_norm, ["phone_norm"], id_column="t")
        exact = exact_candidates(q_norm, exact_index, ["phone_norm"], id_column="q")
        sparse = SparseRetriever(ngram_range=(2, 4), max_features=10_000, min_df=1, n_jobs=1)
        sparse.fit(t_norm["t"].to_list(), t_norm["combined_roman"].to_list())
        sparse_pairs = sparse.search(q_norm["q"].to_list(), q_norm["combined_roman"].to_list(), top_k=5)
        candidates = fuse_candidates([exact, sparse_pairs], top_k=6, rrf_k=30)
        features = build_pair_features(candidates, q_norm, t_norm)
        features = features.join(
            truth.with_columns(pl.lit(1).alias("label")), on=["q", "t"], how="left"
        ).with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
        features = assign_group_folds(features, folds=int(settings.get("validation.folds")), seed=int(settings.get("seed", 42)))
        feature_names = numeric_feature_columns(features)
        model_dir = root / "models"
        train_cross_fitted(
            features,
            model_dir,
            feature_names=feature_names,
            rounds=int(settings.get("gbdt.rounds", 50)),
            seed=int(settings.get("seed", 42)),
            threads=int(settings.get("runtime.threads", 2)),
            params=settings.get("gbdt.params", {}),
        )
        score = predict_ensemble(features, model_dir, use_full_model=True)
        scored = features.select("q", "t").with_columns(pl.Series("score", score))
        selected = decode_pairs(
            scored,
            strategy=settings.get("decoder.strategy"),
            threshold=float(settings.get("decoder.threshold", 0.7)),
            beta=float(settings.get("validation.beta", 0.5)),
            k_max=int(settings.get("decoder.k_max", 11)),
            extra_miss=float(settings.get("decoder.extra_miss", 0.0)),
        )
        matching, candidate = write_submission(queries.select("q"), scored, selected, output)
        validation_data = root / "validator-data"
        validation_data.mkdir(parents=True, exist_ok=True)
        queries.rename({"q": "entity_id"}).write_csv(
            validation_data / "test_source1.tsv", separator="\t"
        )
        for prefix, filename in (("S2-", "test_source2.tsv"), ("S3-", "test_source3.tsv")):
            targets.filter(pl.col("t").str.starts_with(prefix)).rename({"t": "entity_id"}).write_csv(
                validation_data / filename, separator="\t"
            )
        validator = Path(__file__).resolve().parents[2] / "tools" / "validate_submission.py"
        subprocess.run(
            [
                sys.executable,
                str(validator),
                "--matching",
                str(matching),
                "--candidate",
                str(candidate),
                "--test-dir",
                str(validation_data),
                "--check-ids",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        metrics = evaluate_sets(pairs_to_sets(truth), pairs_to_sets(selected), pairs_to_sets(candidates))
        summary = {
            "metrics": metrics,
            "candidates": candidates.height,
            "features": len(feature_names),
            "validator_passed": True,
        }
        summary_path = root / "smoke-summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        record_outputs(manifest, [matching, candidate, summary_path])
    return summary
