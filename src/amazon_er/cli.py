"""Command-line entry points for baseline and modular pipeline stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

from .advanced_features import build_pair_features, normalize_records
from .advanced_submission import write_submission
from .candidates.fusion import fuse_candidates
from .candidates.workflow import generate_candidates
from .dataset import attach_labels, prepare_split, read_frame
from .decision.decoder import decode_pairs
from .evaluation.splits import assign_group_folds
from .models.gbdt_ensemble import predict_ensemble, train_cross_fitted
from .models.hard_negatives import mine_hard_negatives
from .settings import load_settings
from .utils.system import print_report, system_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amazon-er", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="check dependencies, paths, RAM and GPU visibility")
    doctor.add_argument("--config", default="configs/production.yaml")

    smoke = commands.add_parser("smoke", help="run a small offline end-to-end pipeline")
    smoke.add_argument("--config", default="configs/smoke.yaml")

    baseline = commands.add_parser("baseline", help="run the audited classical baseline")
    baseline.add_argument("baseline_args", nargs=argparse.REMAINDER)

    prepare = commands.add_parser("prepare-data", help="convert raw competition TSVs to canonical Parquet")
    prepare.add_argument("--data-dir", default="dataset")
    prepare.add_argument("--output", default="artifacts/data")
    prepare.add_argument("--split", choices=["train", "test", "both"], default="both")

    generate = commands.add_parser("generate-candidates", help="run exact, sparse and optional dense retrieval")
    generate.add_argument("--queries", required=True)
    generate.add_argument("--targets", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--config", default="configs/production.yaml")

    fuse = commands.add_parser("fuse-candidates", help="reciprocal-rank fuse candidate parquet files")
    fuse.add_argument("--inputs", nargs="+", required=True)
    fuse.add_argument("--output", required=True)
    fuse.add_argument("--top-k", type=int, required=True)
    fuse.add_argument("--rrf-k", type=float, default=60.0)
    fuse.add_argument("--per-source-cap", type=int)

    normalize = commands.add_parser("normalize", help="add native and romanized record views")
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--output", required=True)
    normalize.add_argument("--id-column", required=True)

    features = commands.add_parser("build-features", help="create structured candidate-pair features")
    features.add_argument("--pairs", required=True)
    features.add_argument("--queries", required=True)
    features.add_argument("--targets", required=True)
    features.add_argument("--output", required=True)
    features.add_argument("--query-id", default="q")
    features.add_argument("--target-id", default="t")

    label = commands.add_parser("label-features", help="join truth and assign entity-grouped folds")
    label.add_argument("--features", required=True)
    label.add_argument("--truth", required=True)
    label.add_argument("--output", required=True)
    label.add_argument("--folds", type=int, default=5)
    label.add_argument("--seed", type=int, default=42)

    negatives = commands.add_parser("mine-negatives", help="mine OOF false positives and target conflicts")
    negatives.add_argument("--pairs", required=True)
    negatives.add_argument("--output", required=True)
    negatives.add_argument("--per-query", type=int, default=10)
    negatives.add_argument("--per-target", type=int, default=3)
    negatives.add_argument("--score-floor", type=float, default=0.01)

    ensemble = commands.add_parser("train-ensemble", help="train grouped cross-fitted LightGBM models")
    ensemble.add_argument("--features", required=True)
    ensemble.add_argument("--output", required=True)
    ensemble.add_argument("--config", default="configs/production.yaml")

    score = commands.add_parser("score-ensemble", help="score a feature parquet with saved LightGBM models")
    score.add_argument("--features", required=True)
    score.add_argument("--model-dir", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--full-model", action="store_true")

    decode = commands.add_parser("decode", help="decode scores and write official TSV files")
    decode.add_argument("--scores", required=True)
    decode.add_argument("--queries", required=True)
    decode.add_argument("--output", required=True)
    decode.add_argument("--strategy", default="expected_f_exclusive")
    decode.add_argument("--threshold", type=float, default=0.7)
    decode.add_argument("--beta", type=float, default=0.5)
    decode.add_argument("--k-max", type=int, default=25)
    decode.add_argument("--query-column", default="q")

    validate = commands.add_parser("validate", help="run the official-format submission validator")
    validate.add_argument("--matching", required=True)
    validate.add_argument("--candidate", required=True)
    validate.add_argument("--test-dir", required=True)
    validate.add_argument("--check-ids", action="store_true")
    return parser


def _write_parquet(frame: pl.DataFrame, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        settings = load_settings(args.config)
        print_report(system_report({name: settings.path(name) for name in ("data_dir", "work_dir", "output_dir")}))
    elif args.command == "smoke":
        from .smoke import run_smoke

        print(json.dumps(run_smoke(load_settings(args.config)), indent=2))
    elif args.command == "baseline":
        from .pipeline import main as baseline_main

        baseline_main(args.baseline_args)
    elif args.command == "prepare-data":
        splits = ("train", "test") if args.split == "both" else (args.split,)
        result = {split: {key: str(value) for key, value in prepare_split(args.data_dir, args.output, split).items()} for split in splits}
        print(json.dumps(result, indent=2))
    elif args.command == "generate-candidates":
        settings = load_settings(args.config)
        dense_enabled = bool(settings.get("candidates.dense.enabled", False))
        fused, frames = generate_candidates(
            read_frame(args.queries),
            read_frame(args.targets),
            output_dir=args.output,
            top_k=int(settings.get("candidates.final_cap")),
            sparse_top_k=int(settings.get("candidates.sparse.top_k")),
            rrf_k=float(settings.get("candidates.rrf_k")),
            max_features=int(settings.get("candidates.sparse.max_features")),
            ngram_range=tuple(settings.get("candidates.sparse.ngram_range")),
            dense_model=settings.get("candidates.dense.model") if dense_enabled else None,
            dense_revision=settings.get("candidates.dense.revision"),
            dense_top_k=int(settings.get("candidates.dense.top_k", 100)),
            device=str(settings.get("runtime.device", "cpu")),
            batch_size=int(settings.get("candidates.dense.batch_size", 256)),
        )
        print(json.dumps({"fused_pairs": fused.height, "sources": {name: frame.height for name, frame in frames.items()}}, indent=2))
    elif args.command == "fuse-candidates":
        frames = [pl.read_parquet(path) for path in args.inputs]
        _write_parquet(fuse_candidates(frames, args.top_k, args.rrf_k, per_source_cap=args.per_source_cap), args.output)
    elif args.command == "normalize":
        _write_parquet(normalize_records(read_frame(args.input), args.id_column), args.output)
    elif args.command == "build-features":
        frame = build_pair_features(
            read_frame(args.pairs),
            read_frame(args.queries),
            read_frame(args.targets),
            args.query_id,
            args.target_id,
        )
        _write_parquet(frame, args.output)
    elif args.command == "label-features":
        frame = attach_labels(read_frame(args.features), read_frame(args.truth))
        _write_parquet(assign_group_folds(frame, folds=args.folds, seed=args.seed), args.output)
    elif args.command == "mine-negatives":
        frame = mine_hard_negatives(pl.read_parquet(args.pairs), args.per_query, args.per_target, args.score_floor)
        _write_parquet(frame, args.output)
    elif args.command == "train-ensemble":
        settings = load_settings(args.config)
        train_cross_fitted(
            pl.read_parquet(args.features),
            args.output,
            rounds=int(settings.get("gbdt.rounds")),
            seed=int(settings.get("seed", 42)),
            threads=int(settings.get("runtime.threads", 1)),
            params=settings.get("gbdt.params", {}),
        )
    elif args.command == "score-ensemble":
        frame = read_frame(args.features)
        score = predict_ensemble(frame, args.model_dir, use_full_model=args.full_model)
        _write_parquet(frame.select("q", "t").with_columns(pl.Series("score", score)), args.output)
    elif args.command == "decode":
        scores, queries = read_frame(args.scores), read_frame(args.queries)
        selected = decode_pairs(scores, args.strategy, args.threshold, args.beta, args.k_max)
        write_submission(queries, scores, selected, args.output, args.query_column)
    elif args.command == "validate":
        validator = Path(__file__).resolve().parents[2] / "tools" / "validate_submission.py"
        command = [sys.executable, str(validator), "--matching", args.matching, "--candidate", args.candidate, "--test-dir", args.test_dir]
        if args.check_ids:
            command.append("--check-ids")
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
