"""Restartable inference: immutable per-chunk TSVs, then atomic final assembly."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import joblib

from .blocking import Blocker, build_index
from .data_io import chunks, dump_json, fingerprint
from .features import FeatureComputer
from .model import candidate_sets, predict_sets
from .normalization import prepare
from .submission import SubmissionWriter


def predict(cfg, model_path):
    bundle = joblib.load(model_path)
    # Candidate settings are part of the fitted pipeline, not independent inference knobs.
    for key in ["block_limit", "prefilter_k", "top_k", "ngram_range", "tfidf_max_features"]:
        if cfg[key] != bundle["config"][key]:
            raise ValueError("Inference configuration disagrees with trained pipeline: " + key)
    if cfg.get("composite_blocks", False) != bundle["config"].get("composite_blocks", False):
        raise ValueError("Composite blocking must match the trained configuration")
    index = build_index(cfg, "test")
    blocker = Blocker(index, cfg)
    computer = FeatureComputer(blocker, bundle["vectorizers"], cfg)
    source = Path(cfg["data_dir"]) / "test/test_source1.tsv"
    model_hash = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    signature = fingerprint(
        [
            source,
            Path(cfg["data_dir"]) / "test/test_source2.tsv",
            Path(cfg["data_dir"]) / "test/test_source3.tsv",
        ],
        {"model_sha256": model_hash, "batch": cfg["inference_batch_size"]},
    )
    parts = Path(cfg["work_dir"]) / "artifacts" / ("inference_" + signature[:16])
    parts.mkdir(exist_ok=True)
    dump_json(
        parts / "metadata.json",
        {"signature": signature, "model_sha256": model_hash, "run_id": bundle["run_id"]},
    )
    count = links = candidates = 0
    start = time.time()
    directories = []
    for i, frame in enumerate(chunks(source, cfg["inference_batch_size"])):
        part = parts / f"{i:06d}"
        directories.append(part)
        if (part / "complete.json").exists():
            info = json.loads((part / "complete.json").read_text(encoding="utf-8"))
        else:
            queries = [prepare(r) for r in frame.to_dict("records")]
            features, stats = computer.compute(queries)
            if len(features):
                probability = bundle["model"].predict_proba(features[bundle["columns"]])[:, 1]
                csets = candidate_sets(features)
                pred = predict_sets(features, probability, bundle["threshold"])
            else:
                csets = {}
                pred = {}
            with SubmissionWriter(part) as writer:
                for q in queries:
                    eid = q["entity_id"]
                    writer.write(eid, pred.get(eid, set()), csets.get(eid, set()))
            info = {
                "rows": len(frame),
                "matches": sum(map(len, pred.values())),
                "candidates": sum(map(len, csets.values())),
                "truncated_blocks": int(stats.truncated_blocks.sum()),
            }
            dump_json(part / "complete.json", info)
        count += info["rows"]
        links += info["matches"]
        candidates += info["candidates"]
        if i % 10 == 0:
            print(
                f"INFERENCE {count:,} S1, {links:,} links, {candidates:,} candidates, {time.time() - start:.0f}s",
                flush=True,
            )
    destination = Path(cfg["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ["matching_results.tsv", "candidate_pairs.tsv"]:
        temporary = destination / (filename + ".partial")
        with temporary.open("w", encoding="utf-8", newline="") as output:
            for i, part in enumerate(directories):
                with (part / filename).open(encoding="utf-8", newline="") as f:
                    header = next(f)
                    if i == 0:
                        output.write(header)
                    shutil.copyfileobj(f, output)
        os.replace(temporary, destination / filename)
    dump_json(
        destination / "inference_summary.json",
        {
            "rows": count,
            "matched_links": links,
            "candidate_links": candidates,
            "run_id": bundle["run_id"],
            "signature": signature,
            "seconds": time.time() - start,
        },
    )
    computer.close()
    blocker.close()
    validator = Path(__file__).resolve().parents[1] / "scripts/official_validate_submission.py"
    if not validator.exists():
        validator = Path(cfg["data_dir"]).parent / "utils/validate_submission.py"
    result = subprocess.run(
        [
            sys.executable,
            str(validator),
            "--matching",
            str(destination / "matching_results.tsv"),
            "--candidate",
            str(destination / "candidate_pairs.tsv"),
            "--test-dir",
            str(source.parent),
            "--check-ids",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    (destination / "official_validation.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError("Official validator failed; see output/official_validation.txt")
    if "not present in" in result.stdout:
        raise RuntimeError("Matches outside candidates")
    return destination
