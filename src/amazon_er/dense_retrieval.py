"""Restartable multilingual embedding and FAISS retrieval benchmark.

This module is optional and has no effect on the CPU V1 pipeline.  It is used to
measure dense-only and lexical+dense candidate recall before dense candidates are
allowed into a submission model.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data_io import chunks, dump_json, id_set, rows
from .metric import evaluate
from .validation import sampled


def record_text(row, prefix):
    name = " ".join(str(row.get("business_name", "")).split())
    address = " ".join(str(row.get("business_address", "")).split())
    country = " ".join(str(row.get("country", "")).split())
    return f"{prefix}: business name: {name} ; address: {address} ; country: {country}"


def _load_encoder(model_name, revision, device):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, revision=revision, device=device)


def _encode(model, texts, batch_size):
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float16)


def encode_corpus(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = _load_encoder(args.model, args.revision, args.device)
    total = 0
    manifests = []
    for source_number in [2, 3]:
        path = Path(args.data_dir) / args.split / f"{args.split}_source{source_number}.tsv"
        for shard_number, frame in enumerate(chunks(path, args.shard_rows)):
            stem = f"corpus_s{source_number}_{shard_number:06d}"
            embedding_path = output / f"{stem}_embeddings.npy"
            ids_path = output / f"{stem}_ids.npy"
            manifest_path = output / f"{stem}.json"
            if embedding_path.exists() and ids_path.exists() and manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest["rows"] != len(frame):
                    raise ValueError("Dense shard row-count mismatch: " + stem)
            else:
                texts = [record_text(row, "passage") for row in frame.to_dict("records")]
                embeddings = _encode(model, texts, args.batch_size)
                identifiers = np.asarray(frame.entity_id.str.encode("utf-8"), dtype="S24")
                temporary_embeddings = embedding_path.with_suffix(".npy.partial")
                temporary_ids = ids_path.with_suffix(".npy.partial")
                with temporary_embeddings.open("wb") as f:
                    np.save(f, embeddings, allow_pickle=False)
                with temporary_ids.open("wb") as f:
                    np.save(f, identifiers, allow_pickle=False)
                os.replace(temporary_embeddings, embedding_path)
                os.replace(temporary_ids, ids_path)
                manifest = {
                    "rows": len(frame),
                    "dimension": int(embeddings.shape[1]),
                    "model": args.model,
                    "revision": args.revision,
                    "split": args.split,
                    "source": source_number,
                    "shard": shard_number,
                }
                dump_json(manifest_path, manifest)
            total += manifest["rows"]
            manifests.append(manifest_path.name)
            print(f"DENSE CORPUS {total:,} records encoded", flush=True)
    dump_json(
        output / "corpus_manifest.json",
        {
            "rows": total,
            "model": args.model,
            "revision": args.revision,
            "split": args.split,
            "shards": manifests,
        },
    )


def _corpus_shards(directory):
    paths = sorted(Path(directory).glob("corpus_s*_embeddings.npy"))
    if not paths:
        raise ValueError("No encoded corpus shards found")
    return paths


def _faiss_gpu(index, device):
    import faiss

    resources = faiss.StandardGpuResources()
    options = faiss.GpuClonerOptions()
    options.useFloat16 = True
    return faiss.index_cpu_to_gpu(resources, device, index, options), resources


def build_index(args):
    import faiss

    directory = Path(args.embedding_dir)
    Path(args.index_path).parent.mkdir(parents=True, exist_ok=True)
    shards = _corpus_shards(directory)
    arrays = [np.load(path, mmap_mode="r") for path in shards]
    dimension = int(arrays[0].shape[1])
    if any(array.shape[1] != dimension for array in arrays):
        raise ValueError("Dense embedding dimensions disagree")
    counts = np.asarray([len(array) for array in arrays], dtype=np.int64)
    total = int(counts.sum())
    rng = np.random.default_rng(args.seed)
    training = []
    remaining = args.training_vectors
    for i, array in enumerate(arrays):
        proportional = round(args.training_vectors * len(array) / total)
        requested = remaining if i == len(arrays) - 1 else proportional
        allocation = min(len(array), max(0, remaining), max(0, requested))
        if allocation:
            indices = rng.choice(len(array), size=allocation, replace=False)
            training.append(np.asarray(array[indices], dtype=np.float32))
            remaining -= allocation
    training_vectors = np.concatenate(training, axis=0)
    training_count = len(training_vectors)
    index = faiss.index_factory(dimension, args.factory, faiss.METRIC_INNER_PRODUCT)
    print(
        f"Training FAISS {args.factory} on {len(training_vectors):,} of {total:,} vectors",
        flush=True,
    )
    index.train(training_vectors)
    del training_vectors
    gpu_resources = None
    if args.gpu:
        index, gpu_resources = _faiss_gpu(index, args.gpu_device)

    target_ids_path = Path(args.index_path).with_name("dense_target_ids.npy")
    target_ids = np.lib.format.open_memmap(target_ids_path, mode="w+", dtype="S24", shape=(total,))
    offset = 0
    for embedding_path, array in zip(shards, arrays):
        ids_path = embedding_path.with_name(embedding_path.name.replace("_embeddings.npy", "_ids.npy"))
        identifiers = np.load(ids_path, mmap_mode="r")
        if len(identifiers) != len(array):
            raise ValueError("Dense embedding/ID shard mismatch: " + embedding_path.name)
        index.add(np.asarray(array, dtype=np.float32))
        target_ids[offset : offset + len(array)] = identifiers
        offset += len(array)
        target_ids.flush()
        print(f"FAISS ADD {offset:,}/{total:,}", flush=True)
    if args.gpu:
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, args.index_path)
    dump_json(
        Path(args.index_path).with_suffix(".json"),
        {
            "rows": total,
            "dimension": dimension,
            "factory": args.factory,
            "training_vectors": training_count,
            "seed": args.seed,
            "target_ids": target_ids_path.name,
        },
    )
    # Keep the GPU resource alive until after the index has been copied back.
    del gpu_resources


def query_index(args):
    import faiss

    index_path = Path(args.index_path)
    metadata = json.loads(index_path.with_suffix(".json").read_text(encoding="utf-8"))
    target_ids = np.load(index_path.with_name(metadata["target_ids"]), mmap_mode="r")
    index = faiss.read_index(str(index_path))
    if hasattr(index, "nprobe"):
        index.nprobe = args.nprobe
    gpu_resources = None
    if args.gpu:
        index, gpu_resources = _faiss_gpu(index, args.gpu_device)
        if hasattr(index, "nprobe"):
            index.nprobe = args.nprobe
    model = _load_encoder(args.model, args.revision, args.device)

    source = Path(args.data_dir) / args.split / f"{args.split}_source1.tsv"
    population = sum(len(frame) for frame in chunks(source, args.shard_rows))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_total = 0
    for shard_number, raw in enumerate(chunks(source, args.shard_rows)):
        if args.sample_entities:
            raw = raw[
                raw.entity_id.map(
                    lambda entity_id: sampled(
                        entity_id, population, args.sample_entities, args.seed
                    )
                )
            ]
        if raw.empty:
            continue
        destination = output / f"dense_candidates_{shard_number:06d}.parquet"
        info_path = destination.with_suffix(".json")
        if destination.exists() and info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            selected_total += info["queries"]
            print(f"DENSE QUERY {selected_total:,} queries (resumed)", flush=True)
            continue
        records = raw.to_dict("records")
        texts = [record_text(row, "query") for row in records]
        embeddings = _encode(model, texts, args.batch_size).astype(np.float32)
        scores, positions = index.search(embeddings, args.top_k)
        output_rows = []
        for query_number, row in enumerate(records):
            for rank, (score, position) in enumerate(
                zip(scores[query_number], positions[query_number]), 1
            ):
                if position < 0:
                    continue
                output_rows.append(
                    {
                        "qid": row["entity_id"],
                        "cid": target_ids[position].decode("utf-8"),
                        "dense_score": float(score),
                        "dense_rank": rank,
                    }
                )
        frame = pd.DataFrame(output_rows)
        temporary = destination.with_suffix(".parquet.partial")
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, destination)
        info = {"queries": len(records), "pairs": len(frame), "top_k": args.top_k}
        dump_json(info_path, info)
        selected_total += len(records)
        print(f"DENSE QUERY {selected_total:,} queries", flush=True)
    dump_json(
        output / "query_manifest.json",
        {
            "queries": selected_total,
            "model": args.model,
            "revision": args.revision,
            "top_k": args.top_k,
            "nprobe": args.nprobe,
            "sample_entities": args.sample_entities,
            "split": args.split,
        },
    )
    del gpu_resources


def _sets(frame):
    return {qid: set(group.cid) for qid, group in frame.groupby("qid", sort=False)}


def evaluate_candidates(args):
    candidate_paths = sorted(Path(args.candidates_dir).glob("dense_candidates_*.parquet"))
    if not candidate_paths:
        raise ValueError("No dense candidate shards found")
    dense = pd.concat(
        [pd.read_parquet(path, columns=["qid", "cid"]) for path in candidate_paths],
        ignore_index=True,
    ).drop_duplicates(["qid", "cid"])
    dense_sets = _sets(dense)
    query_ids = set(dense_sets)
    truth = {}
    truth_path = Path(args.data_dir) / "train" / "train_ground_truth.tsv"
    for row in rows(truth_path):
        if row["source1_entity_id"] in query_ids:
            truth[row["source1_entity_id"]] = id_set(row["matched_entity_ids"])
    if set(truth) != query_ids:
        raise ValueError("Dense query IDs and training truth do not agree")
    query_records = {}
    for row in rows(Path(args.data_dir) / "train" / "train_source1.tsv"):
        if row["entity_id"] in query_ids:
            query_records[row["entity_id"]] = row

    result = {"dense": evaluate(truth, {}, dense_sets)}
    result["dense"]["mean_candidates_s1"] = len(dense) / max(1, len(truth))
    if args.lexical_candidates:
        lexical = pd.read_parquet(args.lexical_candidates, columns=["qid", "cid"])
        lexical = lexical[lexical.qid.isin(query_ids)].drop_duplicates(["qid", "cid"])
        lexical_sets = _sets(lexical)
        union_sets = {
            qid: dense_sets.get(qid, set()) | lexical_sets.get(qid, set()) for qid in truth
        }
        result["lexical"] = evaluate(truth, {}, lexical_sets)
        result["lexical"]["mean_candidates_s1"] = len(lexical) / max(1, len(truth))
        result["union"] = evaluate(truth, {}, union_sets)
        result["union"]["mean_candidates_s1"] = sum(map(len, union_sets.values())) / max(
            1, len(truth)
        )
    for label, candidate_set in [("dense", dense_sets)]:
        result[label]["by_country_candidate_recall"] = {
            country: evaluate(
                {qid: targets for qid, targets in truth.items() if query_records[qid]["country"] == country},
                {},
                candidate_set,
            )["candidate_recall"]
            for country in sorted({query_records[qid]["country"] for qid in truth})
        }
        result[label]["by_source_candidate_recall"] = {
            prefix: evaluate(
                {qid: {target for target in targets if target.startswith(prefix)} for qid, targets in truth.items()},
                {},
                {
                    qid: {target for target in candidate_set.get(qid, set()) if target.startswith(prefix)}
                    for qid in truth
                },
            )["candidate_recall"]
            for prefix in ["S2-", "S3-"]
        }
    dump_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


def parser():
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    encode = commands.add_parser("encode-corpus")
    encode.add_argument("--data-dir", required=True)
    encode.add_argument("--output-dir", required=True)
    encode.add_argument("--split", choices=["train", "test"], default="train")
    encode.add_argument("--model", default="intfloat/multilingual-e5-small")
    encode.add_argument("--revision", required=True)
    encode.add_argument("--device", default="cuda")
    encode.add_argument("--batch-size", type=int, default=512)
    encode.add_argument("--shard-rows", type=int, default=100000)
    encode.set_defaults(func=encode_corpus)

    build = commands.add_parser("build-index")
    build.add_argument("--embedding-dir", required=True)
    build.add_argument("--index-path", required=True)
    build.add_argument("--factory", default="IVF32768,PQ48x8")
    build.add_argument("--training-vectors", type=int, default=1000000)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--gpu", action="store_true")
    build.add_argument("--gpu-device", type=int, default=0)
    build.set_defaults(func=build_index)

    query = commands.add_parser("query")
    query.add_argument("--data-dir", required=True)
    query.add_argument("--index-path", required=True)
    query.add_argument("--output-dir", required=True)
    query.add_argument("--split", choices=["train", "test"], default="train")
    query.add_argument("--model", default="intfloat/multilingual-e5-small")
    query.add_argument("--revision", required=True)
    query.add_argument("--device", default="cuda")
    query.add_argument("--batch-size", type=int, default=512)
    query.add_argument("--shard-rows", type=int, default=100000)
    query.add_argument("--top-k", type=int, default=20)
    query.add_argument("--nprobe", type=int, default=64)
    query.add_argument("--sample-entities", type=int, default=50000)
    query.add_argument("--seed", type=int, default=42)
    query.add_argument("--gpu", action="store_true")
    query.add_argument("--gpu-device", type=int, default=0)
    query.set_defaults(func=query_index)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--data-dir", required=True)
    evaluate_parser.add_argument("--candidates-dir", required=True)
    evaluate_parser.add_argument("--lexical-candidates")
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.set_defaults(func=evaluate_candidates)
    return root


def main():
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
