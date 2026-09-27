from pathlib import Path

import pandas as pd

from .data_io import chunks, id_set, rows
from .features import FeatureComputer
from .normalization import prepare
from .validation import sampled, split_for


def sample_entities(cfg):
    path = Path(cfg["data_dir"]) / "train/train_source1.tsv"
    population = sum(len(c) for c in chunks(path))
    entities = {}
    for row in rows(path):
        if sampled(row["entity_id"], population, cfg["sample_entities"], cfg["seed"]):
            entities[row["entity_id"]] = prepare(row)
    truth = {}
    for row in rows(Path(cfg["data_dir"]) / "train/train_ground_truth.tsv"):
        if row["source1_entity_id"] in entities:
            if row["source1_entity_id"] in truth:
                raise ValueError("Duplicate ground truth entity")
            truth[row["source1_entity_id"]] = id_set(row["matched_entity_ids"])
    if set(truth) != set(entities):
        raise ValueError("Ground truth does not cover sample")
    splits = {eid: split_for(eid, cfg["seed"]) for eid in entities}
    return entities, truth, splits


def build_pairs(entities, truth, blocker, vectorizers, cfg):
    frames = []
    stats = []
    records = list(entities.values())
    size = cfg["inference_batch_size"]
    with FeatureComputer(blocker, vectorizers, cfg) as computer:
        for start in range(0, len(records), size):
            frame, stat = computer.compute(records[start : start + size])
            if len(frame):
                frame["label"] = [int(c in truth[q]) for q, c in zip(frame.qid, frame.cid)]
                frames.append(frame)
            stats.append(stat)
            print(f"Pairs: {min(start + size, len(records)):,}/{len(records):,} entities", flush=True)
    return pd.concat(frames, ignore_index=True), pd.concat(stats, ignore_index=True)
