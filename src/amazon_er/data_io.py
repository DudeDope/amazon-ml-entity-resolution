import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


def config(path="configs/baseline.yaml"):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent.parent
    for key in ["data_dir", "output_dir", "work_dir"]:
        cfg[key] = str((base / cfg[key]).resolve())
    return cfg


def chunks(path, size=100000):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=size)


def rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def id_set(text):
    values = text.split(",") if text else []
    if len(values) != len(set(values)):
        raise ValueError("Duplicate IDs in label/list")
    return set(values)


def stable_hash(text):
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little") & (
        (1 << 63) - 1
    )


def fingerprint(paths, cfg):
    files = [(str(p), Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(json.dumps([files, cfg], sort_keys=True).encode()).hexdigest()


def dump_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
