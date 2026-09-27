"""Hash sampling is label-independent; all pairs of an S1 share a split."""

from .data_io import stable_hash


def split_for(entity_id, seed=42):
    value = stable_hash(f"split:{seed}:{entity_id}") % 100
    return "fit" if value < 60 else ("calibration" if value < 80 else "holdout")


def sampled(entity_id, population, sample_size, seed=42):
    return stable_hash(f"sample:{seed}:{entity_id}") / (1 << 63) < min(1, sample_size / population)
