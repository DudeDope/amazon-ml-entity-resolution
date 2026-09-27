"""Collective graph features and deterministic target constraints."""

from .collective import add_collective_features
from .matching import target_exclusive_mask

__all__ = ["add_collective_features", "target_exclusive_mask"]
