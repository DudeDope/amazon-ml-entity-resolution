"""Pair models, calibration, hard-negative mining, and optional neural reranking."""

from .hard_negatives import mine_hard_negatives

__all__ = ["mine_hard_negatives"]
