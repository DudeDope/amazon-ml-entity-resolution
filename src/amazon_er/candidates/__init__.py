"""Candidate generation and fusion interfaces."""

from .fusion import CandidateSchemaError, fuse_candidates
from .transliteration import script_group, text_variants

__all__ = ["CandidateSchemaError", "fuse_candidates", "script_group", "text_variants"]
