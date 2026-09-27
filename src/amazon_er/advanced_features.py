"""Structured, multilingual pair features for the modular pipeline."""

from __future__ import annotations

import re
from collections.abc import Sequence

import polars as pl
from rapidfuzz.fuzz import ratio, token_set_ratio

from .candidates.transliteration import normalize_native, romanize_anyascii, script_group

DEFAULT_FIELDS = ("name", "address", "phone", "email", "website")
FIELD_ALIASES = {"name": "business_name", "address": "business_address"}
_DIGITS = re.compile(r"\d+")
_TOKENS = re.compile(r"\w+", flags=re.UNICODE)


def normalize_records(
    frame: pl.DataFrame,
    id_column: str,
    fields: Sequence[str] = DEFAULT_FIELDS,
) -> pl.DataFrame:
    """Add native and romanized views while preserving every original column."""
    if id_column not in frame.columns:
        raise ValueError(f"Missing ID column: {id_column}")
    result = frame
    for canonical, source in FIELD_ALIASES.items():
        if canonical not in result.columns and source in result.columns:
            result = result.with_columns(pl.col(source).alias(canonical))
    present = [field for field in fields if field in result.columns]
    for field in present:
        result = result.with_columns(
            pl.col(field)
            .cast(pl.String)
            .fill_null("")
            .map_elements(normalize_native, return_dtype=pl.String)
            .alias(f"{field}_norm"),
            pl.col(field)
            .cast(pl.String)
            .fill_null("")
            .map_elements(lambda value: romanize_anyascii(normalize_native(value)), return_dtype=pl.String)
            .alias(f"{field}_roman"),
        )
    if "name" in result.columns:
        result = result.with_columns(
            pl.col("name")
            .cast(pl.String)
            .fill_null("")
            .map_elements(script_group, return_dtype=pl.String)
            .alias("script")
        )
    combined = [f"{field}_norm" for field in present]
    romanized = [f"{field}_roman" for field in present]
    if combined:
        result = result.with_columns(
            pl.concat_str(combined, separator=" ").str.strip_chars().alias("combined_norm"),
            pl.concat_str(romanized, separator=" ").str.strip_chars().alias("combined_roman"),
        )
    return result


def _tokens(value: object) -> set[str]:
    return set(_TOKENS.findall(str(value or "")))


def _digits(value: object) -> set[str]:
    return set(_DIGITS.findall(str(value or "")))


def _safe_ratio(left: object, right: object) -> float:
    a, b = str(left or ""), str(right or "")
    return float(ratio(a, b) / 100.0) if a and b else 0.0


def _token_ratio(left: object, right: object) -> float:
    a, b = str(left or ""), str(right or "")
    return float(token_set_ratio(a, b) / 100.0) if a and b else 0.0


def _jaccard(left: object, right: object) -> float:
    a, b = _tokens(left), _tokens(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _digit_jaccard(left: object, right: object) -> float:
    a, b = _digits(left), _digits(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _length_ratio(left: object, right: object) -> float:
    a, b = len(str(left or "")), len(str(right or ""))
    return min(a, b) / max(a, b) if max(a, b) else 0.0


def _pair_struct(left: str, right: str) -> dict[str, float]:
    return {
        "fuzzy": _safe_ratio(left, right),
        "token_set": _token_ratio(left, right),
        "token_jaccard": _jaccard(left, right),
        "digit_jaccard": _digit_jaccard(left, right),
        "length_ratio": _length_ratio(left, right),
        "exact": float(bool(left) and left == right),
    }


def build_pair_features(
    pairs: pl.DataFrame,
    queries: pl.DataFrame,
    targets: pl.DataFrame,
    query_id: str = "q",
    target_id: str = "t",
    fields: Sequence[str] = DEFAULT_FIELDS,
) -> pl.DataFrame:
    """Join candidate pairs to records and calculate deterministic similarity features.

    Retrieval columns already present in ``pairs`` are retained. Records may be raw or may have
    been passed through :func:`normalize_records`; raw records are normalized automatically.
    """
    if not {"q", "t"}.issubset(pairs.columns):
        raise ValueError("pairs must contain q and t")
    if query_id not in queries.columns or target_id not in targets.columns:
        raise ValueError("query/target ID columns are missing")
    q_frame = queries if "combined_norm" in queries.columns else normalize_records(queries, query_id, fields)
    t_frame = targets if "combined_norm" in targets.columns else normalize_records(targets, target_id, fields)
    present = [field for field in fields if f"{field}_norm" in q_frame.columns and f"{field}_norm" in t_frame.columns]
    q_select = [pl.col(query_id).alias("q")]
    t_select = [pl.col(target_id).alias("t")]
    for field in present:
        q_select.extend([pl.col(f"{field}_norm").alias(f"{field}_q"), pl.col(f"{field}_roman").alias(f"{field}_roman_q")])
        t_select.extend([pl.col(f"{field}_norm").alias(f"{field}_t"), pl.col(f"{field}_roman").alias(f"{field}_roman_t")])
    for name in ("country", "script"):
        if name in q_frame.columns:
            q_select.append(pl.col(name).alias(f"{name}_q"))
        if name in t_frame.columns:
            t_select.append(pl.col(name).alias(f"{name}_t"))
    joined = pairs.join(q_frame.select(q_select), on="q", how="left", validate="m:1")
    joined = joined.join(t_frame.select(t_select), on="t", how="left", validate="m:1")
    expressions = []
    feature_type = pl.Struct(
        [
            pl.Field("fuzzy", pl.Float64),
            pl.Field("token_set", pl.Float64),
            pl.Field("token_jaccard", pl.Float64),
            pl.Field("digit_jaccard", pl.Float64),
            pl.Field("length_ratio", pl.Float64),
            pl.Field("exact", pl.Float64),
        ]
    )
    for field in present:
        for suffix in ("", "_roman"):
            left, right = f"{field}{suffix}_q", f"{field}{suffix}_t"
            name = f"{field}{suffix}"
            expressions.append(
                pl.struct(left, right)
                .map_elements(lambda row, l=left, r=right: _pair_struct(row[l], row[r]), return_dtype=feature_type)
                .alias(f"_{name}")
            )
    result = joined.with_columns(expressions) if expressions else joined
    for field in present:
        for suffix in ("", "_roman"):
            name = f"{field}{suffix}"
            result = result.unnest(f"_{name}").rename(
                {metric: f"{name}_{metric}" for metric in ("fuzzy", "token_set", "token_jaccard", "digit_jaccard", "length_ratio", "exact")}
            )
    if {"country_q", "country_t"}.issubset(result.columns):
        result = result.with_columns(
            (pl.col("country_q").fill_null("") == pl.col("country_t").fill_null(""))
            .cast(pl.Float32)
            .alias("same_country")
        )
    if {"script_q", "script_t"}.issubset(result.columns):
        result = result.with_columns(
            (pl.col("script_q") == pl.col("script_t")).cast(pl.Float32).alias("same_script")
        )
    numeric = [name for name, dtype in result.schema.items() if dtype.is_numeric()]
    floats = [name for name, dtype in result.schema.items() if dtype in (pl.Float32, pl.Float64)]
    if floats:
        result = result.with_columns(pl.col(floats).fill_nan(0))
    return result.with_columns(pl.col(numeric).fill_null(0)) if numeric else result


def numeric_feature_columns(frame: pl.DataFrame) -> list[str]:
    """Return model-safe numeric columns while excluding IDs and labels."""
    excluded = {"label", "y", "fold"}
    return [
        name
        for name, dtype in frame.schema.items()
        if name not in excluded and name not in {"q", "t"} and dtype.is_numeric()
    ]
