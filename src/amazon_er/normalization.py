"""Conservative Unicode and optional accent-folded representations."""

import re
import unicodedata

SUFFIXES = frozenset(
    "private pvt limited ltd incorporated inc corporation corp company co llc llp plc sarl sas sa".split()
)


def basic(value: object) -> str:
    if value is None or str(value).lower() in {"nan", "<na>"}:
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold().replace("&", " and ")
    return " ".join(re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE).split())


def accent_fold(value: object) -> str:
    # Preserve non-Latin scripts instead of deleting them via ASCII encoding.
    return "".join(c for c in unicodedata.normalize("NFKD", basic(value)) if not unicodedata.combining(c))


def aggressive(value: object) -> str:
    return " ".join(t for t in accent_fold(value).split() if t not in SUFFIXES)


def numbers(value: str) -> tuple[str, ...]:
    return tuple(sorted(set(re.findall(r"\d+", value))))


def prepare(row: dict) -> dict:
    row = dict(row)
    row["name"] = accent_fold(row["business_name"])
    row["address"] = accent_fold(row["business_address"])
    row["core"] = aggressive(row["business_name"])
    row["digits"] = " ".join(numbers(row["address"]))
    return row
