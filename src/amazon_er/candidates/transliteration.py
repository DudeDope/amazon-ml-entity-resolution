"""Unicode-preserving normalization and optional transliteration views."""

from __future__ import annotations

import re
import subprocess
import unicodedata
from functools import lru_cache
from shutil import which

from anyascii import anyascii

_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)

SCRIPT_RANGES = {
    "devanagari": ((0x0900, 0x097F),),
    "bengali": ((0x0980, 0x09FF),),
    "gurmukhi": ((0x0A00, 0x0A7F),),
    "gujarati": ((0x0A80, 0x0AFF),),
    "oriya": ((0x0B00, 0x0B7F),),
    "tamil": ((0x0B80, 0x0BFF),),
    "telugu": ((0x0C00, 0x0C7F),),
    "kannada": ((0x0C80, 0x0CFF),),
    "malayalam": ((0x0D00, 0x0D7F),),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F)),
    "latin": ((0x0041, 0x024F),),
}


def normalize_native(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("&", " and ")
    return _SPACE.sub(" ", _PUNCT.sub(" ", text)).strip()


def script_group(value: object) -> str:
    text = normalize_native(value)
    counts = {name: 0 for name in SCRIPT_RANGES}
    for char in text:
        codepoint = ord(char)
        for name, ranges in SCRIPT_RANGES.items():
            if any(start <= codepoint <= end for start, end in ranges):
                counts[name] += 1
                break
    nonzero = [(count, name) for name, count in counts.items() if count]
    if not nonzero:
        return "unknown"
    nonzero.sort(reverse=True)
    dominant = nonzero[0][1]
    represented = sum(1 for count, _ in nonzero if count > 0)
    return dominant if represented == 1 else f"mixed_{dominant}"


@lru_cache(maxsize=250_000)
def romanize_anyascii(text: str) -> str:
    return normalize_native(anyascii(normalize_native(text)))


@lru_cache(maxsize=100_000)
def romanize_uroman(text: str) -> str:
    executable = which("uroman")
    if not executable:
        raise RuntimeError("uroman executable is unavailable; install it or use method=anyascii")
    result = subprocess.run(
        [executable],
        input=normalize_native(text) + "\n",
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    return normalize_native(result.stdout.splitlines()[0] if result.stdout else "")


def text_variants(value: object, method: str = "anyascii", keep_native: bool = True) -> tuple[str, ...]:
    native = normalize_native(value)
    if not native:
        return ()
    if method == "none":
        romanized = native
    elif method == "anyascii":
        romanized = romanize_anyascii(native)
    elif method == "uroman":
        romanized = romanize_uroman(native)
    else:
        raise ValueError(f"Unknown transliteration method: {method}")
    values = [native] if keep_native else []
    if romanized and romanized not in values:
        values.append(romanized)
    compact = romanized.replace(" ", "")
    if compact and len(compact) >= 4 and compact not in values:
        values.append(compact)
    return tuple(values)

