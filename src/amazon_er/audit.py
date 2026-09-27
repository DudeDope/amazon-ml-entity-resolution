"""Full-file streaming audit; bounded memory plus compact ID/hash arrays."""

import json
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .data_io import chunks, dump_json


def quantiles(hist):
    if not hist:
        return {}
    values = np.array(sorted(hist))
    counts = np.array([hist[x] for x in values])
    cum = counts.cumsum()
    return {
        str(q): int(values[min(len(values) - 1, np.searchsorted(cum, q * cum[-1]))])
        for q in [0, 0.5, 0.9, 0.95, 0.99, 1]
    }


def run_audit(cfg):
    report = {}
    start = time.time()
    for path in sorted(Path(cfg["data_dir"]).rglob("*.tsv")):
        name = str(path.relative_to(cfg["data_dir"]))
        out = {"rows": 0, "country": {}, "missing": {}, "text": {}}
        ids = []
        hashes = []
        countries = Counter()
        hist = {}
        unicode_chars = Counter()
        sizes = Counter()
        targets = []
        for frame in chunks(path, cfg["chunk_size"]):
            out["rows"] += len(frame)
            idcol = "entity_id" if "entity_id" in frame else "source1_entity_id"
            if not frame[idcol].str.fullmatch(r"S[123]-(0|[1-9][0-9]{0,8})").all():
                raise ValueError(
                    "Audit integer ID encoding requires canonical S1/S2/S3 numeric IDs under 1 billion"
                )
            ids.append(frame[idcol].str.split("-").str[-1].astype("int64").to_numpy())
            hashes.append(pd.util.hash_pandas_object(frame, index=False).to_numpy())
            for col in frame:
                out["missing"][col] = out["missing"].get(col, 0) + int(frame[col].str.strip().eq("").sum())
            if "country" in frame:
                countries.update(frame.country.value_counts().to_dict())
                for col in ["business_name", "business_address"]:
                    s = frame[col]
                    h = hist.setdefault(col, {"chars": Counter(), "tokens": Counter()})
                    h["chars"].update(s.str.len().value_counts().to_dict())
                    h["tokens"].update(s.str.count(r"\S+").value_counts().to_dict())
                    stats = out["text"].setdefault(col, Counter())
                    for label, pattern in [
                        ("non_ascii", r"[^\x00-\x7f]"),
                        ("punctuation", r"[^\w\s]"),
                        ("numbers", r"\d"),
                        ("accents", r"[\u00c0-\u024f\u0300-\u036f]"),
                    ]:
                        stats[label] += int(s.str.contains(pattern, regex=True).sum())
                    for value in s[s.str.contains(r"[^\x00-\x7f]", regex=True)]:
                        unicode_chars.update(c for c in value if ord(c) > 127)
            else:
                sizes.update(
                    frame.matched_entity_ids.map(lambda s: s.count(",") + 1 if s else 0)
                    .value_counts()
                    .to_dict()
                )
                for s in frame.matched_entity_ids:
                    if s:
                        # Encode source in integer target IDs to preserve S2/S3 namespace.
                        values = s.split(",")
                        if len(values) != len(set(values)):
                            raise ValueError("Duplicate ground-truth target within an entity")
                        targets.extend((int(v[1]) * 10**10 + int(v[3:])) for v in s.split(","))
                out["both_sources"] = out.get("both_sources", 0) + int(
                    (
                        frame.matched_entity_ids.str.contains("S2-")
                        & frame.matched_entity_ids.str.contains("S3-")
                    ).sum()
                )
        all_ids = np.concatenate(ids)
        out["unique_ids"] = int(len(np.unique(all_ids)))
        out["duplicate_ids"] = out["rows"] - out["unique_ids"]
        out["duplicate_rows_hash_based"] = out["rows"] - len(np.unique(np.concatenate(hashes)))
        out["country"] = dict(countries)
        for col, h in hist.items():
            out["text"][col] = dict(out["text"][col])
            out["text"][col]["length_quantiles"] = quantiles(h["chars"])
            out["text"][col]["token_quantiles"] = quantiles(h["tokens"])
        out["non_ascii_characters"] = [
            {"character": c, "name": unicodedata.name(c, "UNNAMED"), "count": n}
            for c, n in unicode_chars.most_common(25)
        ]
        if sizes:
            out["match_cardinality"] = dict(sorted(sizes.items()))
            out["max_cardinality"] = max(sizes)
            out["singleton_fraction"] = sizes[0] / out["rows"]
            out["both_sources_fraction"] = out["both_sources"] / out["rows"]
            t = np.asarray(targets, dtype=np.int64)
            unique, counts = np.unique(t, return_counts=True)
            out["links"] = len(t)
            out["targets_linked_multiple_times"] = int((counts > 1).sum())
            out["max_target_frequency"] = int(counts.max(initial=0))
            out["s2_links"] = int((t // 10**10 == 2).sum())
            out["s3_links"] = int((t // 10**10 == 3).sum())
            out["target_frequency_histogram"] = {str(k): int(v) for k, v in Counter(counts).items()}
            del t, unique, counts, targets
        report[name] = out
        print("AUDIT", name, out["rows"], out["country"], "elapsed", round(time.time() - start), flush=True)
        dump_json(Path(cfg["work_dir"]) / "reports/data_audit.json", report)
    lines = [
        "# Full dataset audit",
        "",
        "All TSVs read with explicit tab separator and strings; blanks preserved.",
        "Duplicate-row counts use 64-bit pandas row hashes (collision risk); ID duplicate counts are exact.",
        "Accent counts cover Latin accented characters/combining marks; all non-ASCII scripts are counted separately.",
        "",
    ]
    for name, out in report.items():
        lines += ["## " + name, "", "```json", json.dumps(out, indent=2, ensure_ascii=False), "```", ""]
    (Path(cfg["work_dir"]) / "reports/data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return report
