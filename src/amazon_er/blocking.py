"""Disk-backed signatures and bounded, multiview lexical candidate retrieval.

This is approximate blocking. It is deliberately not described as exhaustive
global TF-IDF nearest-neighbour search. Recall loss is measured on held-out IDs.
"""

import sqlite3
import time
from pathlib import Path

from rapidfuzz import fuzz

from .data_io import fingerprint, rows, stable_hash
from .normalization import prepare

INDEX_VERSION = 1
COLS = ["entity_id", "business_name", "business_address", "country", "name", "address", "core", "digits"]
COMPOSITES = [
    (1, "k3"),
    (2, "k3"),
    (1, "substr(core,1,4)"),
    (2, "substr(core,1,4)"),
    (3, "substr(core,1,4)"),
    (4, "substr(core,1,4)"),
]


def add_composite_indexes(con, cfg):
    if not cfg.get("composite_blocks", False):
        return
    if con.execute("SELECT value FROM metadata WHERE key='composites_v1'").fetchone():
        return
    for j, (i, second) in enumerate(COMPOSITES):
        print(f"Building selective composite index {j + 1}/{len(COMPOSITES)}", flush=True)
        con.execute(f"CREATE INDEX IF NOT EXISTS composite{j} ON records(k{i},{second}) WHERE k{i} != -1")
        con.commit()
    con.execute("INSERT OR REPLACE INTO metadata VALUES ('composites_v1','complete')")
    con.commit()


def keys(record):
    tokens = sorted(set(record["core"].split()), key=lambda t: (-len(t), t))
    anchor = tokens[0] if tokens else ""
    address_words = sorted(
        set(t for t in record["address"].split() if t.isalpha()), key=lambda t: (-len(t), t)
    )
    strings = [
        " ".join(sorted(tokens)),
        anchor[:5] if len(anchor) >= 4 else "",
        anchor[-5:] if len(anchor) >= 4 else "",
        record["digits"],
        " ".join(sorted(address_words[:2])) if len(address_words) >= 2 else "",
    ]
    return tuple(stable_hash(f"{i}:{s}") if s else -1 for i, s in enumerate(strings))


def build_index(cfg, split):
    dest = Path(cfg["work_dir"]) / "artifacts" / f"{split}.sqlite"
    dest.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(cfg["data_dir"]) / split / f"{split}_source{s}.tsv" for s in [2, 3]]
    stamp = fingerprint(paths, {"version": INDEX_VERSION})
    con = sqlite3.connect(dest)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-262144")
    con.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT)")
    state = dict(con.execute("SELECT key,value FROM metadata"))
    if state.get("complete") == stamp:
        add_composite_indexes(con, cfg)
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.close()
        return dest
    if state.get("fingerprint") not in [None, stamp]:
        raise ValueError("Stale index: use a fresh work_dir; source/config fingerprint changed")
    con.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", ("fingerprint", stamp))
    con.execute(
        "CREATE TABLE IF NOT EXISTS records ("
        + ",".join(c + " TEXT" for c in COLS)
        + ","
        + ",".join(f"k{i} INTEGER" for i in range(5))
        + ")"
    )
    con.commit()
    for path in paths:
        if state.get(path.name) == "done":
            continue
        # A single transaction per source makes interruption resumable without partial duplicates.
        start = time.time()
        batch = []
        count = 0
        for raw in rows(path):
            rec = prepare(raw)
            if not rec["entity_id"].startswith("S" + path.stem[-1] + "-"):
                raise ValueError("Source ID prefix mismatch")
            batch.append(tuple(rec[c] for c in COLS) + keys(rec))
            if len(batch) >= cfg["index_batch_size"]:
                con.executemany(
                    "INSERT INTO records VALUES (" + ",".join("?" for _ in range(13)) + ")", batch
                )
                count += len(batch)
                batch = []
                if count % 250000 == 0:
                    print(
                        f"{split}: indexed {path.name} {count:,} rows in {time.time() - start:.0f}s",
                        flush=True,
                    )
        if batch:
            con.executemany("INSERT INTO records VALUES (" + ",".join("?" for _ in range(13)) + ")", batch)
        con.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (path.name, "done"))
        con.commit()
    for i in range(5):
        print(f"{split}: building signature index {i}", flush=True)
        con.execute(f"CREATE INDEX IF NOT EXISTS block{i} ON records(k{i}) WHERE k{i} != -1")
        con.commit()
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS record_id ON records(entity_id)")
    con.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", ("complete", stamp))
    con.commit()
    add_composite_indexes(con, cfg)
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    return dest


class Blocker:
    def __init__(self, path, cfg):
        self.cfg = cfg
        self.path = str(path)
        self.con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
        self.con.execute("PRAGMA cache_size=-262144")
        self.con.row_factory = sqlite3.Row
        if not self.con.execute("SELECT value FROM metadata WHERE key='complete'").fetchone():
            raise ValueError("Index incomplete")
        if (
            cfg.get("composite_blocks", False)
            and not self.con.execute("SELECT value FROM metadata WHERE key='composites_v1'").fetchone()
        ):
            raise ValueError("Composite indexes missing; run index construction first")

    def get(self, entity_id):
        row = self.con.execute("SELECT rowid,* FROM records WHERE entity_id=?", (entity_id,)).fetchone()
        return dict(row) if row else None

    def retrieve(self, query):
        pool = {}
        truncated = 0
        query_keys = keys(query)
        lookups = []
        for i, key in enumerate(query_keys):
            if key == -1:
                continue
            lookups.append((i, f"k{i}=? AND k{i} != -1", (key,)))
        if self.cfg.get("composite_blocks", False):
            for j, (i, second) in enumerate(COMPOSITES):
                value = query_keys[3] if second == "k3" else query["core"][:4]
                if query_keys[i] == -1 or value in [-1, ""]:
                    continue
                lookups.append((j + 5, f"k{i}=? AND k{i} != -1 AND {second}=?", (query_keys[i], value)))
        for i, condition, parameters in lookups:
            result = self.con.execute(
                f"SELECT rowid,* FROM records WHERE {condition} ORDER BY rowid LIMIT ?",
                (*parameters, self.cfg["block_limit"] + 1),
            ).fetchall()
            truncated += int(len(result) > self.cfg["block_limit"])
            for row in result[: self.cfg["block_limit"]]:
                rid = row["rowid"]
                if rid not in pool:
                    pool[rid] = dict(row)
                    pool[rid]["block_mask"] = 0
                pool[rid]["block_mask"] |= 1 << i
        records = list(pool.values())
        for r in records:
            r["retrieval_name"] = (
                fuzz.WRatio(query["name"], r["name"]) / 100 if query["name"] and r["name"] else 0
            )
            r["retrieval_address"] = (
                fuzz.WRatio(query["address"], r["address"]) / 100 if query["address"] and r["address"] else 0
            )
            r["retrieval_combined"] = (r["retrieval_name"] + r["retrieval_address"]) / 2
        # Independent views survive even when the other field is corrupt.
        selected = {}
        for view in ["name", "address", "combined"]:
            ranked = sorted(records, key=lambda r: (-r["retrieval_" + view], r["entity_id"]))
            for rank, r in enumerate(ranked, 1):
                r["rank_" + view] = rank
            for r in ranked[: self.cfg["prefilter_k"]]:
                selected[r["entity_id"]] = r
        for r in records:
            if (
                query["name"]
                and query["address"]
                and r["name"] == query["name"]
                and r["address"] == query["address"]
            ):
                selected[r["entity_id"]] = r
        return sorted(selected.values(), key=lambda r: r["entity_id"]), {
            "pool_size": len(records),
            "truncated_blocks": truncated,
        }

    def close(self):
        self.con.close()
