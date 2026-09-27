"""Country-independent string, numeric, retrieval and character TF-IDF features."""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalization import basic, numbers


def fit_vectorizers(blocker, cfg):
    total = blocker.con.execute("SELECT max(rowid) FROM records").fetchone()[0]
    rng = np.random.default_rng(cfg["seed"])
    ids = sorted(rng.choice(total, size=min(total, cfg["tfidf_sample"]), replace=False) + 1)
    records = [
        blocker.con.execute("SELECT name,address FROM records WHERE rowid=?", (int(i),)).fetchone()
        for i in ids
    ]
    result = {}
    for i, field in enumerate(["name", "address"]):
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=tuple(cfg["ngram_range"]),
            max_features=cfg["tfidf_max_features"],
            dtype=np.float32,
            sublinear_tf=True,
        )
        vec.fit([r[i] for r in records] + ["emptyfallback"])
        result[field] = vec
    return result


def overlaps(a, b):
    a, b = set(a), set(b)
    return (len(a & b) / max(1, len(a | b)), len(a & b) / max(1, min(len(a), len(b))))


def lexical_features(q, r):
    out = {}
    for field in ["name", "address"]:
        a, b = q[field], r[field]
        at, bt = a.split(), b.split()
        jac, overlap = overlaps(at, bt)
        out.update(
            {
                field + "_exact": int(bool(a) and a == b),
                field + "_missing": int(not a or not b),
                field + "_jaccard": jac,
                field + "_overlap": overlap,
                field + "_ratio": fuzz.ratio(a, b) / 100 if a and b else 0,
                field + "_partial": fuzz.partial_ratio(a, b) / 100 if a and b else 0,
                field + "_token_sort": fuzz.token_sort_ratio(a, b) / 100 if a and b else 0,
                field + "_token_set": fuzz.token_set_ratio(a, b) / 100 if a and b else 0,
                field + "_edit": Levenshtein.normalized_similarity(a, b) if a and b else 0,
                field + "_jaro": JaroWinkler.normalized_similarity(a, b) if a and b else 0,
                field + "_length_ratio": min(len(a), len(b)) / max(1, len(a), len(b)),
                field + "_token_difference": abs(len(at) - len(bt)),
                field + "_prefix": int(bool(a) and bool(b) and a[:4] == b[:4]),
                field + "_suffix": int(bool(a) and bool(b) and a[-4:] == b[-4:]),
            }
        )
    nq, nr = set(numbers(q["address"])), set(numbers(r["address"]))
    pq, pr = {x for x in nq if len(x) >= 4}, {x for x in nr if len(x) >= 4}
    aq = "".join(t[0] for t in q["core"].split())
    ar = "".join(t[0] for t in r["core"].split())
    out.update(
        core_exact=int(bool(q["core"]) and q["core"] == r["core"]),
        unicode_name_exact=int(
            bool(q["business_name"]) and basic(q["business_name"]) == basic(r["business_name"])
        ),
        acronym_equal=int(bool(aq) and aq == ar),
        numeric_jaccard=overlaps(nq, nr)[0],
        numeric_disagreement=len(nq ^ nr),
        numeric_missing=int(not nq or not nr),
        postal_overlap=len(pq & pr),
        postal_conflict=int(bool(pq) and bool(pr) and not pq & pr),
        same_country=int(basic(q["country"]) == basic(r["country"])),
        source3=int(r["entity_id"].startswith("S3-")),
        retrieval_count=int(r["block_mask"]).bit_count(),
    )
    out["name_address_product"] = out["name_ratio"] * out["address_ratio"]
    out["name_address_min"] = min(out["name_ratio"], out["address_ratio"])
    out["name_strong_address_weak"] = int(out["name_ratio"] > 0.9 and out["address_ratio"] < 0.5)
    for view in ["name", "address", "combined"]:
        out["retrieval_" + view] = r["retrieval_" + view]
        out["rank_" + view] = r["rank_" + view]
    for i in range(11):
        out[f"block_{i}"] = int(bool(r["block_mask"] & (1 << i)))
    return out


def make_features(queries, blocker, vectorizers, cfg):
    pairs = []
    stats = []
    qmap = {q["entity_id"]: q for q in queries}
    rmap = {}
    for q in queries:
        records, stat = blocker.retrieve(q)
        stats.append({"qid": q["entity_id"], **stat})
        for r in records:
            # Retrieval metadata belongs to the pair, not the target record.
            pairs.append((q, r))
            rmap[r["entity_id"]] = r
    if not pairs:
        return pd.DataFrame(), pd.DataFrame(stats)
    features = pd.DataFrame([lexical_features(q, r) for q, r in pairs], dtype=np.float32)
    if not cfg.get("composite_blocks", False):
        features = features.drop(columns=[f"block_{i}" for i in range(5, 11)])
    features.insert(0, "qid", [q["entity_id"] for q, r in pairs])
    features.insert(1, "cid", [r["entity_id"] for q, r in pairs])
    qlist = list(qmap)
    rlist = list(rmap)
    qi = {x: i for i, x in enumerate(qlist)}
    ri = {x: i for i, x in enumerate(rlist)}
    ix = np.array([qi[q["entity_id"]] for q, r in pairs])
    jx = np.array([ri[r["entity_id"]] for q, r in pairs])
    for field, vec in vectorizers.items():
        qa = vec.transform([qmap[x][field] for x in qlist])
        ra = vec.transform([rmap[x][field] for x in rlist])
        features[field + "_tfidf"] = np.asarray(qa[ix].multiply(ra[jx]).sum(axis=1)).ravel()
    features["combined_tfidf"] = (features.name_tfidf + features.address_tfidf) / 2
    return select_candidates(features, cfg["top_k"]), pd.DataFrame(stats)


def select_candidates(features, k, views=None, include_exact=True):
    """Final candidate union; uses no labels and permits arbitrary cardinality."""
    if features.empty:
        return features
    views = views or ["name_tfidf", "address_tfidf", "combined_tfidf"]
    keep = set()
    for field in views:
        ranked = features.sort_values(["qid", field, "cid"], ascending=[True, False, True])
        keep.update(ranked.groupby("qid", sort=False).head(k).index)
    if include_exact:
        keep.update(features.index[(features.name_exact == 1) & (features.address_exact == 1)])
    return features.loc[sorted(keep)].reset_index(drop=True)


_WORKER = None


def _initialize_worker(index_path, vectorizers, cfg):
    global _WORKER
    from threadpoolctl import threadpool_limits

    from .blocking import Blocker

    threadpool_limits(limits=1)
    _WORKER = (Blocker(index_path, cfg), vectorizers, cfg)


def _worker_features(queries):
    return make_features(queries, *_WORKER)


class FeatureComputer:
    """Persistent CPU workers with one read-only SQLite connection per process."""

    def __init__(self, blocker, vectorizers, cfg):
        self.blocker, self.vectorizers, self.cfg = blocker, vectorizers, cfg
        self.pool = None
        self.workers = cfg.get("workers", 1)
        if self.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import get_context

            self.pool = ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_initialize_worker,
                initargs=(blocker.path, vectorizers, cfg),
                mp_context=get_context("spawn"),
            )

    def compute(self, queries):
        if self.pool is None:
            return make_features(queries, self.blocker, self.vectorizers, self.cfg)
        size = max(1, (len(queries) + self.workers - 1) // self.workers)
        tasks = [
            self.pool.submit(_worker_features, queries[i : i + size]) for i in range(0, len(queries), size)
        ]
        results = [f.result() for f in tasks]
        return (
            pd.concat([r[0] for r in results], ignore_index=True),
            pd.concat([r[1] for r in results], ignore_index=True),
        )

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
