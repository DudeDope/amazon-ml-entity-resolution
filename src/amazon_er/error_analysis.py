import json
from collections import Counter

import pandas as pd

from .normalization import numbers


def analyze(entities, truth, predictions, candidates, frame, blocker, outdir):
    categories = Counter()
    examples = []
    singleton = []
    probabilities = {(q, c): float(p) for q, c, p in zip(frame.qid, frame.cid, frame.probability)}
    for eid, actual in truth.items():
        pred = predictions.get(eid, set())
        cand = candidates.get(eid, set())
        q = entities[eid]
        scores = sorted([probabilities.get((eid, c), 0) for c in cand], reverse=True)
        singleton.append(
            {
                "qid": eid,
                "singleton": not actual,
                "country": q["country"],
                "top1": scores[0] if scores else 0,
                "margin": scores[0] - (scores[1] if len(scores) > 1 else 0) if scores else 0,
                "n_above_05": sum(p >= 0.5 for p in scores),
                "true_count": len(actual),
                "pred_count": len(pred),
            }
        )
        if actual and not pred:
            categories["false_singleton"] += 1
        if not actual and pred:
            categories["false_merge_singleton"] += 1
        if len(actual) > 1 and actual & pred and actual - pred:
            categories["multi_match_partial"] += 1
        for cid in (actual - pred) | (pred - actual):
            r = blocker.get(cid)
            if r is None:
                raise ValueError("Ground truth references unknown target " + cid)
            kind = (
                "false_positive"
                if cid in pred
                else ("blocking_failure" if cid not in cand else "matching_failure")
            )
            tags = [kind]
            if q["name"] == r["name"]:
                tags.append("exact_name")
            if q["address"] == r["address"]:
                tags.append("exact_address")
            if set(numbers(q["address"])) != set(numbers(r["address"])):
                tags.append("number_mismatch")
            if q["core"] == r["core"] and q["name"] != r["name"]:
                tags.append("legal_suffix")
            if sorted(q["name"].split()) == sorted(r["name"].split()) and q["name"] != r["name"]:
                tags.append("reordered_tokens")
            if q["country"] != r["country"]:
                tags.append("country_mismatch")
            for tag in tags:
                categories[tag] += 1
            if sum(e["kind"] == kind for e in examples) < 20:
                examples.append(
                    {
                        "kind": kind,
                        "tags": tags,
                        "qid": eid,
                        "cid": cid,
                        "score": probabilities.get((eid, cid)),
                        "query_name": q["business_name"],
                        "target_name": r["business_name"],
                        "query_address": q["business_address"],
                        "target_address": r["business_address"],
                    }
                )
    pd.DataFrame(singleton).to_csv(outdir / "singleton_analysis.csv", index=False)
    pd.DataFrame(examples).to_csv(outdir / "error_examples.csv", index=False)
    lines = [
        "# V1 holdout error analysis",
        "",
        "Tags overlap; categories with no count were not observed. Text-pattern tags are heuristics, not verified causes.",
        "",
        "```json",
        json.dumps(dict(categories), indent=2),
        "```",
        "",
        "Candidate failure is measured after the final TF-IDF candidate budget; it includes all earlier retrieval losses.",
        "",
        "## Record examples",
        "",
    ]
    for e in examples[:24]:
        lines += ["```json", json.dumps(e, ensure_ascii=False, indent=2), "```", ""]
    (outdir / "v1_error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return dict(categories)
