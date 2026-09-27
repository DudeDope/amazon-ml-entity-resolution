"""Standalone all-empty and conservative exact-match baselines."""

from datetime import datetime, timezone
from pathlib import Path

from .blocking import Blocker, build_index
from .data_io import dump_json
from .metric import evaluate
from .pair_dataset import sample_entities


def run_v0(cfg):
    work = Path(cfg["work_dir"])
    (work / "artifacts").mkdir(parents=True, exist_ok=True)
    blocker = Blocker(build_index(cfg, "train"), cfg)
    entities, truth, splits = sample_entities(cfg)
    actual = {q: t for q, t in truth.items() if splits[q] == "holdout"}
    predictions = {}
    candidates = {}
    examples = []
    for qid, targets in actual.items():
        q = entities[qid]
        records, _ = blocker.retrieve(q)
        candidates[qid] = {r["entity_id"] for r in records}
        predictions[qid] = {
            r["entity_id"]
            for r in records
            if q["name"]
            and q["address"]
            and q["name"] == r["name"]
            and q["address"] == r["address"]
            and q["country"] == r["country"]
        }
        if len(examples) < 24:
            for label, ids in [
                ("TP", targets & predictions[qid]),
                ("FP", predictions[qid] - targets),
                ("FN", targets - predictions[qid]),
            ]:
                for cid in sorted(ids)[:1]:
                    r = blocker.get(cid)
                    examples.append(
                        {
                            "kind": label,
                            "qid": qid,
                            "cid": cid,
                            "name1": q["business_name"],
                            "name2": r["business_name"],
                            "address1": q["business_address"],
                            "address2": r["business_address"],
                        }
                    )
    result = {
        "all_empty": evaluate(actual, {}),
        "exact": evaluate(actual, predictions, candidates),
        "examples": examples,
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = work / "artifacts" / ("v0_" + run_id)
    out.mkdir()
    dump_json(out / "metrics.json", result)
    dump_json(out / "predictions.json", {q: sorted(p) for q, p in predictions.items()})
    from .pipeline import record_experiment, save_report

    (work / "reports").mkdir(exist_ok=True)
    save_report(work / "reports/v0_results.md", "V0 standalone holdout", result)
    for exp, metrics in [("E000", result["all_empty"]), ("E001", result["exact"])]:
        record_experiment(cfg, run_id, exp, "holdout", metrics, None, "Standalone V0, no trained model")
    blocker.close()
    return result
