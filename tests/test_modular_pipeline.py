from pathlib import Path

import numpy as np
import polars as pl

from amazon_er.advanced_features import build_pair_features, normalize_records
from amazon_er.advanced_submission import write_submission
from amazon_er.candidates.fusion import CandidateSchemaError, fuse_candidates
from amazon_er.dataset import attach_labels, prepare_split, truth_to_pairs
from amazon_er.decision.decoder import decode_pairs
from amazon_er.decision.expected_f import expected_f_select
from amazon_er.evaluation.metrics import entity_fbeta, evaluate_sets
from amazon_er.evaluation.splits import assign_group_folds
from amazon_er.graph.matching import target_exclusive_mask
from amazon_er.models.hard_negatives import mine_hard_negatives


def test_normalization_and_pair_features_support_raw_competition_columns():
    queries = pl.DataFrame(
        {"q": ["S1-1"], "business_name": ["श्री गणेश"], "business_address": ["12 MG Road"], "country": ["IN"]}
    )
    targets = pl.DataFrame(
        {"t": ["S2-1"], "business_name": ["Shri Ganesh"], "business_address": ["12, M.G. Road"], "country": ["IN"]}
    )
    q_norm = normalize_records(queries, "q")
    assert {"name_norm", "name_roman", "address_norm", "script"}.issubset(q_norm.columns)
    result = build_pair_features(pl.DataFrame({"q": ["S1-1"], "t": ["S2-1"]}), queries, targets)
    assert result["address_digit_jaccard"][0] == 1.0
    assert result["same_country"][0] == 1.0
    assert result["name_roman_fuzzy"][0] > 0.5


def test_candidate_fusion_is_unique_ranked_and_validated():
    first = pl.DataFrame(
        {"q": ["q1", "q1"], "t": ["t1", "t2"], "retrieval_source": ["a", "a"], "rank": [1, 2], "retrieval_score": [0.9, 0.8]}
    )
    second = pl.DataFrame(
        {"q": ["q1", "q1"], "t": ["t2", "t3"], "retrieval_source": ["b", "b"], "rank": [1, 2], "retrieval_score": [0.95, 0.7]}
    )
    fused = fuse_candidates([first, second], top_k=2, rrf_k=10)
    assert fused.height == 2
    assert fused["t"].to_list()[0] == "t2"
    assert fused["source_count"].to_list()[0] == 2
    bad = first.with_columns(pl.lit(0).alias("rank"))
    try:
        fuse_candidates([bad], top_k=2)
    except CandidateSchemaError:
        pass
    else:
        raise AssertionError("rank zero should be rejected")


def test_group_folds_never_split_a_query():
    frame = pl.DataFrame({"q": ["a", "a", "b", "b"], "t": ["1", "2", "3", "4"]})
    folded = assign_group_folds(frame, folds=3, seed=7)
    assert folded.group_by("q").agg(pl.col("fold").n_unique())["fold"].max() == 1


def test_expected_f_and_target_exclusivity():
    q = np.asarray(["q1", "q1", "q2"], dtype=object)
    t = np.asarray(["t1", "t2", "t1"], dtype=object)
    score = np.asarray([0.9, 0.1, 0.8])
    exclusive = target_exclusive_mask(q, t, score)
    assert exclusive.tolist() == [True, True, False]
    selected = expected_f_select(q, np.where(exclusive, score, 0.0), k_max=3)
    assert selected.tolist() == [True, False, False]
    assert expected_f_select(np.asarray([]), np.asarray([])).size == 0


def test_decoder_and_submission_include_empty_queries(tmp_path: Path):
    scores = pl.DataFrame({"q": ["S1-1", "S1-2"], "t": ["S2-1", "S2-2"], "score": [0.99, 0.01]})
    selected = decode_pairs(scores, strategy="threshold_exclusive", threshold=0.8)
    matching, candidates = write_submission(
        pl.DataFrame({"q": ["S1-1", "S1-2", "S1-3"]}), scores, selected, tmp_path
    )
    lines = matching.read_text(encoding="utf-8").splitlines()
    assert lines == ["source1_entity_id\tmatched_entity_ids", "S1-1\tS2-1", "S1-2\t", "S1-3\t"]
    assert candidates.exists()


def test_hard_negative_sources_cover_query_and_target_confusion():
    pairs = pl.DataFrame(
        {"q": ["a", "a", "b"], "t": ["x", "y", "x"], "label": [0, 0, 0], "score": [0.9, 0.8, 0.85]}
    )
    result = mine_hard_negatives(pairs, per_query=1, per_target=1)
    assert result.height >= 2
    assert set(result["negative_source"].to_list()) <= {"query_confusion", "target_competition"}


def test_metrics_empty_entity_is_perfect():
    assert entity_fbeta(set(), set()) == 1.0
    metrics = evaluate_sets({"q1": {"t1"}, "q2": set()}, {"q1": {"t1"}, "q2": set()})
    assert metrics["macro_fbeta"] == 1.0


def test_truth_and_competition_tsv_preparation(tmp_path: Path):
    data = tmp_path / "dataset" / "train"
    data.mkdir(parents=True)
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    (data / "train_source1.tsv").write_text(header + "S1-1\tAlpha\tOne Road\tUS\n", encoding="utf-8")
    (data / "train_source2.tsv").write_text(header + "S2-1\tAlpha\tOne Rd\tUS\n", encoding="utf-8")
    (data / "train_source3.tsv").write_text(header + "S3-1\tBeta\tTwo Rd\tUS\n", encoding="utf-8")
    (data / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\n", encoding="utf-8"
    )
    paths = prepare_split(tmp_path / "dataset", tmp_path / "prepared", "train")
    assert set(paths) == {"queries", "targets", "truth"}
    assert pl.read_parquet(paths["targets"]).height == 2
    truth = truth_to_pairs(pl.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": ["S2-1,S3-1"]}))
    features = pl.DataFrame({"q": ["S1-1", "S1-1"], "t": ["S2-1", "S2-X"]})
    assert attach_labels(features, truth)["label"].to_list() == [1, 0]
