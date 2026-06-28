"""Unit tests for the ranking-metrics module.

Hand-computed reference values check Precision/Recall/NDCG/HitRate against
small fixtures. Run with: pytest tests/test_metrics.py
"""
import math

import pytest

from src.metrics import (
    evaluate,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_perfect():
    assert precision_at_k([1, 2, 3], [1, 2, 3], k=3) == 1.0


def test_precision_at_k_partial():
    # 2 of 3 hits
    assert precision_at_k([1, 2, 99], [1, 2, 3], k=3) == pytest.approx(2 / 3)


def test_precision_at_k_zero_hits():
    assert precision_at_k([7, 8, 9], [1, 2, 3], k=3) == 0.0


def test_precision_at_k_truncates_to_k():
    # Recommender returns more than k items; only top-k count
    assert precision_at_k([1, 2, 99, 1, 2], [1, 2, 3], k=2) == 1.0


def test_recall_at_k_full_coverage():
    assert recall_at_k([1, 2, 3], [1, 2, 3], k=3) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k([1, 2], [1, 2, 3], k=2) == pytest.approx(2 / 3)


def test_recall_at_k_empty_relevant():
    assert recall_at_k([1, 2, 3], [], k=3) == 0.0


def test_hit_rate_at_k_hit():
    assert hit_rate_at_k([7, 8, 1], [1, 2, 3], k=3) == 1.0


def test_hit_rate_at_k_miss():
    assert hit_rate_at_k([7, 8, 9], [1, 2, 3], k=3) == 0.0


def test_ndcg_at_k_perfect_ranking():
    # All top-3 relevant in best order -> NDCG = 1.0
    assert ndcg_at_k([1, 2, 3], [1, 2, 3], k=3) == pytest.approx(1.0)


def test_ndcg_at_k_reversed_relevant_still_one():
    # Relevant set order doesn't matter; binary relevance
    assert ndcg_at_k([1, 2, 3], [3, 2, 1], k=3) == pytest.approx(1.0)


def test_ndcg_at_k_one_hit_rank_two():
    # Hit at rank 2 (zero-indexed rank 1): gain = 1/log2(3)
    # IDCG with 1 relevant in top-3 = 1/log2(2) = 1
    expected = (1 / math.log2(3)) / 1.0
    assert ndcg_at_k([99, 1, 88], [1], k=3) == pytest.approx(expected)


def test_ndcg_at_k_zero_hits():
    assert ndcg_at_k([7, 8, 9], [1, 2, 3], k=3) == 0.0


def test_evaluate_aggregates_across_users():
    recommendations = {
        "u1": [1, 2, 3],   # 1 hit
        "u2": [4, 5, 6],   # 0 hits
    }
    relevant = {"u1": [1], "u2": [99]}
    out = evaluate(recommendations, relevant, k=3)
    assert out["n_users"] == 2
    assert out["hit_rate@3"] == pytest.approx(0.5)
    assert out["precision@3"] == pytest.approx((1 / 3 + 0) / 2)


def test_evaluate_skips_users_with_no_relevant():
    out = evaluate({"u1": [1, 2]}, {"u1": []}, k=2)
    assert out["n_users"] == 0


def test_precision_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        precision_at_k([1, 2], [1], k=0)
