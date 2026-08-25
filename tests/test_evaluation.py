"""Unit tests for the evaluation-harness helpers (cold-start, Bonferroni)."""
import numpy as np
import pytest
from scipy import sparse

from src.evaluation import (
    bonferroni_adjust,
    cold_start_masks,
    stratified_evaluate,
)


def test_cold_start_masks_partition_users():
    # User 0: 3 clicks (cold @ threshold=10)
    # User 1: 12 clicks (warm)
    # User 2: 0 clicks (cold)
    rows = [0, 0, 0] + [1] * 12
    cols = list(range(3)) + list(range(12))
    data = [1] * 15
    train = sparse.coo_matrix((data, (rows, cols)), shape=(3, 15)).tocsr()
    warm, cold = cold_start_masks(train, threshold=10)
    np.testing.assert_array_equal(warm, [False, True, False])
    np.testing.assert_array_equal(cold, [True, False, True])
    # Warm and cold must partition (never both True for same user).
    assert not (warm & cold).any()
    # Every user must be labelled.
    assert (warm | cold).all()


def test_cold_start_threshold_zero_flags_only_zero_click_users():
    rows = [0, 1]
    cols = [0, 0]
    data = [1, 1]
    train = sparse.coo_matrix((data, (rows, cols)), shape=(3, 2)).tocsr()
    warm, cold = cold_start_masks(train, threshold=1)
    np.testing.assert_array_equal(warm, [True, True, False])
    np.testing.assert_array_equal(cold, [False, False, True])


def test_stratified_evaluate_restricts_to_mask():
    # 4 users; warm mask picks 1 and 3.
    recs = {0: [0, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1]}
    relevant = {0: [2], 1: [0], 2: [3], 3: [1]}
    mask = np.array([False, True, False, True])
    m = stratified_evaluate(recs, relevant, mask, k=2)
    # Both masked users have a hit → precision@2 = 0.5 for each → mean 0.5
    assert m["precision@2"] == pytest.approx(0.5)
    assert m["n_users"] == 2


def test_bonferroni_scales_p_values():
    p = np.array([0.01, 0.05, 0.20, 0.30])
    adj = bonferroni_adjust(p, n_comparisons=5)
    np.testing.assert_allclose(adj, [0.05, 0.25, 1.0, 1.0])


def test_bonferroni_caps_at_one():
    p = np.array([0.5, 1.0])
    adj = bonferroni_adjust(p, n_comparisons=10)
    np.testing.assert_allclose(adj, [1.0, 1.0])


def test_bonferroni_invalid_n_raises():
    with pytest.raises(ValueError):
        bonferroni_adjust(np.array([0.1]), n_comparisons=0)
