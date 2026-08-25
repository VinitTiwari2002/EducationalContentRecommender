"""Unit tests for the ALS collaborative-filtering recommender.

These tests match the shape of test_svd.py so any regression in the
recommender interface is caught for both CF implementations.
"""
import numpy as np
import pytest
from scipy import sparse

from src.als import ALSRecommender


@pytest.fixture
def block_pattern_train():
    """Same two-block interaction pattern used in test_svd.py."""
    rows = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    cols = [0, 1, 0, 1, 0, 1, 3, 4, 3, 4, 3, 4]
    data = [5, 5, 4, 4, 3, 3, 5, 5, 4, 4, 3, 3]
    return sparse.coo_matrix((data, (rows, cols)), shape=(6, 5)).tocsr()


def test_within_block_users_get_similar_scores(block_pattern_train):
    model = ALSRecommender(n_factors=2, iterations=20, random_state=0)
    model.fit(block_pattern_train)
    # Users 0, 1, 2 share the same block; their score vectors should be
    # closer to each other than to any user from block B (3, 4, 5).
    scores = np.stack([model.score(u) for u in range(6)])
    same_block = np.linalg.norm(scores[0] - scores[1])
    cross_block = np.linalg.norm(scores[0] - scores[3])
    assert same_block < cross_block


def test_excludes_seen(block_pattern_train):
    model = ALSRecommender(n_factors=2, iterations=10, random_state=0)
    model.fit(block_pattern_train)
    recs = model.recommend(0, k=5)
    assert 0 not in recs and 1 not in recs


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError):
        ALSRecommender().recommend(0, k=3)


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        ALSRecommender().score(0)


def test_invalid_n_factors_raises():
    with pytest.raises(ValueError):
        ALSRecommender(n_factors=0)


def test_candidates_restriction(block_pattern_train):
    model = ALSRecommender(n_factors=2, iterations=5, random_state=0)
    model.fit(block_pattern_train)
    recs = model.recommend(0, k=5, candidates=np.array([2, 3, 4]))
    assert set(recs).issubset({2, 3, 4})


def test_reproducible_with_seed(block_pattern_train):
    a = ALSRecommender(n_factors=4, iterations=10, random_state=7)
    b = ALSRecommender(n_factors=4, iterations=10, random_state=7)
    a.fit(block_pattern_train)
    b.fit(block_pattern_train)
    # ALS is deterministic given random_state; scores should match within
    # numerical tolerance.
    for u in range(6):
        np.testing.assert_allclose(a.score(u), b.score(u), atol=1e-4)


def test_als_can_back_hybrid(block_pattern_train):
    """The Hybrid should accept an ALSRecommender as its CF component."""
    from src.content import ContentRecommender
    from src.hybrid import HybridRecommender

    features = np.eye(5, dtype=np.float32)
    outcome = np.full(5, 70.0, dtype=np.float32)
    als = ALSRecommender(n_factors=2, iterations=5, random_state=0)
    content = ContentRecommender()
    hybrid = HybridRecommender(alpha=0.5, beta=0.5, gamma=0.0, cf=als, content=content)
    hybrid.fit(block_pattern_train, features, outcome)
    # cf property should be the ALS instance we passed in.
    assert hybrid.cf is als
    recs = hybrid.recommend(0, k=3)
    assert isinstance(recs, list)
