"""Unit tests for the SVD collaborative-filtering recommender."""
import numpy as np
import pytest
from scipy import sparse

from src.svd import SVDRecommender


@pytest.fixture
def block_pattern_train():
    """Two blocks of users with disjoint item preferences.

    Users 0-2 like items 0, 1. Users 3-5 like items 3, 4. Item 2 is
    universally rarely accessed. A rank-2 SVD should recover the two
    latent user groups and rank the "other user group's" item high for
    users who have already clicked their in-group items.
    """
    rows = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    cols = [0, 1, 0, 1, 0, 1, 3, 4, 3, 4, 3, 4]
    data = [5, 5, 4, 4, 3, 3, 5, 5, 4, 4, 3, 3]
    return sparse.coo_matrix((data, (rows, cols)), shape=(6, 5)).tocsr()


def test_reproducible_with_seed(block_pattern_train):
    a = SVDRecommender(n_factors=2, random_state=7)
    b = SVDRecommender(n_factors=2, random_state=7)
    a.fit(block_pattern_train)
    b.fit(block_pattern_train)
    # Score arrays should match up to numerical precision. ARPACK can flip
    # the sign of essentially-zero values (order ~1e-7) across runs, so we
    # compare with a mild absolute tolerance rather than relative.
    for u in range(6):
        np.testing.assert_allclose(a.score(u), b.score(u), atol=1e-5)


def test_within_block_users_get_similar_scores(block_pattern_train):
    model = SVDRecommender(n_factors=2, random_state=0)
    model.fit(block_pattern_train)
    # Users 0, 1, 2 share the same interaction pattern (block A); their
    # score profiles should therefore be far more similar to each other
    # than to any user in block B (users 3, 4, 5).
    scores = np.stack([model.score(u) for u in range(6)])
    same_block = np.linalg.norm(scores[0] - scores[1])
    cross_block = np.linalg.norm(scores[0] - scores[3])
    assert same_block < cross_block


def test_excludes_seen(block_pattern_train):
    model = SVDRecommender(n_factors=2, random_state=0)
    model.fit(block_pattern_train)
    recs = model.recommend(0, k=5)
    assert 0 not in recs and 1 not in recs


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError):
        SVDRecommender().recommend(0, k=3)


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        SVDRecommender().score(0)


def test_n_factors_capped_to_matrix_rank():
    # A 3x4 matrix has rank at most 3; asking for 100 factors should not crash.
    m = sparse.csr_matrix(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]], dtype=float))
    model = SVDRecommender(n_factors=100, random_state=0)
    model.fit(m)
    scores = model.score(0)
    assert scores.shape == (4,)


def test_invalid_n_factors_raises():
    with pytest.raises(ValueError):
        SVDRecommender(n_factors=0)
