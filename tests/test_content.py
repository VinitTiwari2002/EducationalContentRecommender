"""Unit tests for the content-based recommender."""
import numpy as np
import pytest
from scipy import sparse

from src.content import ContentRecommender


@pytest.fixture
def tiny_setup():
    # 3 users, 4 items, 3 features.
    # Items 0 and 1 share feature 0 (call it "topic A"); items 2 and 3 share
    # feature 1 ("topic B"); item 3 has an outcome signal on feature 2.
    features = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    # User 0 clicked item 0 (topic A) once.
    # User 1 clicked items 2 (B) and 3 (B+outcome).
    # User 2 clicked nothing (cold-start).
    rows = [0, 1, 1]
    cols = [0, 2, 3]
    data = [1, 1, 1]
    train = sparse.coo_matrix((data, (rows, cols)), shape=(3, 4)).tocsr()
    return train, features


def test_topic_a_user_prefers_topic_a_items(tiny_setup):
    train, features = tiny_setup
    model = ContentRecommender()
    model.fit(train, features)
    # User 0 has clicked topic-A item 0; unseen topic-A item 1 should rank first.
    recs = model.recommend(0, k=3)
    assert recs[0] == 1


def test_topic_b_user_prefers_topic_b_items(tiny_setup):
    train, features = tiny_setup
    model = ContentRecommender()
    model.fit(train, features)
    # User 1 clicked both topic-B items already; unseen items 0 and 1 both
    # have zero cosine with user 1's B profile — recommender returns them
    # in the arbitrary stable order.
    recs = model.recommend(1, k=2)
    assert set(recs) == {0, 1}


def test_cold_start_user_returns_zero_scored_candidates(tiny_setup):
    train, features = tiny_setup
    model = ContentRecommender()
    model.fit(train, features)
    # User 2 has no history; the profile is zero → all scores tie at 0;
    # stable order returns items in column order.
    recs = model.recommend(2, k=4)
    assert recs == [0, 1, 2, 3]


def test_excludes_seen(tiny_setup):
    train, features = tiny_setup
    model = ContentRecommender()
    model.fit(train, features)
    recs = model.recommend(0, k=4)
    assert 0 not in recs  # user 0 has seen item 0


def test_candidates_restriction(tiny_setup):
    train, features = tiny_setup
    model = ContentRecommender()
    model.fit(train, features)
    recs = model.recommend(0, k=4, candidates=np.array([2, 3]))
    assert set(recs).issubset({2, 3})


def test_dimension_mismatch_raises():
    train = sparse.eye(3, format="csr")
    bad_features = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        ContentRecommender().fit(train, bad_features)


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError):
        ContentRecommender().recommend(0, k=3)
