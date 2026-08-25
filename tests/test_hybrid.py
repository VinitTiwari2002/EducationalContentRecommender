"""Unit tests for the hybrid recommender + ablation behaviour."""
import numpy as np
import pytest
from scipy import sparse

from src.hybrid import HybridRecommender


@pytest.fixture
def tiny_setup():
    # 3 users, 4 items, 2 features (topic A vs topic B).
    features = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    # User 0 clicked topic-A items 0 and 1.
    # User 1 clicked topic-B items 2 and 3.
    # User 2 clicked topic-A item 0 only.
    rows = [0, 0, 1, 1, 2]
    cols = [0, 1, 2, 3, 0]
    data = [3, 2, 3, 2, 4]
    train = sparse.coo_matrix((data, (rows, cols)), shape=(3, 4)).tocsr()
    # Outcome: item 3 was accessed by high scorers.
    outcome = np.array([50.0, 50.0, 50.0, 90.0], dtype=np.float32)
    return train, features, outcome


def test_full_hybrid_returns_valid_items(tiny_setup):
    train, features, outcome = tiny_setup
    model = HybridRecommender(alpha=0.5, beta=0.3, gamma=0.2)
    model.fit(train, features, outcome)
    recs = model.recommend(2, k=3)
    assert set(recs).issubset({1, 2, 3})  # user 2 has seen 0
    assert len(recs) == 3


def test_gamma_zero_falls_back_to_cf_plus_content(tiny_setup):
    train, features, outcome = tiny_setup
    model = HybridRecommender(alpha=0.5, beta=0.5, gamma=0.0)
    model.fit(train, features, outcome)
    # With gamma=0 the outcome signal contributes 0 for every item.
    # Recommendations are driven by CF+content alone.
    recs = model.recommend(2, k=3)
    assert isinstance(recs, list) and len(recs) == 3


def test_outcome_only_recommends_highest_outcome_item(tiny_setup):
    train, features, outcome = tiny_setup
    # alpha=beta=0, gamma=1: recommend by outcome_score alone.
    model = HybridRecommender(alpha=0.0, beta=0.0, gamma=1.0)
    model.fit(train, features, outcome)
    # User 2's unseen items: {1, 2, 3}. outcome[3]=90 is the max.
    recs = model.recommend(2, k=3)
    assert recs[0] == 3


def test_content_only_matches_topic(tiny_setup):
    train, features, outcome = tiny_setup
    model = HybridRecommender(alpha=0.0, beta=1.0, gamma=0.0)
    model.fit(train, features, outcome)
    # User 2 clicked topic-A item 0; unseen topic-A item 1 should rank first.
    recs = model.recommend(2, k=3)
    assert recs[0] == 1


def test_all_zero_weights_raises():
    with pytest.raises(ValueError):
        HybridRecommender(alpha=0, beta=0, gamma=0)


def test_negative_weight_raises():
    with pytest.raises(ValueError):
        HybridRecommender(alpha=-0.1, beta=0.5, gamma=0.5)


def test_outcome_dimension_mismatch_raises(tiny_setup):
    train, features, _ = tiny_setup
    bad_outcome = np.zeros(999, dtype=np.float32)
    model = HybridRecommender()
    with pytest.raises(ValueError):
        model.fit(train, features, bad_outcome)


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError):
        HybridRecommender().recommend(0, k=3)


def test_excludes_seen(tiny_setup):
    train, features, outcome = tiny_setup
    model = HybridRecommender()
    model.fit(train, features, outcome)
    recs = model.recommend(0, k=5)  # user 0 has seen 0, 1
    assert 0 not in recs and 1 not in recs
