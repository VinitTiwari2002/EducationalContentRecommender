"""Unit tests for the gated (switching) hybrid."""
import numpy as np
import pytest
from scipy import sparse

from src.baselines import PopularityRecommender
from src.gated_hybrid import GatedHybridRecommender
from src.hybrid import HybridRecommender


@pytest.fixture
def tiny_setup():
    # 3 users, 4 items, 2 features.
    features = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    # User 0: 4 clicks total (warm at threshold=3, cold at threshold=5).
    # User 1: 2 clicks (cold at threshold=3).
    # User 2: 1 click (cold at any threshold >= 2).
    rows = [0, 0, 0, 0, 1, 1, 2]
    cols = [0, 1, 2, 3, 2, 3, 0]
    data = [1, 1, 1, 1, 1, 1, 1]
    train = sparse.coo_matrix((data, (rows, cols)), shape=(3, 4)).tocsr()
    outcome = np.full(4, 70.0, dtype=np.float32)
    return train, features, outcome


def test_gate_routes_cold_users_to_cold_model(tiny_setup):
    train, features, outcome = tiny_setup
    gated = GatedHybridRecommender(
        cold_model=PopularityRecommender(),
        warm_model=HybridRecommender(alpha=0.5, beta=0.5, gamma=0.0),
        threshold=3,
    )
    gated.fit(train, features, outcome)
    # User 0 has 4 unique items → warm; user 1 has 2 → cold; user 2 has 1 → cold.
    assert not gated.is_cold(0)
    assert gated.is_cold(1)
    assert gated.is_cold(2)


def test_recommendations_come_from_correct_branch(tiny_setup):
    train, features, outcome = tiny_setup
    # Force a distinguishable behaviour on the cold branch by using
    # Popularity on the raw click matrix: item 0 has clicks 1+1=2, items
    # 2 and 3 have clicks 1+1=2 too, item 1 has 1. All ties broken by
    # column order in the popularity recommender.
    gated = GatedHybridRecommender(
        cold_model=PopularityRecommender(),
        warm_model=HybridRecommender(alpha=0.5, beta=0.5, gamma=0.0),
        threshold=3,
    )
    gated.fit(train, features, outcome)
    # Cold user 2 should route through Popularity, which excludes seen
    # (item 0), leaving {1, 2, 3} ordered by clicks then column: [2, 3, 1].
    assert gated.recommend(2, k=3) == [2, 3, 1]


def test_threshold_zero_makes_everyone_warm(tiny_setup):
    train, features, outcome = tiny_setup
    gated = GatedHybridRecommender(
        cold_model=PopularityRecommender(),
        warm_model=HybridRecommender(),
        threshold=0,
    )
    gated.fit(train, features, outcome)
    for u in range(3):
        assert not gated.is_cold(u)


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError):
        GatedHybridRecommender(
            cold_model=PopularityRecommender(),
            warm_model=HybridRecommender(),
        ).recommend(0, k=3)


def test_negative_threshold_raises():
    with pytest.raises(ValueError):
        GatedHybridRecommender(
            cold_model=PopularityRecommender(),
            warm_model=HybridRecommender(),
            threshold=-1,
        )
