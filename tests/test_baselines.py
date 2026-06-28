"""Unit tests for baselines, using a tiny synthetic interaction matrix."""
import numpy as np
import pytest
from scipy import sparse

from src.baselines import PopularityRecommender, RandomRecommender


@pytest.fixture
def tiny_train() -> sparse.csr_matrix:
    # 3 users, 5 items. Item 0 most popular (3+2=5 clicks), item 4 second (2+1=3).
    # User 0 has seen items 0, 1. User 1 has seen item 4. User 2 has seen items 0, 4.
    rows = [0, 0, 1, 2, 2]
    cols = [0, 1, 4, 0, 4]
    data = [3, 1, 2, 2, 1]
    return sparse.coo_matrix((data, (rows, cols)), shape=(3, 5)).tocsr()


def test_popularity_global_ranking(tiny_train):
    model = PopularityRecommender()
    model.fit(tiny_train)
    # Item totals: 0->5, 1->1, 2->0, 3->0, 4->3
    # Ranking desc: 0, 4, 1, 2, 3
    # User 0 has seen 0, 1 -> top-3 unseen = [4, 2, 3]
    assert model.recommend(user_row=0, k=3) == [4, 2, 3]


def test_popularity_includes_all_if_not_excluding(tiny_train):
    model = PopularityRecommender()
    model.fit(tiny_train)
    # Without excluding seen, every user gets the same global top-K
    assert model.recommend(user_row=0, k=2, exclude_seen=False) == [0, 4]
    assert model.recommend(user_row=2, k=2, exclude_seen=False) == [0, 4]


def test_popularity_restricts_to_candidates(tiny_train):
    model = PopularityRecommender()
    model.fit(tiny_train)
    # Candidate pool [1, 2, 3]; user 0 has seen items 0, 1 so 1 is excluded.
    # Remaining items 2 and 3 both have score 0; stable sort preserves index order.
    assert model.recommend(
        user_row=0, k=3, candidates=np.array([1, 2, 3])
    ) == [2, 3]


def test_popularity_candidates_without_excluding_seen(tiny_train):
    model = PopularityRecommender()
    model.fit(tiny_train)
    # With exclude_seen=False, candidates [1, 2, 3] ranked by score
    # (1 has score 1 -> first, then 2 and 3 tie at 0)
    assert model.recommend(
        user_row=0, k=3, candidates=np.array([1, 2, 3]), exclude_seen=False
    ) == [1, 2, 3]


def test_random_deterministic_with_seed(tiny_train):
    a = RandomRecommender(seed=42)
    b = RandomRecommender(seed=42)
    a.fit(tiny_train)
    b.fit(tiny_train)
    assert a.recommend(0, k=3) == b.recommend(0, k=3)


def test_random_excludes_seen(tiny_train):
    model = RandomRecommender(seed=7)
    model.fit(tiny_train)
    seen = set(tiny_train[0].indices.tolist())
    recs = model.recommend(0, k=3)
    assert not (set(recs) & seen)


def test_random_restricts_to_candidates(tiny_train):
    model = RandomRecommender(seed=0)
    model.fit(tiny_train)
    candidates = np.array([2, 3])  # tiny pool
    recs = model.recommend(0, k=5, candidates=candidates)
    assert set(recs).issubset(set(candidates.tolist()))
    assert len(recs) == 2


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError):
        PopularityRecommender().recommend(0, k=5)
    with pytest.raises(RuntimeError):
        RandomRecommender().recommend(0, k=5)


def test_random_returns_at_most_available(tiny_train):
    # User 0 has seen 2 items out of 5; only 3 candidates available
    model = RandomRecommender(seed=0)
    model.fit(tiny_train)
    recs = model.recommend(0, k=10)
    assert len(recs) == 3
