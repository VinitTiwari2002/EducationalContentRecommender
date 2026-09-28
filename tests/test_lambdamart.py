"""Unit tests for the LambdaMART two-stage reranker."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.features import ItemFeatures
from src.hybrid import HybridRecommender
from src.lambdamart import LambdaMARTReranker
from src.reranker_features import UserFeatures, build_pair_features, pair_feature_names


@pytest.fixture
def tiny_setup():
    """Tiny setup: 8 users, 6 items, 2 item-feature columns."""
    features_matrix = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0],
         [0.0, 1.0], [0.5, 0.5], [0.3, 0.7]], dtype=np.float32
    )
    outcome = np.array([50.0, 55.0, 60.0, 90.0, 70.0, 65.0], dtype=np.float32)
    # 8 users, all click items 0-1 (topic A) or 2-3 (topic B) etc.
    rows = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    cols = [0, 1, 0, 1, 2, 3, 2, 3, 0, 4, 1, 4, 2, 5, 3, 5]
    data = [3, 2] * 8
    train = sparse.coo_matrix((data, (rows, cols)), shape=(8, 6)).tocsr()

    item_features = ItemFeatures(
        matrix=features_matrix,
        feature_names=["topic_A", "topic_B"],
        outcome_score=outcome,
    )
    user_features = UserFeatures(
        matrix=np.random.default_rng(0).random((8, 7)).astype(np.float32),
        feature_names=["gender_M", "gender_F", "imd_ordinal", "age_ordinal",
                        "disability_Y", "log_train_clicks_norm",
                        "mean_past_assessment_norm"],
    )
    hybrid = HybridRecommender(alpha=0.4, beta=0.4, gamma=0.2)
    hybrid.fit(train, features_matrix, outcome)
    return train, item_features, user_features, hybrid


def test_build_pair_features_shape(tiny_setup):
    _, item_features, user_features, hybrid = tiny_setup
    cand = np.array([0, 2, 4], dtype=np.int64)
    X = build_pair_features(0, cand, hybrid, item_features, user_features)
    # 3 stage-1 + 2 item + 7 user = 12 features
    assert X.shape == (3, 12)


def test_build_pair_features_empty_candidates(tiny_setup):
    _, item_features, user_features, hybrid = tiny_setup
    X = build_pair_features(0, np.array([], dtype=np.int64),
                            hybrid, item_features, user_features)
    assert X.shape == (0, 12)


def test_pair_feature_names_matches_shape(tiny_setup):
    _, item_features, user_features, _ = tiny_setup
    names = pair_feature_names(item_features, user_features)
    assert len(names) == 12
    assert names[0] == "stage1_cf"
    assert "item__topic_A" in names
    assert "user__gender_M" in names


def test_reranker_stage1_type_check(tiny_setup):
    _, item_features, user_features, _ = tiny_setup
    with pytest.raises(TypeError):
        LambdaMARTReranker(
            stage1="not a hybrid",  # type: ignore[arg-type]
            item_features=item_features,
            user_features=user_features,
        )


def test_reranker_fit_ranker_and_recommend(tiny_setup):
    train, item_features, user_features, hybrid = tiny_setup
    reranker = LambdaMARTReranker(
        stage1=hybrid,
        item_features=item_features,
        user_features=user_features,
        top_n=5,
        lgbm_params=dict(n_estimators=20, num_leaves=7,
                          min_child_samples=1, min_data_in_leaf=1,
                          min_gain_to_split=0.0),
    )
    reranker.fit(train)

    # Fake labels: user u prefers items adjacent to what they clicked.
    positives = {
        0: [2],  # user 0 clicked 0, 1 — pretend 2 is their next click
        1: [3],
        2: [0],
        3: [1],
        4: [3],
        5: [3],
        6: [4],
        7: [4],
    }
    candidate_pools = {u: np.array([0, 1, 2, 3, 4, 5], dtype=np.int64) for u in range(8)}
    reranker.fit_ranker(candidate_pools, positives)

    recs = reranker.recommend(0, k=3, candidates=candidate_pools[0])
    assert isinstance(recs, list)
    assert len(recs) <= 3
    for c in recs:
        assert c in {0, 1, 2, 3, 4, 5}


def test_reranker_recommend_before_fit_raises(tiny_setup):
    _, item_features, user_features, hybrid = tiny_setup
    reranker = LambdaMARTReranker(
        stage1=hybrid,
        item_features=item_features,
        user_features=user_features,
    )
    with pytest.raises(RuntimeError):
        reranker.recommend(0, k=3)


def test_reranker_recommend_before_fit_ranker_raises(tiny_setup):
    train, item_features, user_features, hybrid = tiny_setup
    reranker = LambdaMARTReranker(
        stage1=hybrid,
        item_features=item_features,
        user_features=user_features,
    )
    reranker.fit(train)
    with pytest.raises(RuntimeError):
        reranker.recommend(0, k=3)


def test_reranker_feature_importance(tiny_setup):
    train, item_features, user_features, hybrid = tiny_setup
    reranker = LambdaMARTReranker(
        stage1=hybrid,
        item_features=item_features,
        user_features=user_features,
        top_n=5,
        lgbm_params=dict(n_estimators=10, num_leaves=5,
                          min_child_samples=1, min_data_in_leaf=1,
                          min_gain_to_split=0.0),
    )
    reranker.fit(train)
    positives = {u: [(u + 1) % 6] for u in range(8)}
    candidate_pools = {u: np.arange(6, dtype=np.int64) for u in range(8)}
    reranker.fit_ranker(candidate_pools, positives)
    fi = reranker.feature_importance(pair_feature_names(item_features, user_features))
    assert set(fi.keys()) == set(pair_feature_names(item_features, user_features))
    assert all(v >= 0 for v in fi.values())
