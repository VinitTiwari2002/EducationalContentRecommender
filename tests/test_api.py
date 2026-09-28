"""Integration tests for the FastAPI service.

Tests populate `src.api.state` directly with a small hand-crafted set of
fitted models — no full pipeline run required. This keeps the suite fast
(<1 s) while exercising the same code paths a real request would.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy import sparse

from src import api
from src.baselines import PopularityRecommender, RandomRecommender
from src.content import ContentRecommender
from src.hybrid import HybridRecommender
from src.persistence import ServingContext


@pytest.fixture
def populated_state(monkeypatch):
    """Populate api.state with tiny fitted models before each test."""
    # 4 users, 5 items, 2 features.
    features = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.5, 0.5]],
        dtype=np.float32,
    )
    outcome = np.array([50.0, 55.0, 60.0, 90.0, 70.0], dtype=np.float32)
    rows = [0, 0, 1, 1, 2, 3]
    cols = [0, 1, 2, 3, 0, 4]
    data = [3, 2, 4, 3, 1, 2]
    train = sparse.coo_matrix((data, (rows, cols)), shape=(4, 5)).tocsr()

    student_ids = np.array([100, 200, 300, 400], dtype=np.int64)
    item_ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)

    pop = PopularityRecommender()
    pop.fit(train)
    rand = RandomRecommender(seed=0)
    rand.fit(train)
    content = ContentRecommender()
    content.fit(train, features)
    hybrid = HybridRecommender(alpha=0.4, beta=0.4, gamma=0.2)
    hybrid.fit(train, features, outcome)

    context = ServingContext(
        student_index=student_ids,
        item_index=item_ids,
        user_candidates=[
            np.array([0, 1, 4], dtype=np.int64),
            np.array([2, 3, 4], dtype=np.int64),
            np.array([0, 1, 2, 3, 4], dtype=np.int64),
            np.array([4], dtype=np.int64),
        ],
        outcome_score=outcome,
        cutoff_date=172,
    )
    manifest = {
        "n_models": 4,
        "model_names": ["Content", "Hybrid", "Popularity", "Random"],
        "n_students": 4,
        "n_items": 5,
        "cutoff_date": 172,
        "git_commit": None,
        "metadata": {"hybrid_weights": [0.4, 0.4, 0.2]},
    }

    api.state.models = {
        "Random": rand,
        "Popularity": pop,
        "Content": content,
        "Hybrid": hybrid,
    }
    api.state.context = context
    api.state.manifest = manifest
    api.state.student_id_to_row = {int(s): int(r) for r, s in enumerate(student_ids)}
    api.state.item_id_to_col = {int(i): int(c) for c, i in enumerate(item_ids)}
    yield
    # Teardown: reset state so tests don't bleed into each other.
    api.state.models = {}
    api.state.context = None
    api.state.manifest = {}
    api.state.student_id_to_row = {}
    api.state.item_id_to_col = {}


@pytest.fixture
def client():
    return TestClient(api.app)


def test_health_ready(populated_state, client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["n_models"] == 4
    assert body["n_students"] == 4
    assert body["n_items"] == 5
    assert body["cutoff_date"] == 172
    assert "Hybrid" in body["model_names"]


def test_health_uninitialised(client):
    # No populated_state fixture → api.state is empty.
    api.state.context = None
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "uninitialised"


def test_recommend_returns_top_k_hybrid(populated_state, client):
    resp = client.get("/recommend/100?k=3&model=hybrid")
    assert resp.status_code == 200
    body = resp.json()
    assert body["student_id"] == 100
    assert body["model"] == "Hybrid"
    assert 1 <= len(body["items"]) <= 3
    # Hybrid response should carry a per-item breakdown.
    first = body["items"][0]
    assert set(first.keys()) >= {"item_id", "rank", "cf_weighted", "content_weighted", "outcome_weighted", "total"}
    # Ranks are 1-based and unique.
    ranks = [it["rank"] for it in body["items"]]
    assert ranks == list(range(1, len(ranks) + 1))
    # Items returned are IDs, not column indices — must be from item_index.
    for it in body["items"]:
        assert it["item_id"] in {10, 20, 30, 40, 50}


def test_recommend_case_insensitive_model_name(populated_state, client):
    resp = client.get("/recommend/100?k=2&model=POPULARITY")
    assert resp.status_code == 200
    assert resp.json()["model"] == "Popularity"


def test_recommend_non_hybrid_has_no_breakdown(populated_state, client):
    resp = client.get("/recommend/100?k=2&model=Random")
    body = resp.json()
    for it in body["items"]:
        assert "cf_weighted" not in it


def test_recommend_unknown_student_returns_404(populated_state, client):
    resp = client.get("/recommend/9999?k=5&model=hybrid")
    assert resp.status_code == 404


def test_recommend_unknown_model_returns_404(populated_state, client):
    resp = client.get("/recommend/100?k=5&model=xgboost")
    assert resp.status_code == 404


def test_recommend_invalid_k_returns_422(populated_state, client):
    resp = client.get("/recommend/100?k=0&model=hybrid")
    assert resp.status_code == 422


def test_decompose_returns_expected_shape(populated_state, client):
    resp = client.get("/decompose/100/40")
    assert resp.status_code == 200
    body = resp.json()
    assert body["student_id"] == 100
    assert body["item_id"] == 40
    for key in ("cf_raw", "content_raw", "outcome_raw",
                "cf_weighted", "content_weighted", "outcome_weighted", "total"):
        assert key in body
    # Item 40 (col 3) has the highest outcome (90.0) — must show that.
    assert body["outcome_raw"] == pytest.approx(90.0)


def test_decompose_unknown_item_returns_404(populated_state, client):
    resp = client.get("/decompose/100/9999")
    assert resp.status_code == 404


def test_uninitialised_endpoints_return_503(client):
    api.state.context = None
    api.state.models = {}
    resp = client.get("/recommend/100?k=1&model=hybrid")
    assert resp.status_code == 503
    resp = client.get("/decompose/100/40")
    assert resp.status_code == 503
