"""Unit tests for the persistence module."""
import json
import numpy as np
import pytest
from scipy import sparse

from src.baselines import PopularityRecommender
from src.persistence import (
    ServingContext,
    artefacts_exist,
    load_models,
    save_models,
)


@pytest.fixture
def tiny_context():
    """Small serving context that round-trips quickly."""
    return ServingContext(
        student_index=np.array([100, 200, 300], dtype=np.int64),
        item_index=np.array([10, 20, 30, 40], dtype=np.int64),
        user_candidates=[
            np.array([0, 1], dtype=np.int64),
            np.array([2, 3], dtype=np.int64),
            np.array([0, 1, 2, 3], dtype=np.int64),
        ],
        outcome_score=np.array([50.0, 60.0, 70.0, 80.0], dtype=np.float32),
        cutoff_date=172,
    )


@pytest.fixture
def tiny_models():
    rows, cols, data = [0, 1, 2], [0, 1, 2], [1, 1, 1]
    train = sparse.coo_matrix((data, (rows, cols)), shape=(3, 4)).tocsr()
    pop = PopularityRecommender()
    pop.fit(train)
    return {"Popularity": pop}


def test_artefacts_exist_false_on_empty(tmp_path):
    assert not artefacts_exist(tmp_path)


def test_save_and_load_roundtrip(tmp_path, tiny_models, tiny_context):
    out = save_models(tiny_models, tiny_context, metadata={"note": "test"}, out_dir=tmp_path)
    assert out == tmp_path
    assert artefacts_exist(tmp_path)

    models, ctx, manifest = load_models(tmp_path)
    assert set(models.keys()) == {"Popularity"}
    np.testing.assert_array_equal(ctx.student_index, tiny_context.student_index)
    np.testing.assert_array_equal(ctx.item_index, tiny_context.item_index)
    np.testing.assert_allclose(ctx.outcome_score, tiny_context.outcome_score)
    assert ctx.cutoff_date == tiny_context.cutoff_date
    assert manifest["n_models"] == 1
    assert manifest["n_students"] == 3
    assert manifest["n_items"] == 4
    assert manifest["metadata"]["note"] == "test"


def test_load_from_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_models(tmp_path)


def test_manifest_is_valid_json(tmp_path, tiny_models, tiny_context):
    save_models(tiny_models, tiny_context, out_dir=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert isinstance(manifest, dict)
    assert "model_names" in manifest


def test_loaded_popularity_still_recommends(tmp_path, tiny_models, tiny_context):
    save_models(tiny_models, tiny_context, out_dir=tmp_path)
    models, _, _ = load_models(tmp_path)
    recs = models["Popularity"].recommend(user_row=0, k=2)
    assert isinstance(recs, list)
    assert len(recs) <= 2
