"""FastAPI service exposing the recommender as a queryable artefact.

Three read-only endpoints (per §3.5 of the design chapter):

    GET /health
    GET /recommend/{student_id}?k=10&model=hybrid
    GET /decompose/{student_id}/{item_id}

The service loads persisted model artefacts once at startup (see
`src.persistence`), so per-request cost is a small numpy operation. There
is no authentication — this is a coursework demonstration on aggregated
data, not a production endpoint.

Run with:
    uvicorn src.api:app --reload

The dashboard (Streamlit) hits this service. In production it would be
fronted by nginx or similar; for the demo we serve it directly.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .hybrid import HybridRecommender
from .persistence import DEFAULT_MODELS_DIR, ServingContext, load_models


# --------------------------------------------------------------------------- #
# Serving state — populated at startup, immutable during a request.
# --------------------------------------------------------------------------- #

class _State:
    """Container for artefacts loaded from disk on startup."""

    models: dict[str, Any] = {}
    context: ServingContext | None = None
    manifest: dict[str, Any] = {}
    student_id_to_row: dict[int, int] = {}
    item_id_to_col: dict[int, int] = {}


state = _State()


def _initialise(models_dir: Path | str = DEFAULT_MODELS_DIR) -> None:
    """Load persisted artefacts. Called once by the startup event."""
    models_dir = Path(models_dir)
    state.models, state.context, state.manifest = load_models(models_dir)
    state.student_id_to_row = {
        int(s): int(r) for r, s in enumerate(state.context.student_index)
    }
    state.item_id_to_col = {
        int(i): int(c) for c, i in enumerate(state.context.item_index)
    }


# --------------------------------------------------------------------------- #
# Response schemas.
# --------------------------------------------------------------------------- #

class HealthResponse(BaseModel):
    status: str
    n_models: int
    model_names: list[str]
    n_students: int
    n_items: int
    cutoff_date: int


class RecommendationItem(BaseModel):
    item_id: int
    rank: int


class HybridBreakdownItem(RecommendationItem):
    cf_weighted: float
    content_weighted: float
    outcome_weighted: float
    total: float


class RecommendResponse(BaseModel):
    student_id: int
    model: str
    k: int
    items: list[RecommendationItem | HybridBreakdownItem]


class DecomposeResponse(BaseModel):
    student_id: int
    item_id: int
    cf_raw: float
    content_raw: float
    outcome_raw: float
    cf_weighted: float
    content_weighted: float
    outcome_weighted: float
    total: float


# --------------------------------------------------------------------------- #
# FastAPI app.
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Load artefacts unless a test has already populated `state` manually."""
    if state.context is None:
        models_dir = Path(os.environ.get("RECSYS_MODELS_DIR", str(DEFAULT_MODELS_DIR)))
        if (models_dir / "manifest.json").exists():
            _initialise(models_dir)
        # Otherwise defer failure until the first request; /health will
        # report "uninitialised" rather than crashing the app import.
    yield


app = FastAPI(
    title="Educational Content Recommender API",
    version="0.1.0",
    description="Query the OULAD hybrid recommender for a specific student.",
    lifespan=_lifespan,
)


def _ensure_loaded() -> None:
    if state.context is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artefacts not loaded. Run `python -m src.pipeline "
                "--persist-models` first, or set RECSYS_MODELS_DIR to the "
                "artefact directory."
            ),
        )


def _resolve_student_row(student_id: int) -> int:
    row = state.student_id_to_row.get(int(student_id))
    if row is None:
        raise HTTPException(status_code=404, detail=f"student_id {student_id} not in split")
    return row


def _resolve_item_col(item_id: int) -> int:
    col = state.item_id_to_col.get(int(item_id))
    if col is None:
        raise HTTPException(status_code=404, detail=f"item_id {item_id} not in catalogue")
    return col


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report the currently-loaded model roster + split cutoff."""
    if state.context is None:
        return HealthResponse(
            status="uninitialised",
            n_models=0,
            model_names=[],
            n_students=0,
            n_items=0,
            cutoff_date=-1,
        )
    return HealthResponse(
        status="ready",
        n_models=len(state.models),
        model_names=sorted(state.models.keys()),
        n_students=int(state.context.student_index.shape[0]),
        n_items=int(state.context.item_index.shape[0]),
        cutoff_date=int(state.context.cutoff_date),
    )


@app.get("/recommend/{student_id}", response_model=RecommendResponse)
def recommend(
    student_id: int,
    k: int = Query(10, ge=1, le=100),
    model: str = Query("hybrid"),
) -> RecommendResponse:
    """Return the top-K item IDs for a student, restricted to their
    course-scoped candidate pool. For the Hybrid/GatedHybrid models, also
    returns the score decomposition per item.
    """
    _ensure_loaded()
    row = _resolve_student_row(student_id)

    # Normalise the model name (case-insensitive; hybrid → Hybrid, etc.).
    canonical = {name.lower(): name for name in state.models}
    key = canonical.get(model.lower())
    if key is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown model {model!r}; available: {sorted(state.models)}",
        )
    recommender = state.models[key]
    candidates = state.context.user_candidates[row]
    recs_cols = recommender.recommend(row, k=k, candidates=candidates)
    item_index = state.context.item_index
    items: list[Any] = []
    # Hybrid + GatedHybrid support score_breakdown for the decomposition
    # display; other models return plain (item_id, rank) pairs.
    hybrid = None
    if isinstance(recommender, HybridRecommender):
        hybrid = recommender
    elif hasattr(recommender, "warm_model") and isinstance(recommender.warm_model, HybridRecommender):
        # GatedHybrid: only expose decomposition for warm users.
        if not recommender.is_cold(row):
            hybrid = recommender.warm_model

    if hybrid is not None and recs_cols:
        bd = hybrid.score_breakdown(user_row=row, items=np.asarray(recs_cols))
        for rank, col in enumerate(recs_cols, start=1):
            i = int(np.where(bd["items"] == col)[0][0])
            items.append(HybridBreakdownItem(
                item_id=int(item_index[col]),
                rank=rank,
                cf_weighted=float(bd["cf_weighted"][i]),
                content_weighted=float(bd["content_weighted"][i]),
                outcome_weighted=float(bd["outcome_weighted"][i]),
                total=float(bd["total"][i]),
            ))
    else:
        for rank, col in enumerate(recs_cols, start=1):
            items.append(RecommendationItem(item_id=int(item_index[col]), rank=rank))

    return RecommendResponse(student_id=int(student_id), model=key, k=k, items=items)


@app.get("/decompose/{student_id}/{item_id}", response_model=DecomposeResponse)
def decompose(student_id: int, item_id: int) -> DecomposeResponse:
    """Explain a single (student, item) pair via the Hybrid decomposition.

    Weighted contributions are computed within the user's course-scoped
    candidate pool (matching what /recommend uses to rank), so the total
    for this item is directly comparable with the totals returned by
    /recommend. The raw scores are the un-normalised component values,
    useful for debugging and dashboard tooltips.
    """
    _ensure_loaded()
    row = _resolve_student_row(student_id)
    col = _resolve_item_col(item_id)

    hybrid = state.models.get("Hybrid")
    if hybrid is None:
        raise HTTPException(
            status_code=404,
            detail="Hybrid model not persisted; /decompose requires it.",
        )
    assert isinstance(hybrid, HybridRecommender)
    # Compute the breakdown over the user's full candidate pool so the
    # min-max normalisation is meaningful, then pick out our target item.
    candidates = state.context.user_candidates[row]
    if col not in candidates:
        # Item is outside the user's course scope; fall back to a
        # single-item breakdown that at least exposes the raw scores.
        bd = hybrid.score_breakdown(user_row=row, items=np.array([col]))
        idx = 0
    else:
        bd = hybrid.score_breakdown(user_row=row, items=np.asarray(candidates))
        idx = int(np.where(bd["items"] == col)[0][0])
    return DecomposeResponse(
        student_id=int(student_id),
        item_id=int(item_id),
        cf_raw=float(bd["cf_raw"][idx]),
        content_raw=float(bd["content_raw"][idx]),
        outcome_raw=float(bd["outcome_raw"][idx]),
        cf_weighted=float(bd["cf_weighted"][idx]),
        content_weighted=float(bd["content_weighted"][idx]),
        outcome_weighted=float(bd["outcome_weighted"][idx]),
        total=float(bd["total"][idx]),
    )
