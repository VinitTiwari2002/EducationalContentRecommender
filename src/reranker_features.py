"""Per-(user, item) feature construction for the LambdaMART reranker.

Stage 2 of the two-stage design (§3.5 rerank branch). Given a user_row and
a set of candidate item columns retrieved by Stage 1, we produce a
(n_candidates, n_features) matrix whose columns are:

    1. Stage-1 scores (3 columns): cf_score, content_score, outcome_score.
       All min-max normalised within the candidate pool so LambdaMART sees
       the same relative ranking Stage 1 produced.
    2. Item features (25 columns): the full ItemFeatures.matrix row for
       each candidate — activity_type one-hot, week_from/to, log-access,
       mean-score-of-accessers.
    3. User demographics (5 columns): gender_M, gender_F, imd_ordinal,
       age_ordinal, disability_Y — derived from studentInfo. Ordinal
       encoding for IMD/age preserves the natural ordering.
    4. User behavioural (2 columns): log-clicks-in-training,
       mean-past-assessment-score. Both train-only.

The resulting layout stays under 40 features, which suits LambdaMART on
a catalogue of ~6K items and ~26K users without overfitting.

All the design chapter's leakage rules still apply: every feature is
computable from data available *before* the primary test window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .features import ItemFeatures
from .hybrid import HybridRecommender
from .oulad import OULAD


IMD_ORDER = ["0-10%", "10-20", "20-30%", "30-40%", "40-50%",
             "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
AGE_ORDER = ["0-35", "35-55", "55<="]


@dataclass
class UserFeatures:
    """Per-user feature matrix aligned with student_index rows."""

    matrix: np.ndarray            # (n_users, 7) float32
    feature_names: list[str]      # length 7


def build_user_features(
    data: OULAD,
    train: sparse.csr_matrix,
    student_index: np.ndarray,
) -> UserFeatures:
    """One row per user in student_index; 7 columns of demographic +
    behavioural features. Every value is a train-only quantity."""
    info = data.student_info.drop_duplicates("id_student", keep="first")
    info = info.set_index("id_student").reindex(student_index)

    gender = info["gender"].to_numpy()
    disability = info["disability"].to_numpy()
    imd = info["imd_band"].astype(str).to_numpy()
    age = info["age_band"].astype(str).to_numpy()

    gender_m = (gender == "M").astype(np.float32)
    gender_f = (gender == "F").astype(np.float32)
    disability_y = (disability == "Y").astype(np.float32)

    imd_ordinal = np.array(
        [IMD_ORDER.index(v) if v in IMD_ORDER else -1 for v in imd],
        dtype=np.float32,
    ) / max(1, len(IMD_ORDER) - 1)  # [0, 1], -1 → -0.11 for "unknown"
    age_ordinal = np.array(
        [AGE_ORDER.index(v) if v in AGE_ORDER else -1 for v in age],
        dtype=np.float32,
    ) / max(1, len(AGE_ORDER) - 1)

    # Behavioural features (train-only): total clicks + mean past assessment score.
    n_clicks = np.asarray(train.sum(axis=1)).ravel().astype(np.float32)
    log_clicks = np.log1p(n_clicks)
    max_log = float(log_clicks.max()) if log_clicks.size else 1.0
    log_clicks_norm = log_clicks / max_log if max_log > 0 else log_clicks

    sa = data.student_assessment.dropna(subset=["score"])
    student_mean_score = sa.groupby("id_student")["score"].mean()
    aligned = student_mean_score.reindex(student_index)
    mean_past = aligned.to_numpy(dtype=np.float32) / 100.0  # → [0, 1]
    mean_past = np.where(np.isnan(mean_past), 0.5, mean_past)  # sensible default

    matrix = np.column_stack([
        gender_m, gender_f, imd_ordinal, age_ordinal, disability_y,
        log_clicks_norm, mean_past,
    ]).astype(np.float32)
    feature_names = [
        "gender_M", "gender_F", "imd_ordinal", "age_ordinal",
        "disability_Y", "log_train_clicks_norm", "mean_past_assessment_norm",
    ]
    return UserFeatures(matrix=matrix, feature_names=feature_names)


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    lo = float(x.min()); hi = float(x.max())
    if hi == lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def build_pair_features(
    user_row: int,
    candidate_cols: np.ndarray,
    hybrid: HybridRecommender,
    item_features: ItemFeatures,
    user_features: UserFeatures,
) -> np.ndarray:
    """Build the (n_candidates, n_features) LambdaMART input for one user.

    Stage-1 scores are min-max normalised within the candidate pool so
    the reranker sees the same relative ranking Stage 1 produced —
    LambdaMART trees on raw scores can leak Stage 1's absolute scale.
    """
    cand_cols = np.asarray(candidate_cols, dtype=np.int64)
    if cand_cols.size == 0:
        return np.zeros((0, 3 + item_features.matrix.shape[1] + user_features.matrix.shape[1]),
                        dtype=np.float32)

    # Stage-1 signal columns.
    cf_raw = hybrid.cf.score(user_row)[cand_cols]
    content_raw = hybrid.content.score(user_row)[cand_cols]
    outcome_raw = hybrid._outcome_score[cand_cols]  # type: ignore[attr-defined]
    cf = _minmax(cf_raw)
    content = _minmax(content_raw)
    outcome = _minmax(outcome_raw)
    stage1 = np.column_stack([cf, content, outcome]).astype(np.float32)

    # Item features — one row per candidate.
    item_block = item_features.matrix[cand_cols]

    # User features — same values repeated for every candidate (trees can
    # still interact these with item/Stage-1 columns effectively).
    user_row_vec = user_features.matrix[user_row]
    user_block = np.tile(user_row_vec, (cand_cols.shape[0], 1))

    return np.hstack([stage1, item_block, user_block]).astype(np.float32)


def pair_feature_names(
    item_features: ItemFeatures,
    user_features: UserFeatures,
) -> list[str]:
    """Return column names for the matrix produced by build_pair_features."""
    return (
        ["stage1_cf", "stage1_content", "stage1_outcome"]
        + [f"item__{n}" for n in item_features.feature_names]
        + [f"user__{n}" for n in user_features.feature_names]
    )
