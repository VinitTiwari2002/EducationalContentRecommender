"""Item feature extractor.

Builds a dense (n_items, n_features) matrix aligned with the column space of
the interaction matrix (i.e. the same ordering as `item_index` on Split).

Features per item:
    1. activity_type one-hot (from vle table; ~20 categories)
    2. week_from    (min-max normalised; NaN → median, "week_known" flag)
    3. week_to      (min-max normalised; NaN → median, "week_known" flag)
    4. log1p(global_access_count)          — train-only
    5. mean_score_of_accessers             — train-only

Rules (per the design chapter and Ricci et al., 2015):
    * Every feature that depends on interactions or assessments is computed
      strictly from the *train* portion of the split. This prevents leakage
      from future assessment scores or clicks into features that a
      recommender then uses to score held-out items.
    * Items in `item_index` that have no vle row (should not happen for
      OULAD, but is guarded) receive zero features except the mean-score
      fallback (global training-set mean).
    * A NaN week is filled with the training-set median week and a
      "week_known" flag distinguishes filled from real values, so the
      recommender can learn whether "no-week" resources behave differently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .oulad import OULAD


@dataclass
class ItemFeatures:
    """Item feature matrix + metadata for interpretation.

    matrix has shape (n_items, n_features), aligned with item_index (col
    ordering matches the interaction-matrix columns).
    outcome_score[i] is the mean assessment score of training-set users who
    accessed item i, in [0, 100]; this is what the outcome-weighted
    precision metric consumes.
    """

    matrix: np.ndarray
    feature_names: list[str]
    outcome_score: np.ndarray


def build_item_features(
    data: OULAD,
    train: sparse.csr_matrix,
    item_index: np.ndarray,
    student_index: np.ndarray,
) -> ItemFeatures:
    """Compute item features aligned with the item_index column ordering.

    Parameters mirror the Split contract: `train` is the training-window
    interaction matrix (rows = students, cols = items) and item_index /
    student_index map matrix positions back to OULAD ids.
    """
    n_items = len(item_index)
    item_id_to_col = pd.Series(np.arange(n_items), index=item_index)

    vle_rows = _align_vle_rows(data.vle, item_index)
    activity_onehot, activity_types = _activity_type_onehot(vle_rows)
    week_from, week_to, week_known = _week_features(vle_rows)
    access_count = _log_access_count(train)
    outcome_score, mean_score_feature = _outcome_features(
        data, train, student_index, item_id_to_col, n_items
    )

    matrix = np.hstack(
        [
            activity_onehot,
            week_from[:, None],
            week_to[:, None],
            week_known[:, None],
            access_count[:, None],
            mean_score_feature[:, None],
        ]
    ).astype(np.float32)

    feature_names = (
        [f"activity_type={a}" for a in activity_types]
        + ["week_from_norm", "week_to_norm", "week_known"]
        + ["log_access_count", "mean_score_of_accessers"]
    )
    return ItemFeatures(matrix=matrix, feature_names=feature_names, outcome_score=outcome_score)


def _align_vle_rows(vle: pd.DataFrame, item_index: np.ndarray) -> pd.DataFrame:
    """Return one row of vle per item_index entry, in item_index order.

    An id_site can appear in vle multiple times (once per presentation it
    belongs to); we take the first row since activity_type and week_from/to
    are properties of the resource, not of the presentation.
    """
    vle_first = vle.drop_duplicates(subset=["id_site"], keep="first").set_index("id_site")
    aligned = vle_first.reindex(item_index)
    return aligned


def _activity_type_onehot(vle_rows: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """One-hot of activity_type; unknown/missing rows become all-zero."""
    types = sorted(vle_rows["activity_type"].dropna().unique().tolist())
    n_items = len(vle_rows)
    onehot = np.zeros((n_items, len(types)), dtype=np.float32)
    for j, t in enumerate(types):
        onehot[:, j] = (vle_rows["activity_type"].to_numpy() == t).astype(np.float32)
    return onehot, types


def _week_features(vle_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise week_from and week_to to [0, 1]; NaN → median; add flag."""
    wf = pd.to_numeric(vle_rows["week_from"], errors="coerce").to_numpy(dtype=np.float32)
    wt = pd.to_numeric(vle_rows["week_to"], errors="coerce").to_numpy(dtype=np.float32)
    week_known = (~np.isnan(wf) & ~np.isnan(wt)).astype(np.float32)

    def _fill_and_normalise(arr: np.ndarray) -> np.ndarray:
        median = float(np.nanmedian(arr)) if np.isfinite(np.nanmedian(arr)) else 0.0
        filled = np.where(np.isnan(arr), median, arr)
        max_val = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 0.0
        return filled / max_val if max_val > 0 else filled

    return _fill_and_normalise(wf), _fill_and_normalise(wt), week_known


def _log_access_count(train: sparse.csr_matrix) -> np.ndarray:
    """log1p(sum of clicks per item column), then min-max normalise so the
    feature sits on a comparable scale to the other features."""
    totals = np.asarray(train.sum(axis=0)).ravel().astype(np.float32)
    logged = np.log1p(totals)
    max_val = float(logged.max()) if logged.size > 0 else 0.0
    return logged / max_val if max_val > 0 else logged


def _outcome_features(
    data: OULAD,
    train: sparse.csr_matrix,
    student_index: np.ndarray,
    item_id_to_col: pd.Series,
    n_items: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean assessment score of training-set users who accessed each
    item. Returns two arrays: the raw outcome_score (in [0, 100]; used by
    the outcome-weighted precision metric) and its normalised version
    (in [0, 1]; used as an item feature).

    A student's assessment score is their mean score across all
    studentAssessment rows (in the range [0, 100]).
    """
    sa = data.student_assessment.dropna(subset=["score"])
    student_mean_score = sa.groupby("id_student")["score"].mean()
    student_row = pd.Series(np.arange(len(student_index)), index=student_index)
    common = student_mean_score.index.intersection(student_index)
    mean_score_by_row = np.full(len(student_index), np.nan, dtype=np.float32)
    if len(common) > 0:
        mean_score_by_row[student_row.loc[common].to_numpy()] = student_mean_score.loc[
            common
        ].to_numpy(dtype=np.float32)

    train_coo = train.tocoo()
    per_item_scores: dict[int, list[float]] = {}
    for r, c in zip(train_coo.row, train_coo.col):
        s = mean_score_by_row[r]
        if not np.isnan(s):
            per_item_scores.setdefault(int(c), []).append(float(s))

    global_mean = float(np.nanmean(mean_score_by_row)) if len(common) > 0 else 0.0
    outcome_score = np.full(n_items, global_mean, dtype=np.float32)
    for col, scores in per_item_scores.items():
        outcome_score[col] = float(np.mean(scores))

    normalised = outcome_score / 100.0
    return outcome_score, normalised
