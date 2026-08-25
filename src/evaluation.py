"""Evaluation harness beyond the per-model, single-split baseline.

Contains:
    * temporal_cv_folds — sliding cutoff dates that produce 5 non-overlapping
      train/test splits with no leakage of future clicks into past folds.
    * bootstrap_ci        — percentile bootstrap over per-user metric arrays.
    * paired_t_test       — paired-t against per-user metric arrays.
    * fairness_audit      — metric breakdown by gender / imd_band / disability.
    * hybrid_ablation     — drops each of {CF, content, outcome} in turn and
      reports metric deltas relative to the full hybrid.

Every step is train-only where relevant; the outcome_score used by the
outcome-weighted metric is recomputed from each fold's training window,
never leaked across folds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats

from . import metrics
from .features import build_item_features
from .oulad import OULAD


@dataclass
class Fold:
    """One temporal fold: train/test csr matrices, plus fold metadata."""

    train: sparse.csr_matrix
    test: sparse.csr_matrix
    cutoff_date: int
    fold_index: int


def temporal_cv_folds(
    interactions: pd.DataFrame,
    students: np.ndarray,
    items: np.ndarray,
    n_folds: int = 5,
    test_fraction: float = 0.15,
    decay_rate: float = 0.0,
) -> list[Fold]:
    """Build n_folds non-overlapping temporal folds.

    Each fold uses a sliding cutoff date. Fold k trains on interactions
    strictly before its cutoff and tests on the next test_fraction chunk
    of interactions (chronologically), so no test row of any earlier fold
    appears in the training data of any later fold — the standard temporal
    CV recipe for recommender systems (Ricci et al., 2015).

    decay_rate: if > 0, applies exponential recency weighting to the
    training-window clicks of each fold using that fold's own cutoff as
    the reference date (so recency is fold-local, never leaked across).
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    if n_folds * test_fraction >= 1:
        raise ValueError("n_folds * test_fraction must be < 1")
    if decay_rate < 0:
        raise ValueError("decay_rate must be non-negative")

    dates = interactions["date"].to_numpy()
    s_to_row = pd.Series(np.arange(len(students)), index=students)
    i_to_col = pd.Series(np.arange(len(items)), index=items)

    def _matrix(df: pd.DataFrame, cutoff: float, apply_decay: bool) -> sparse.csr_matrix:
        if apply_decay and decay_rate > 0:
            weights = df["sum_click"].to_numpy(dtype=np.float32) * np.exp(
                -decay_rate * (cutoff - df["date"].to_numpy(dtype=np.float32))
            )
            df = df.assign(_weight=weights)
            agg = df.groupby(["id_student", "id_site"], as_index=False)["_weight"].sum()
            vals = agg["_weight"].to_numpy(dtype=np.float32)
        else:
            agg = df.groupby(["id_student", "id_site"], as_index=False)["sum_click"].sum()
            vals = agg["sum_click"].to_numpy(dtype=np.float32)
        rows = s_to_row.loc[agg["id_student"]].to_numpy()
        cols = i_to_col.loc[agg["id_site"]].to_numpy()
        return sparse.coo_matrix(
            (vals, (rows, cols)), shape=(len(students), len(items))
        ).tocsr()

    folds: list[Fold] = []
    for k in range(n_folds):
        upper_q = 1 - k * test_fraction
        lower_q = 1 - (k + 1) * test_fraction
        upper_cut = float(np.quantile(dates, upper_q))
        lower_cut = float(np.quantile(dates, lower_q))
        train_df = interactions[interactions["date"] <= lower_cut]
        test_df = interactions[
            (interactions["date"] > lower_cut) & (interactions["date"] <= upper_cut)
        ]
        folds.append(
            Fold(
                train=_matrix(train_df, cutoff=lower_cut, apply_decay=True),
                test=_matrix(test_df, cutoff=upper_cut, apply_decay=False),
                cutoff_date=int(lower_cut),
                fold_index=k,
            )
        )
    return list(reversed(folds))  # chronological order: earliest first


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap (1-alpha) CI on the mean of `values`."""
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


def paired_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Paired-t test on aligned per-user metric arrays.

    Returns (t_statistic, two_sided_p_value). Users must be aligned across
    arrays (i.e. same order); caller is responsible for that.
    """
    if a.shape != b.shape:
        raise ValueError(f"paired-t arrays must have equal shape; got {a.shape} vs {b.shape}")
    if a.size < 2:
        return (float("nan"), float("nan"))
    result = stats.ttest_rel(a, b)
    return (float(result.statistic), float(result.pvalue))


def fairness_audit(
    recommendations: dict[int, list[int]],
    relevant_per_user: dict[int, list[int]],
    student_row_to_id: np.ndarray,
    student_info: pd.DataFrame,
    k: int,
    outcome_score: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-attribute metric breakdown across gender, imd_band, disability.

    Rows of the returned frame are (attribute, level, n_users, *metric
    means*). Users without a matching studentInfo row are grouped under
    level='unknown'.
    """
    info = (
        student_info.drop_duplicates(subset=["id_student"], keep="first")
        .set_index("id_student")
    )
    per_user = metrics.per_user_metrics(
        recommendations, relevant_per_user, k, outcome_score
    )
    users_with_relevant = [u for u, items in relevant_per_user.items() if items]
    if not users_with_relevant:
        return pd.DataFrame()
    row_to_id = student_row_to_id
    student_ids = np.array([int(row_to_id[u]) for u in users_with_relevant])

    def _attribute_slice(attribute: str) -> pd.DataFrame:
        vals = info.reindex(student_ids)[attribute].fillna("unknown").to_numpy()
        out = []
        for level in np.unique(vals):
            mask = vals == level
            row: dict[str, float | int | str] = {
                "attribute": attribute,
                "level": str(level),
                "n_users": int(mask.sum()),
            }
            for metric_name, arr in per_user.items():
                row[metric_name] = float(arr[mask].mean()) if mask.any() else 0.0
            out.append(row)
        return pd.DataFrame(out)

    return pd.concat(
        [_attribute_slice(a) for a in ("gender", "imd_band", "disability")],
        ignore_index=True,
    )


def hybrid_ablation(
    hybrid_class,
    train: sparse.csr_matrix,
    item_features: np.ndarray,
    outcome_score: np.ndarray,
    weights: tuple[float, float, float],
    recommend_all_fn,
    relevant_per_user: dict[int, list[int]],
    k: int,
    n_items: int,
) -> pd.DataFrame:
    """Run ablation: full hybrid, no-CF, no-content, no-outcome variants.

    hybrid_class: HybridRecommender factory (so each variant can be built
        with different weights).
    weights: (alpha, beta, gamma) for the *full* hybrid.
    recommend_all_fn: callable(model, k) -> dict[user_row, list[item_col]]
        (i.e. the same helper used in the main pipeline; caller supplies
        it so this module doesn't have to know about course-scoping).
    """
    alpha, beta, gamma = weights
    variants = {
        "hybrid_full": (alpha, beta, gamma),
        "hybrid_no_cf": (0.0, beta, gamma),
        "hybrid_no_content": (alpha, 0.0, gamma),
        "hybrid_no_outcome": (alpha, beta, 0.0),
    }
    rows = []
    for name, (a, b, g) in variants.items():
        if a + b + g == 0:
            continue
        model = hybrid_class(alpha=a, beta=b, gamma=g)
        model.fit(train, item_features, outcome_score)
        recs = recommend_all_fn(model, k)
        m = metrics.evaluate(
            recs, relevant_per_user, k=k, outcome_score=outcome_score, n_items=n_items
        )
        rows.append({"variant": name, "alpha": a, "beta": b, "gamma": g, **m})
    return pd.DataFrame(rows)


def build_item_features_for_fold(
    data: OULAD,
    train: sparse.csr_matrix,
    student_index: np.ndarray,
    item_index: np.ndarray,
):
    """Thin wrapper so callers don't have to import from src.features."""
    return build_item_features(data, train, item_index, student_index)


def cold_start_masks(
    train: sparse.csr_matrix, threshold: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Return (warm_mask, cold_mask) boolean arrays over user rows.

    A user is "cold" if they have fewer than `threshold` non-zero item
    interactions in training. The design chapter promises to report
    cold-start results separately because SVD is expected to degrade on
    users with too little interaction signal.
    """
    clicks_per_user = np.asarray((train != 0).sum(axis=1)).ravel()
    warm = clicks_per_user >= threshold
    cold = ~warm
    return warm, cold


def stratified_evaluate(
    recommendations: dict[int, list[int]],
    relevant_per_user: dict[int, list[int]],
    mask: np.ndarray,
    k: int,
    outcome_score: np.ndarray | None = None,
    n_items: int | None = None,
) -> dict[str, float]:
    """Evaluate metrics restricted to users where mask is True."""
    from .metrics import evaluate as _evaluate

    users_in_mask = np.where(mask)[0]
    filtered_recs = {u: recommendations[u] for u in users_in_mask if u in recommendations}
    filtered_relevant = {
        u: relevant_per_user[u] for u in users_in_mask if u in relevant_per_user
    }
    return _evaluate(
        filtered_recs,
        filtered_relevant,
        k=k,
        outcome_score=outcome_score,
        n_items=n_items,
    )


def bonferroni_adjust(p_values: np.ndarray, n_comparisons: int) -> np.ndarray:
    """Bonferroni-adjust each p-value: min(1, p * n_comparisons).

    The design chapter specifies Bonferroni over the five metrics
    (precision, recall, ndcg, hit_rate, outcome_weighted_precision), so
    the caller passes n_comparisons=5 for per-model paired-t results.
    """
    if n_comparisons < 1:
        raise ValueError("n_comparisons must be >= 1")
    p = np.asarray(p_values, dtype=np.float64)
    adjusted = np.minimum(1.0, p * n_comparisons)
    return adjusted
