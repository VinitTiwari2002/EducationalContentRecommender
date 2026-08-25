"""Hyperparameter tuning for SVD and the hybrid weights.

Strategy: build a temporal *tuning* sub-split from the primary training
window — i.e. train_tune ends *before* the primary train/test cutoff so
tuning never touches the primary test set. Grid-search on this sub-split
and return the chosen hyperparameters; the primary evaluation then re-fits
with those hyperparameters on the full primary training window.

Grids follow the design chapter:
    * SVD n_factors ∈ {20, 50, 100}
    * Hybrid (α, β, γ) on the 3-simplex in 0.2 steps (21 points)

Objective metric: NDCG@10 on the tuning validation window. NDCG is chosen
per the primary metric argument in the design chapter (Rendle et al. 2012),
and @10 aligns with the K we care about for the final report table.

The design chapter also lists a regularisation grid λ ∈ {0.01, 0.05, 0.1}
for SVD; that grid is not applicable to plain truncated SVD (which has no
explicit λ — the truncation is itself the regulariser via low-rank
approximation). The implementation chapter documents this deviation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .als import ALSRecommender
from .content import ContentRecommender
from .features import build_item_features
from .hybrid import HybridRecommender
from .metrics import evaluate
from .oulad import OULAD
from .svd import SVDRecommender


@dataclass
class TuningSplit:
    """Temporal train/val split used only for hyperparameter selection.

    tune_train and tune_val share the same student / item row/col index
    space so metrics on tune_val can be computed with the same helpers
    used for the primary evaluation.
    """

    tune_train: sparse.csr_matrix
    tune_val: sparse.csr_matrix
    student_index: np.ndarray
    item_index: np.ndarray
    tune_cutoff: int
    val_cutoff: int


def build_tuning_split(
    data: OULAD,
    student_index: np.ndarray,
    item_index: np.ndarray,
    primary_cutoff: int,
    val_fraction: float = 0.25,
) -> TuningSplit:
    """Sub-split the primary training window into tune_train / tune_val.

    Interactions with date <= primary_cutoff make up the primary training
    window. We further split them: the last val_fraction of dates in that
    window becomes tune_val; the rest is tune_train. This guarantees no
    date in tune_val exceeds primary_cutoff (so the primary test set is
    never involved) while still giving a temporally-honest validation set.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")

    vle = data.student_vle[["id_student", "id_site", "date", "sum_click"]]
    vle = vle[
        vle["id_student"].isin(student_index)
        & vle["id_site"].isin(item_index)
        & (vle["date"] <= primary_cutoff)
    ]
    tune_cutoff = int(np.quantile(vle["date"].to_numpy(), 1 - val_fraction))

    s_to_row = pd.Series(np.arange(len(student_index)), index=student_index)
    i_to_col = pd.Series(np.arange(len(item_index)), index=item_index)

    def _matrix(df: pd.DataFrame) -> sparse.csr_matrix:
        agg = df.groupby(["id_student", "id_site"], as_index=False)["sum_click"].sum()
        rows = s_to_row.loc[agg["id_student"]].to_numpy()
        cols = i_to_col.loc[agg["id_site"]].to_numpy()
        vals = agg["sum_click"].to_numpy(dtype=np.float32)
        return sparse.coo_matrix(
            (vals, (rows, cols)), shape=(len(student_index), len(item_index))
        ).tocsr()

    tune_train_df = vle[vle["date"] <= tune_cutoff]
    tune_val_df = vle[vle["date"] > tune_cutoff]
    return TuningSplit(
        tune_train=_matrix(tune_train_df),
        tune_val=_matrix(tune_val_df),
        student_index=student_index,
        item_index=item_index,
        tune_cutoff=tune_cutoff,
        val_cutoff=primary_cutoff,
    )


def simplex_grid(step: float = 0.2) -> list[tuple[float, float, float]]:
    """Enumerate (α, β, γ) triples on the 3-simplex at the given step."""
    if not 0 < step <= 1:
        raise ValueError("step must be in (0, 1]")
    n = int(round(1.0 / step))
    grid: list[tuple[float, float, float]] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            grid.append((round(i * step, 3), round(j * step, 3), round(k * step, 3)))
    # Drop the degenerate all-zeros triple (cannot occur with n=1/step)
    return [w for w in grid if sum(w) > 0]


def _relevant_per_user(mat: sparse.csr_matrix) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for r in range(mat.shape[0]):
        items = mat[r].indices.tolist()
        if items:
            out[r] = items
    return out


def _recommend_all(
    model, n_users: int, k: int, user_candidates: list[np.ndarray] | None
) -> dict[int, list[int]]:
    if user_candidates is None:
        return {u: model.recommend(u, k=k) for u in range(n_users)}
    return {
        u: model.recommend(u, k=k, candidates=user_candidates[u])
        for u in range(n_users)
    }


def tune_svd(
    tune: TuningSplit,
    user_candidates: list[np.ndarray] | None,
    n_factors_grid: tuple[int, ...] = (20, 50, 100),
    k: int = 10,
    random_state: int = 0,
) -> tuple[int, pd.DataFrame]:
    """Grid-search SVD n_factors. Returns (best_n_factors, results_df)."""
    relevant = _relevant_per_user(tune.tune_val)
    n_users = tune.tune_train.shape[0]
    rows = []
    for n_factors in n_factors_grid:
        model = SVDRecommender(n_factors=n_factors, random_state=random_state)
        model.fit(tune.tune_train)
        recs = _recommend_all(model, n_users=n_users, k=k, user_candidates=user_candidates)
        m = evaluate(recs, relevant, k=k)
        rows.append({"n_factors": n_factors, **m})
    df = pd.DataFrame(rows).sort_values(f"ndcg@{k}", ascending=False).reset_index(drop=True)
    return int(df.iloc[0]["n_factors"]), df


def tune_als(
    tune: TuningSplit,
    user_candidates: list[np.ndarray] | None,
    n_factors_grid: tuple[int, ...] = (32, 64, 128),
    regularization_grid: tuple[float, ...] = (0.01, 0.1),
    alpha_grid: tuple[float, ...] = (1.0, 10.0, 40.0),
    iterations: int = 15,
    k: int = 10,
    random_state: int = 0,
) -> tuple[dict, pd.DataFrame]:
    """Grid-search ALS n_factors, regularization and confidence-alpha on
    the tuning split. Alpha is Hu-Koren-Volinsky's confidence scale; the
    original paper uses alpha=40 for music, and the sensible grid spans
    weak (1) to strong (40) confidence. Returns (best_params_dict, df).
    """
    relevant = _relevant_per_user(tune.tune_val)
    n_users = tune.tune_train.shape[0]
    rows = []
    for n_factors in n_factors_grid:
        for reg in regularization_grid:
            for a in alpha_grid:
                model = ALSRecommender(
                    n_factors=n_factors,
                    regularization=reg,
                    iterations=iterations,
                    alpha=a,
                    random_state=random_state,
                )
                model.fit(tune.tune_train)
                recs = _recommend_all(model, n_users=n_users, k=k, user_candidates=user_candidates)
                m = evaluate(recs, relevant, k=k)
                rows.append({"n_factors": n_factors, "regularization": reg, "alpha": a, **m})
    df = pd.DataFrame(rows).sort_values(f"ndcg@{k}", ascending=False).reset_index(drop=True)
    best = {
        "n_factors": int(df.iloc[0]["n_factors"]),
        "regularization": float(df.iloc[0]["regularization"]),
        "alpha": float(df.iloc[0]["alpha"]),
    }
    return best, df


def tune_hybrid_weights(
    tune: TuningSplit,
    data: OULAD,
    user_candidates: list[np.ndarray] | None,
    cf_kind: str = "als",
    cf_params: dict | None = None,
    step: float = 0.2,
    k: int = 10,
    random_state: int = 0,
) -> tuple[tuple[float, float, float], pd.DataFrame]:
    """Grid-search (α, β, γ) on the simplex.

    Fits the CF component (SVD or ALS) and Content once, then sweeps
    hybrid weights via set_weights() to avoid re-fitting per config.
    Returns (best_weights, results_df).

    Parameters
    ----------
    cf_kind: "als" (default; the implementation-chapter choice) or "svd"
        (the design-chapter choice, retained for like-for-like comparison).
    cf_params: hyperparameters for the CF backbone. For als: n_factors,
        regularization, iterations, alpha. For svd: n_factors.
    """
    cf_params = cf_params or {}
    features = build_item_features(data, tune.tune_train, tune.item_index, tune.student_index)
    if cf_kind == "als":
        cf = ALSRecommender(
            n_factors=int(cf_params.get("n_factors", 64)),
            regularization=float(cf_params.get("regularization", 0.01)),
            iterations=int(cf_params.get("iterations", 15)),
            alpha=float(cf_params.get("alpha", 1.0)),
            random_state=random_state,
        )
    elif cf_kind == "svd":
        cf = SVDRecommender(
            n_factors=int(cf_params.get("n_factors", 50)),
            random_state=random_state,
        )
    else:
        raise ValueError(f"unknown cf_kind: {cf_kind!r}")
    cf.fit(tune.tune_train)
    content = ContentRecommender()
    content.fit(tune.tune_train, features.matrix)
    hybrid = HybridRecommender(alpha=1.0, beta=0.0, gamma=0.0, cf=cf, content=content)
    hybrid.fit(tune.tune_train, features.matrix, features.outcome_score, refit_components=False)

    relevant = _relevant_per_user(tune.tune_val)
    n_users = tune.tune_train.shape[0]
    grid = simplex_grid(step=step)
    rows = []
    for alpha, beta, gamma in grid:
        hybrid.set_weights(alpha, beta, gamma)
        recs = _recommend_all(hybrid, n_users=n_users, k=k, user_candidates=user_candidates)
        m = evaluate(recs, relevant, k=k)
        rows.append({"alpha": alpha, "beta": beta, "gamma": gamma, **m})
    df = pd.DataFrame(rows).sort_values(f"ndcg@{k}", ascending=False).reset_index(drop=True)
    best = (
        float(df.iloc[0]["alpha"]),
        float(df.iloc[0]["beta"]),
        float(df.iloc[0]["gamma"]),
    )
    return best, df
