"""End-to-end pipeline runner.

Usage:
    python -m src.pipeline                     # full run (single split + CV)
    python -m src.pipeline --rebuild-split     # force rebuild of cached split
    python -m src.pipeline --k 5 10 20         # evaluate at multiple K
    python -m src.pipeline --no-course-scoping # diagnostic: skip per-user
                                                  candidate restriction
    python -m src.pipeline --no-cv             # skip 5-fold temporal CV
                                                  (fast dev-loop mode)

The pipeline runs in three passes:

    1. Single-split evaluation of all five models (Random, Popularity, SVD,
       Content, Hybrid) with per-user metrics and 95% bootstrap CIs.
    2. Popularity-bias metrics (catalogue coverage + recommendation Gini),
       fairness audit by gender/imd_band/disability, and hybrid ablation.
    3. 5-fold temporal cross-validation of all five models (unless
       --no-cv), giving per-fold means for a proper paired-t test.

Outputs (under evaluation/):
    baseline_results{,_no_course_scoping}.csv
    metrics_with_ci.csv        # per-model K x (metric, mean, ci_lo, ci_hi)
    popularity_bias.csv        # per-model coverage + Gini
    fairness_audit.csv         # per-attribute breakdown for the hybrid
    hybrid_ablation.csv        # full vs no-CF/no-content/no-outcome
    cv_results.csv             # per-fold per-model per-K metrics
    paired_t_tests.csv         # hybrid vs each baseline, per metric
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from . import oulad
from .als import ALSRecommender
from .baselines import PopularityRecommender, RandomRecommender
from .content import ContentRecommender
from .evaluation import (
    bonferroni_adjust,
    bootstrap_ci,
    cold_start_masks,
    fairness_audit,
    hybrid_ablation,
    paired_t_test,
    stratified_evaluate,
    temporal_cv_folds,
)
from .features import ItemFeatures, build_item_features
from .gated_hybrid import GatedHybridRecommender
from .hybrid import HybridRecommender
from .metrics import evaluate, per_user_metrics
from .persistence import DEFAULT_MODELS_DIR, ServingContext, save_models
from .preprocess import Split, build_split
from .svd import SVDRecommender
from .tuning import build_tuning_split, tune_als, tune_hybrid_weights, tune_svd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_DIR = PROJECT_ROOT / "evaluation"

# Default hyperparameters used when tuning is skipped (--no-tuning).
DEFAULT_HYBRID_WEIGHTS = (0.5, 0.3, 0.2)
DEFAULT_SVD_N_FACTORS = 50
DEFAULT_ALS_N_FACTORS = 64
DEFAULT_ALS_REGULARIZATION = 0.01
DEFAULT_ALS_ITERATIONS = 15
DEFAULT_ALS_ALPHA = 1.0
DEFAULT_COLD_START_THRESHOLD = 10


def _relevant_per_user(test: sparse.csr_matrix) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for user_row in range(test.shape[0]):
        items = test[user_row].indices.tolist()
        if items:
            out[user_row] = items
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


def _make_als(als_params: dict, random_state: int) -> ALSRecommender:
    return ALSRecommender(
        n_factors=int(als_params.get("n_factors", DEFAULT_ALS_N_FACTORS)),
        regularization=float(
            als_params.get("regularization", DEFAULT_ALS_REGULARIZATION)
        ),
        iterations=int(als_params.get("iterations", DEFAULT_ALS_ITERATIONS)),
        alpha=float(als_params.get("alpha", DEFAULT_ALS_ALPHA)),
        random_state=random_state,
    )


def _fit_all_models(
    train: sparse.csr_matrix,
    features: ItemFeatures,
    svd_n_factors: int,
    hybrid_weights: tuple[float, float, float],
    als_params: dict,
    cold_start_threshold: int = DEFAULT_COLD_START_THRESHOLD,
) -> dict[str, object]:
    """Fit all seven recommenders on `train`. Item features / outcome are
    supplied by the caller so that when the pipeline runs under CV each
    fold gets its own train-only features.

    The full roster (per the design chapter + implementation extensions):
        Random, Popularity, SVD, ALS, Content, Hybrid, GatedHybrid.
    Hybrid uses ALS as its CF component (SVD contributed negatively in
    tuning — see the implementation chapter for the analysis); SVD is
    retained as a standalone baseline for direct CF comparison.
    """
    random = RandomRecommender(seed=0)
    random.fit(train)
    popularity = PopularityRecommender()
    popularity.fit(train)
    svd = SVDRecommender(n_factors=svd_n_factors, random_state=0)
    svd.fit(train)
    als = _make_als(als_params, random_state=0)
    als.fit(train)
    content = ContentRecommender()
    content.fit(train, features.matrix)
    alpha, beta, gamma = hybrid_weights
    hybrid = HybridRecommender(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        cf=_make_als(als_params, random_state=1),
        content=ContentRecommender(),
    )
    hybrid.fit(train, features.matrix, features.outcome_score)
    gated = GatedHybridRecommender(
        cold_model=PopularityRecommender(),
        warm_model=HybridRecommender(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            cf=_make_als(als_params, random_state=2),
            content=ContentRecommender(),
        ),
        threshold=cold_start_threshold,
    )
    gated.fit(train, features.matrix, features.outcome_score)
    return {
        "Random": random,
        "Popularity": popularity,
        "SVD": svd,
        "ALS": als,
        "Content": content,
        "Hybrid": hybrid,
        "GatedHybrid": gated,
    }


def _default_als_params() -> dict:
    return {
        "n_factors": DEFAULT_ALS_N_FACTORS,
        "regularization": DEFAULT_ALS_REGULARIZATION,
        "iterations": DEFAULT_ALS_ITERATIONS,
        "alpha": DEFAULT_ALS_ALPHA,
    }


def _single_split_run(
    split: Split,
    features: ItemFeatures,
    course_scoping: bool,
    k_values: list[int],
    svd_n_factors: int,
    hybrid_weights: tuple[float, float, float],
    als_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[int, dict[int, list[int]]]]]:
    """Fit + evaluate all models on the single primary split.

    Returns three artefacts:
        - results: model x K x metric summary
        - results_with_ci: same but with 95% bootstrap CIs on per-user means
        - all_recs: nested dict model -> K -> recommendations, kept for
          downstream popularity-bias / fairness / paired-t work
    """
    n_users = split.train.shape[0]
    relevant = _relevant_per_user(split.test)
    candidates = split.user_candidates if course_scoping else None

    print("Fitting all seven models on the primary split...")
    models = _fit_all_models(
        split.train, features, svd_n_factors, hybrid_weights, als_params
    )

    rows_summary: list[dict] = []
    rows_ci: list[dict] = []
    all_recs: dict[str, dict[int, dict[int, list[int]]]] = {}
    n_items = split.train.shape[1]

    for model_name, model in models.items():
        print(f"  {model_name}: generating recommendations...")
        all_recs[model_name] = {}
        for k in k_values:
            recs = _recommend_all(model, n_users=n_users, k=k, user_candidates=candidates)
            all_recs[model_name][k] = recs
            m = evaluate(
                recs,
                relevant,
                k=k,
                outcome_score=features.outcome_score,
                n_items=n_items,
            )
            rows_summary.append({"model": model_name, "k": k, **m})

            per_user = per_user_metrics(recs, relevant, k, outcome_score=features.outcome_score)
            for metric_name, arr in per_user.items():
                lo, hi = bootstrap_ci(arr, n_boot=1000, seed=0)
                rows_ci.append(
                    {
                        "model": model_name,
                        "k": k,
                        "metric": metric_name,
                        "mean": float(arr.mean()) if arr.size else 0.0,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "n_users": int(arr.size),
                    }
                )
            print(f"    K={k}: precision={m[f'precision@{k}']:.4f}, "
                  f"ndcg={m[f'ndcg@{k}']:.4f}, owp={m.get(f'outcome_weighted_precision@{k}', float('nan')):.4f}")

    return pd.DataFrame(rows_summary), pd.DataFrame(rows_ci), all_recs


def _paired_t_hybrid_vs_baselines(
    all_recs: dict[str, dict[int, dict[int, list[int]]]],
    relevant: dict[int, list[int]],
    outcome: np.ndarray,
    k: int,
) -> pd.DataFrame:
    """Paired-t test of Hybrid against every other model at a given K."""
    hybrid_per_user = per_user_metrics(all_recs["Hybrid"][k], relevant, k, outcome_score=outcome)
    rows = []
    others = [m for m in ("Random", "Popularity", "SVD", "ALS", "Content", "GatedHybrid") if m in all_recs]
    for other in others:
        other_per_user = per_user_metrics(all_recs[other][k], relevant, k, outcome_score=outcome)
        for metric_name in hybrid_per_user:
            t, p = paired_t_test(hybrid_per_user[metric_name], other_per_user[metric_name])
            rows.append(
                {
                    "hybrid_vs": other,
                    "k": k,
                    "metric": metric_name,
                    "t": t,
                    "p_value": p,
                    "hybrid_mean": float(hybrid_per_user[metric_name].mean()),
                    "other_mean": float(other_per_user[metric_name].mean()),
                }
            )
    return pd.DataFrame(rows)


def _cross_validation(
    data: oulad.OULAD,
    split: Split,
    k_values: list[int],
    svd_n_factors: int,
    hybrid_weights: tuple[float, float, float],
    als_params: dict,
    decay_rate: float,
    n_folds: int = 5,
    course_scoping: bool = True,
) -> pd.DataFrame:
    """Run n_folds temporal CV. Each fold rebuilds item features from its
    own training window so the outcome signal never leaks across folds."""
    print(f"\n=== 5-fold temporal cross-validation (n_folds={n_folds}) ===")
    student_index = split.student_index
    item_index = split.item_index
    interactions = data.student_vle[["id_student", "id_site", "date", "sum_click"]]
    interactions = interactions[
        interactions["id_student"].isin(student_index)
        & interactions["id_site"].isin(item_index)
    ]
    folds = temporal_cv_folds(
        interactions,
        student_index,
        item_index,
        n_folds=n_folds,
        decay_rate=decay_rate,
    )

    rows = []
    for fold in folds:
        print(f"\nFold {fold.fold_index + 1}/{n_folds} — cutoff_date={fold.cutoff_date}, "
              f"train nnz={fold.train.nnz:,}, test nnz={fold.test.nnz:,}")
        features = build_item_features(data, fold.train, item_index, student_index)
        models = _fit_all_models(
            fold.train, features, svd_n_factors, hybrid_weights, als_params
        )
        relevant = _relevant_per_user(fold.test)
        candidates = split.user_candidates if course_scoping else None
        n_users = fold.train.shape[0]
        n_items = fold.train.shape[1]
        for model_name, model in models.items():
            for k in k_values:
                recs = _recommend_all(model, n_users, k, candidates)
                m = evaluate(
                    recs,
                    relevant,
                    k=k,
                    outcome_score=features.outcome_score,
                    n_items=n_items,
                )
                rows.append({"fold": fold.fold_index, "model": model_name, "k": k, **m})
    return pd.DataFrame(rows)


def _popularity_bias(
    all_recs: dict[str, dict[int, dict[int, list[int]]]], k: int, n_items: int
) -> pd.DataFrame:
    """Catalogue coverage + Gini for each model at a given K."""
    from .metrics import catalogue_coverage, recommendation_gini

    rows = []
    for model_name, by_k in all_recs.items():
        recs = by_k[k]
        rows.append(
            {
                "model": model_name,
                "k": k,
                "catalogue_coverage": catalogue_coverage(recs, n_items),
                "recommendation_gini": recommendation_gini(recs),
            }
        )
    return pd.DataFrame(rows)


def _ablation(
    split: Split,
    features: ItemFeatures,
    course_scoping: bool,
    k: int,
    hybrid_weights: tuple[float, float, float],
) -> pd.DataFrame:
    """Ablation: full hybrid vs no-CF, no-content, no-outcome."""
    n_users = split.train.shape[0]
    candidates = split.user_candidates if course_scoping else None
    relevant = _relevant_per_user(split.test)

    def _recommend_all_local(model, k_local):
        return _recommend_all(model, n_users, k_local, candidates)

    return hybrid_ablation(
        HybridRecommender,
        split.train,
        features.matrix,
        features.outcome_score,
        weights=hybrid_weights,
        recommend_all_fn=_recommend_all_local,
        relevant_per_user=relevant,
        k=k,
        n_items=split.train.shape[1],
    )


def _cold_start_split_run(
    split: Split,
    features: ItemFeatures,
    all_recs: dict[str, dict[int, dict[int, list[int]]]],
    k: int,
    threshold: int = 10,
) -> pd.DataFrame:
    """Per-model cold-start vs warm-user metric breakdown at K.

    Uses the already-computed recommendation dicts so we don't need to
    refit anything — cold-start evaluation is purely a stratification
    over the primary-split results.
    """
    warm_mask, cold_mask = cold_start_masks(split.train, threshold=threshold)
    relevant = _relevant_per_user(split.test)
    n_items = split.train.shape[1]
    rows = []
    for model_name, by_k in all_recs.items():
        recs = by_k[k]
        for stratum_name, mask in (("warm", warm_mask), ("cold", cold_mask)):
            m = stratified_evaluate(
                recs,
                relevant,
                mask=mask,
                k=k,
                outcome_score=features.outcome_score,
                n_items=n_items,
            )
            rows.append(
                {
                    "model": model_name,
                    "stratum": stratum_name,
                    "threshold": threshold,
                    **m,
                }
            )
    return pd.DataFrame(rows)


def run(
    rebuild_split: bool,
    k_values: list[int],
    course_scoping: bool = True,
    run_cv: bool = True,
    run_tuning: bool = True,
    cold_start_threshold: int = 10,
    hybrid_grid_step: float = 0.2,
    decay_rate: float = 0.0,
    persist_models: bool = False,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> dict[str, pd.DataFrame]:
    cache_exists = (PROCESSED_DIR / "train.npz").exists() and (
        PROCESSED_DIR / "user_candidates.npy"
    ).exists()

    if rebuild_split or not cache_exists:
        print("Loading OULAD CSVs...")
        data = oulad.load()
        print("Summary:")
        print(data.summary().to_string(index=False))
        print(
            f"Building temporal split with course-scoped candidates "
            f"(decay_rate={decay_rate})..."
        )
        split = build_split(data, decay_rate=decay_rate)
        split.save(PROCESSED_DIR)
        print(f"Split saved to {PROCESSED_DIR}; cutoff_date={split.cutoff_date}")
    else:
        print(f"Loading cached split from {PROCESSED_DIR}")
        split = Split.load(PROCESSED_DIR)
        print("Loading OULAD CSVs for feature construction + fairness audit...")
        data = oulad.load()

    candidate_sizes = np.array([len(c) for c in split.user_candidates])
    print(
        f"train shape={split.train.shape}, nnz={split.train.nnz:,}; "
        f"test shape={split.test.shape}, nnz={split.test.nnz:,}; "
        f"cutoff_date={split.cutoff_date}"
    )
    print(
        f"user candidate set sizes: "
        f"min={candidate_sizes.min()}, median={int(np.median(candidate_sizes))}, "
        f"mean={candidate_sizes.mean():.1f}, max={candidate_sizes.max()}"
    )
    if not course_scoping:
        print("\n*** DIAGNOSTIC MODE: course-scoping DISABLED ***\n")

    print("\nBuilding item features (activity_type, week, log-access, outcome)...")
    features = build_item_features(data, split.train, split.item_index, split.student_index)
    print(f"  item feature matrix: shape={features.matrix.shape}, "
          f"{len(features.feature_names)} features")

    svd_n_factors = DEFAULT_SVD_N_FACTORS
    hybrid_weights = DEFAULT_HYBRID_WEIGHTS
    als_params = _default_als_params()
    svd_tuning_df = pd.DataFrame()
    als_tuning_df = pd.DataFrame()
    hybrid_tuning_df = pd.DataFrame()
    if run_tuning:
        print("\n=== Pass 0: hyperparameter tuning on temporal sub-split ===")
        tune = build_tuning_split(
            data, split.student_index, split.item_index, primary_cutoff=split.cutoff_date
        )
        print(
            f"  tuning split: tune_train nnz={tune.tune_train.nnz:,}, "
            f"tune_val nnz={tune.tune_val.nnz:,}, tune_cutoff={tune.tune_cutoff}"
        )
        candidates = split.user_candidates if course_scoping else None
        print("  Grid-searching SVD n_factors...")
        svd_n_factors, svd_tuning_df = tune_svd(
            tune,
            user_candidates=candidates,
            n_factors_grid=(20, 50, 100),
        )
        print(f"    best SVD n_factors={svd_n_factors}")
        print("  Grid-searching ALS n_factors × regularization × alpha...")
        best_als, als_tuning_df = tune_als(
            tune,
            user_candidates=candidates,
            n_factors_grid=(32, 64, 128),
            regularization_grid=(0.01, 0.1),
            alpha_grid=(1.0, 10.0, 40.0),
            iterations=DEFAULT_ALS_ITERATIONS,
        )
        als_params = {**als_params, **best_als}
        print(
            f"    best ALS n_factors={best_als['n_factors']}, "
            f"regularization={best_als['regularization']}, "
            f"alpha={best_als['alpha']}"
        )
        print("  Grid-searching hybrid weights (α, β, γ) with ALS backbone...")
        hybrid_weights, hybrid_tuning_df = tune_hybrid_weights(
            tune,
            data,
            user_candidates=candidates,
            cf_kind="als",
            cf_params=als_params,
            step=hybrid_grid_step,
        )
        print(f"    best hybrid weights={hybrid_weights}")

    print("\n=== Pass 1: single-split evaluation with bootstrap CIs ===")
    summary, ci, all_recs = _single_split_run(
        split,
        features,
        course_scoping,
        k_values,
        svd_n_factors,
        hybrid_weights,
        als_params,
    )

    print("\n=== Pass 2: popularity bias, fairness audit, ablation, cold-start ===")
    k_primary = 10 if 10 in k_values else k_values[0]
    pop_bias = _popularity_bias(all_recs, k=k_primary, n_items=split.train.shape[1])
    fairness = fairness_audit(
        all_recs["Hybrid"][k_primary],
        _relevant_per_user(split.test),
        student_row_to_id=split.student_index,
        student_info=data.student_info,
        k=k_primary,
        outcome_score=features.outcome_score,
    )
    ablation = _ablation(split, features, course_scoping, k=k_primary, hybrid_weights=hybrid_weights)
    paired_t = _paired_t_hybrid_vs_baselines(
        all_recs, _relevant_per_user(split.test), features.outcome_score, k=k_primary
    )
    # Bonferroni-adjust p-values across the 5 metrics (per design chapter).
    paired_t["p_adjusted"] = bonferroni_adjust(paired_t["p_value"].to_numpy(), n_comparisons=5)
    cold_start = _cold_start_split_run(
        split, features, all_recs, k=k_primary, threshold=cold_start_threshold
    )

    cv_results = pd.DataFrame()
    if run_cv:
        cv_results = _cross_validation(
            data,
            split,
            k_values,
            svd_n_factors,
            hybrid_weights,
            als_params,
            decay_rate=decay_rate,
            n_folds=5,
            course_scoping=course_scoping,
        )

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if course_scoping else "_no_course_scoping"
    outputs = {
        f"baseline_results{suffix}": summary,
        f"metrics_with_ci{suffix}": ci,
        f"popularity_bias{suffix}": pop_bias,
        f"fairness_audit{suffix}": fairness,
        f"hybrid_ablation{suffix}": ablation,
        f"paired_t_tests{suffix}": paired_t,
        f"cold_start_results{suffix}": cold_start,
    }
    if run_tuning:
        outputs[f"tuning_svd{suffix}"] = svd_tuning_df
        outputs[f"tuning_als{suffix}"] = als_tuning_df
        outputs[f"tuning_hybrid{suffix}"] = hybrid_tuning_df
    if run_cv:
        outputs[f"cv_results{suffix}"] = cv_results
    for name, df in outputs.items():
        out_path = EVAL_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path}  ({len(df)} rows)")

    if persist_models:
        print("\n=== Persisting fitted models for serving ===")
        # Refit the full-catalogue models on the primary split so the
        # persisted artefacts match the numbers in Table 5.1.
        models_to_persist = _fit_all_models(
            split.train, features, svd_n_factors, hybrid_weights, als_params
        )
        context = ServingContext(
            student_index=split.student_index,
            item_index=split.item_index,
            user_candidates=list(split.user_candidates),
            outcome_score=features.outcome_score,
            cutoff_date=int(split.cutoff_date),
        )
        metadata = {
            "svd_n_factors": svd_n_factors,
            "hybrid_weights": list(hybrid_weights),
            "als_params": als_params,
            "decay_rate": decay_rate,
            "course_scoping": course_scoping,
        }
        target = save_models(models_to_persist, context, metadata=metadata, out_dir=models_dir)
        print(f"Saved fitted models to {target}")

    print("\nDone.")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument(
        "--no-course-scoping",
        dest="course_scoping",
        action="store_false",
        help=(
            "Diagnostic: disable per-user course-scoped candidate restriction. "
            "Reproduces the pathological Popularity@K result documented in the "
            "prelim report. NOT the production configuration."
        ),
    )
    parser.add_argument(
        "--no-cv",
        dest="run_cv",
        action="store_false",
        help="Skip 5-fold temporal cross-validation (fast dev-loop mode).",
    )
    parser.add_argument(
        "--no-tuning",
        dest="run_tuning",
        action="store_false",
        help="Skip hyperparameter tuning; use default SVD/hybrid params.",
    )
    parser.add_argument(
        "--cold-start-threshold",
        type=int,
        default=10,
        help="Minimum train clicks for a user to count as 'warm' (default: 10).",
    )
    parser.add_argument(
        "--hybrid-grid-step",
        type=float,
        default=0.2,
        help="Simplex grid step for hybrid weights (default: 0.2 → 21 configs).",
    )
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=0.0,
        help=(
            "Exponential time-decay rate applied to training-window clicks: "
            "weight = clicks * exp(-decay * (cutoff - date)). Requires "
            "--rebuild-split to take effect."
        ),
    )
    parser.add_argument(
        "--persist-models",
        dest="persist_models",
        action="store_true",
        help=(
            "After evaluation, save fitted recommenders + serving context to "
            "data/processed/models/ so the FastAPI service can load them at startup."
        ),
    )
    parser.set_defaults(
        course_scoping=True, run_cv=True, run_tuning=True, persist_models=False
    )
    args = parser.parse_args()
    run(
        rebuild_split=args.rebuild_split,
        k_values=args.k,
        course_scoping=args.course_scoping,
        run_cv=args.run_cv,
        run_tuning=args.run_tuning,
        cold_start_threshold=args.cold_start_threshold,
        hybrid_grid_step=args.hybrid_grid_step,
        decay_rate=args.decay_rate,
        persist_models=args.persist_models,
    )


if __name__ == "__main__":
    main()
