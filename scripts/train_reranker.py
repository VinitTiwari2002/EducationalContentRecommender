"""Train the LambdaMART reranker + evaluate against Stage-1 GatedHybrid.

Setup (honest three-window design):
    * Fit Stage-1 (a fresh Hybrid with the tuned weights) on tune_train
      — the earlier slice of the primary training window.
    * Labels for reranker training: items each user clicked in tune_val
      — the later slice of the primary training window. This is the
      held-out feedback signal that teaches LambdaMART "of Stage-1's
      top-N candidates, which ones actually got clicked next?".
    * At inference (evaluation on split.test), swap in the persisted
      Stage-1 that was fit on the full primary train window. The
      trained LGBMRanker model transfers unchanged — its features are
      min-max-normalised within each user's candidate pool, so the
      Stage-1 window doesn't affect feature scale.

Usage:
    .venv/bin/python scripts/train_reranker.py

Writes:
    data/processed/models/reranker.joblib   — the trained LGBMRanker
    evaluation/reranker_results.csv        — headline metrics vs Stage 1
    evaluation/reranker_importance.csv     — LightGBM gain per feature
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import oulad  # noqa: E402
from src.als import ALSRecommender  # noqa: E402
from src.content import ContentRecommender  # noqa: E402
from src.features import build_item_features  # noqa: E402
from src.hybrid import HybridRecommender  # noqa: E402
from src.lambdamart import LambdaMARTReranker  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.persistence import DEFAULT_MODELS_DIR, load_models  # noqa: E402
from src.preprocess import Split  # noqa: E402
from src.reranker_features import UserFeatures, build_user_features, pair_feature_names  # noqa: E402
from src.tuning import build_tuning_split  # noqa: E402


# Tuned Stage-1 hyperparameters (match Table 5.1 in the evaluation chapter).
HYBRID_WEIGHTS = (0.0, 0.8, 0.2)
ALS_N_FACTORS = 32
ALS_REGULARIZATION = 0.01
ALS_ALPHA = 1.0
ALS_ITERATIONS = 15
TOP_N = 100
K_PRIMARY = 10


def _make_hybrid() -> HybridRecommender:
    alpha, beta, gamma = HYBRID_WEIGHTS
    return HybridRecommender(
        alpha=alpha, beta=beta, gamma=gamma,
        cf=ALSRecommender(
            n_factors=ALS_N_FACTORS,
            regularization=ALS_REGULARIZATION,
            iterations=ALS_ITERATIONS,
            alpha=ALS_ALPHA,
            random_state=0,
        ),
        content=ContentRecommender(),
    )


def _relevant_per_user(mat):
    out = {}
    for r in range(mat.shape[0]):
        items = mat[r].indices.tolist()
        if items:
            out[r] = items
    return out


def _candidate_pools_dict(user_candidates_list, users):
    return {u: user_candidates_list[u] for u in users}


def _recommend_all(model, n_users, k, candidates_by_user):
    return {
        u: model.recommend(u, k=k, candidates=candidates_by_user[u])
        for u in range(n_users)
    }


def main() -> None:
    print("=== LambdaMART reranker training ===\n")

    print("Loading OULAD + cached primary split + persisted Stage-1...")
    data = oulad.load()
    split = Split.load(PROJECT_ROOT / "data" / "processed")
    persisted_models, context, manifest = load_models(DEFAULT_MODELS_DIR)
    hybrid_full = persisted_models["Hybrid"]
    print(f"  primary split: cutoff={split.cutoff_date}, "
          f"train_nnz={split.train.nnz:,}, test_nnz={split.test.nnz:,}")

    # Item features on the FULL primary train (matches persisted Stage-1).
    item_features_full = build_item_features(
        data, split.train, split.item_index, split.student_index
    )
    # User features are Stage-1-window-independent: they use studentInfo
    # + train counts + past assessments. Compute them once for the full
    # split; the same values are used at both training and inference.
    user_features = build_user_features(data, split.train, split.student_index)
    print(f"  item features: {item_features_full.matrix.shape}; "
          f"user features: {user_features.matrix.shape}")

    # ----------------------------------------------------------------- #
    # 1) Build the tuning sub-split for held-out reranker training.
    # ----------------------------------------------------------------- #
    print("\nBuilding tuning sub-split for reranker labels...")
    tune = build_tuning_split(
        data, split.student_index, split.item_index,
        primary_cutoff=split.cutoff_date,
    )
    print(f"  tune_train nnz={tune.tune_train.nnz:,}, "
          f"tune_val nnz={tune.tune_val.nnz:,}, cutoff={tune.tune_cutoff}")

    # Fresh Stage-1 fit on tune_train (avoids leakage of tune_val info
    # into feature scores).
    item_features_tune = build_item_features(
        data, tune.tune_train, tune.item_index, tune.student_index
    )
    hybrid_tune = _make_hybrid()
    print("  fitting Stage-1 Hybrid on tune_train...")
    hybrid_tune.fit(
        tune.tune_train,
        item_features_tune.matrix,
        item_features_tune.outcome_score,
    )

    # ----------------------------------------------------------------- #
    # 2) Train the LambdaMART reranker.
    # ----------------------------------------------------------------- #
    print("\nTraining LambdaMART reranker on Stage-1 top-N candidates...")
    reranker_train = LambdaMARTReranker(
        stage1=hybrid_tune,
        item_features=item_features_tune,
        user_features=user_features,
        top_n=TOP_N,
        random_state=0,
    )
    reranker_train.fit(tune.tune_train)

    # Positive labels come from tune_val (chronologically later than tune_train).
    positives = _relevant_per_user(tune.tune_val)
    # Training pool: same course-scoped pool as inference — the persisted
    # user_candidates already restricts to enrolled presentations.
    train_pools = {
        int(u): np.asarray(split.user_candidates[int(u)], dtype=np.int64)
        for u in positives
    }
    reranker_train.fit_ranker(
        user_candidate_pools=train_pools,
        positive_items_per_user=positives,
        verbose=True,
    )

    # ----------------------------------------------------------------- #
    # 3) Transfer the trained LGBMRanker to the inference reranker (Stage-1
    #    now points at the persisted full-train Hybrid + full-train features).
    # ----------------------------------------------------------------- #
    print("\nAssembling inference reranker with persisted full-train Stage-1...")
    reranker = LambdaMARTReranker(
        stage1=hybrid_full,
        item_features=item_features_full,
        user_features=user_features,
        top_n=TOP_N,
        random_state=0,
    )
    reranker.fit(split.train)
    reranker._model = reranker_train._model  # transfer trained booster
    print(f"  reranker ready; TOP_N={TOP_N}")

    # ----------------------------------------------------------------- #
    # 4) Evaluate on the primary test split.
    # ----------------------------------------------------------------- #
    print("\n=== Evaluating on primary split ===")
    relevant = _relevant_per_user(split.test)
    n_users = split.train.shape[0]
    n_items = split.train.shape[1]
    pools_all = {u: np.asarray(split.user_candidates[u], dtype=np.int64) for u in range(n_users)}

    print("  Stage-1 (GatedHybrid persisted) — baseline...")
    gated = persisted_models["GatedHybrid"]
    stage1_recs = _recommend_all(gated, n_users, K_PRIMARY, pools_all)
    stage1_metrics = evaluate(
        stage1_recs, relevant, k=K_PRIMARY,
        outcome_score=item_features_full.outcome_score, n_items=n_items,
    )

    print("  LambdaMART reranker...")
    reranker_recs = _recommend_all(reranker, n_users, K_PRIMARY, pools_all)
    reranker_metrics = evaluate(
        reranker_recs, relevant, k=K_PRIMARY,
        outcome_score=item_features_full.outcome_score, n_items=n_items,
    )

    print("\n=== Results ===")
    rows = [
        {"model": "GatedHybrid (Stage 1)", **stage1_metrics},
        {"model": "LambdaMART (rerank)", **reranker_metrics},
    ]
    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "evaluation" / "reranker_results.csv"
    df.to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    print(f"\nSaved to {out}")

    # ----------------------------------------------------------------- #
    # 5) Feature importance + persistence.
    # ----------------------------------------------------------------- #
    names = pair_feature_names(item_features_full, user_features)
    fi = reranker.feature_importance(names)
    fi_df = pd.DataFrame(
        sorted(fi.items(), key=lambda kv: -kv[1]),
        columns=["feature", "gain"],
    )
    fi_out = PROJECT_ROOT / "evaluation" / "reranker_importance.csv"
    fi_df.to_csv(fi_out, index=False)
    print(f"\nTop-10 features by LightGBM gain:")
    print(fi_df.head(10).to_string(index=False))
    print(f"\nSaved feature importance to {fi_out}")

    reranker_out = DEFAULT_MODELS_DIR / "reranker.joblib"
    joblib.dump(reranker._model, reranker_out, compress=3)
    print(f"Saved LGBMRanker booster to {reranker_out}")


if __name__ == "__main__":
    main()
