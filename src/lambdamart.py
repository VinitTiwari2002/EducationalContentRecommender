"""LambdaMART two-stage reranker.

Stage 1 (retrieval): a pre-fit HybridRecommender picks the top-N candidates
from the course-scoped candidate pool for each user.

Stage 2 (rerank): a LightGBM LGBMRanker trained with the `lambdarank`
objective reorders those N candidates using a richer per-(user, item)
feature vector (see `src.reranker_features`). Only the top-K are returned.

Standard industry pattern (Burges et al., 2010; Bing/Amazon/YouTube), and
the highest-ceiling addition to the recommender roster per the design
chapter's future-work section. Introduced in the extended forward
workplan (post-Week 17) because the underlying persistence + FastAPI
plumbing needed to exist first.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from lightgbm import LGBMRanker
from scipy import sparse

from .features import ItemFeatures
from .hybrid import HybridRecommender
from .reranker_features import UserFeatures, build_pair_features


class LambdaMARTReranker:
    """Two-stage recommender wrapping a Stage-1 Hybrid + LambdaMART Stage-2.

    Fit contract mirrors the other recommenders (`fit(train, ...)`) for
    interchangeability in the evaluation harness — but the actual ranker
    training happens in `fit_ranker` because it needs per-user positive
    labels drawn from a *held-out* window.
    """

    def __init__(
        self,
        stage1: HybridRecommender,
        item_features: ItemFeatures,
        user_features: UserFeatures,
        top_n: int = 100,
        lgbm_params: dict | None = None,
        random_state: int = 0,
    ) -> None:
        if not isinstance(stage1, HybridRecommender):
            raise TypeError(
                "stage1 must be a HybridRecommender (its .cf / .content components "
                "are consumed by the feature builder). Peek the .warm_model of a "
                "GatedHybrid if you want to wrap a switching Stage 1."
            )
        self.stage1 = stage1
        self.item_features = item_features
        self.user_features = user_features
        self.top_n = int(top_n)
        self.random_state = int(random_state)
        # Sensible defaults: shallow trees, modest learning rate. Tuned on
        # OULAD in a single train/val split (see scripts/train_reranker.py).
        default_params = dict(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=random_state,
            verbose=-1,
        )
        if lgbm_params:
            default_params.update(lgbm_params)
        self.lgbm_params = default_params

        self._model: LGBMRanker | None = None
        self._train: sparse.csr_matrix | None = None
        self._n_items: int = 0

    # ------------------------------------------------------------------ #
    # Recommender-interface parity: fit stores the train matrix so the
    # standard `_recommend_all` helper can call `recommend(u, k, candidates)`.
    # ------------------------------------------------------------------ #

    def fit(
        self,
        train: sparse.csr_matrix,
        item_features: ItemFeatures | None = None,
        outcome_score: np.ndarray | None = None,
        refit_components: bool = True,
    ) -> None:
        """Store the train matrix and note Stage 1 is externally-fit.

        Signature mirrors HybridRecommender.fit so the evaluation harness
        can treat this class interchangeably. The actual reranker
        training is done via `fit_ranker`.
        """
        self._train = train
        self._n_items = train.shape[1]

    # ------------------------------------------------------------------ #
    # Reranker-specific training.
    # ------------------------------------------------------------------ #

    def fit_ranker(
        self,
        user_candidate_pools: dict[int, np.ndarray],
        positive_items_per_user: dict[int, Iterable[int]],
        max_users: int | None = None,
        verbose: bool = False,
    ) -> "LambdaMARTReranker":
        """Train the LGBMRanker on Stage-1-scored candidates.

        Parameters
        ----------
        user_candidate_pools: mapping user_row → the user's course-scoped
            candidate pool (item-column indices).
        positive_items_per_user: mapping user_row → list/set of item-column
            indices the user actually clicked in the *label* window
            (typically an earlier held-out sub-window than the primary test).
        max_users: optionally cap the training set for smoke tests.
        """
        if self._train is None:
            raise RuntimeError("fit() must be called before fit_ranker()")
        rows_X: list[np.ndarray] = []
        rows_y: list[np.ndarray] = []
        groups: list[int] = []
        n_users_trained = 0

        for user_row, cand_pool in user_candidate_pools.items():
            if max_users is not None and n_users_trained >= max_users:
                break
            top_n_cols = self.stage1.recommend(
                user_row=user_row,
                k=self.top_n,
                candidates=np.asarray(cand_pool),
                exclude_seen=True,
            )
            if not top_n_cols:
                continue
            top_n_arr = np.asarray(top_n_cols, dtype=np.int64)
            positive = set(int(x) for x in positive_items_per_user.get(user_row, []))
            labels = np.array(
                [1 if int(c) in positive else 0 for c in top_n_arr], dtype=np.int32
            )
            if int(labels.sum()) == 0:
                continue  # no positives in Stage-1's top-N → nothing to rank
            features = build_pair_features(
                user_row=user_row,
                candidate_cols=top_n_arr,
                hybrid=self.stage1,
                item_features=self.item_features,
                user_features=self.user_features,
            )
            rows_X.append(features)
            rows_y.append(labels)
            groups.append(int(top_n_arr.shape[0]))
            n_users_trained += 1
            if verbose and n_users_trained % 500 == 0:
                print(f"    {n_users_trained} training users prepared")

        if not rows_X:
            raise RuntimeError(
                "No training rows constructed — every user's Stage-1 top-N "
                "missed the positive labels. Check the label window."
            )
        X = np.vstack(rows_X).astype(np.float32)
        y = np.concatenate(rows_y).astype(np.int32)
        group = np.array(groups, dtype=np.int32)
        if verbose:
            print(f"  fitting LGBMRanker on {X.shape[0]:,} rows × "
                  f"{X.shape[1]} features across {group.size} query groups...")

        self._model = LGBMRanker(**self.lgbm_params)
        self._model.fit(X, y, group=group)
        return self

    # ------------------------------------------------------------------ #
    # Inference — Stage 1 retrieves, Stage 2 rescores + reorders.
    # ------------------------------------------------------------------ #

    def recommend(
        self,
        user_row: int,
        k: int,
        candidates: np.ndarray | None = None,
        exclude_seen: bool = True,
    ) -> list[int]:
        if self._train is None:
            raise RuntimeError("fit() must be called before recommend()")
        if self._model is None:
            raise RuntimeError("fit_ranker() must be called before recommend()")
        top_n_cols = self.stage1.recommend(
            user_row=user_row,
            k=self.top_n,
            candidates=candidates,
            exclude_seen=exclude_seen,
        )
        if not top_n_cols:
            return []
        top_n_arr = np.asarray(top_n_cols, dtype=np.int64)
        features = build_pair_features(
            user_row=user_row,
            candidate_cols=top_n_arr,
            hybrid=self.stage1,
            item_features=self.item_features,
            user_features=self.user_features,
        )
        scores = self._model.predict(features)
        order = np.argsort(-scores, kind="stable")
        top_k = top_n_arr[order][:k]
        return top_k.tolist()

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Return LightGBM's gain-based feature importance as a name→gain dict."""
        if self._model is None:
            raise RuntimeError("fit_ranker() must be called first")
        booster = self._model.booster_
        gains = booster.feature_importance(importance_type="gain")
        return dict(zip(feature_names, gains.astype(float).tolist()))
