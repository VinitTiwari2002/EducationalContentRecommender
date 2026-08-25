"""Content-based recommender.

Ranks items by cosine similarity between (a) the item's feature vector and
(b) the user's profile vector, where the user profile is the click-weighted
mean of the feature vectors of items the user has accessed in training.

Design rationale (from the design chapter, following Bousbahi & Chorfi,
2015): SVD collapses under interaction sparsity for cold-start users, but
content-based filtering degrades gracefully because it only requires a
single accessed item to build a profile.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


class ContentRecommender:
    """Cosine-similarity content-based recommender.

    Fit consumes both the training interaction matrix and a dense item
    feature matrix (see `src.features.build_item_features`). At recommend
    time, the user's profile is the click-weighted mean of the L2-normalised
    feature vectors of items the user has clicked in training; cosine
    similarity reduces to a dot product against the L2-normalised item
    features.
    """

    def __init__(self) -> None:
        self._train: sparse.csr_matrix | None = None
        self._item_features_norm: np.ndarray | None = None
        self._user_profiles: np.ndarray | None = None
        self._n_items: int = 0

    def fit(self, train: sparse.csr_matrix, item_features: np.ndarray) -> None:
        if train.shape[1] != item_features.shape[0]:
            raise ValueError(
                f"train has {train.shape[1]} items but item_features has "
                f"{item_features.shape[0]} rows"
            )
        self._train = train
        self._n_items = train.shape[1]

        norms = np.linalg.norm(item_features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._item_features_norm = (item_features / norms).astype(np.float32)

        # user_profile[u] = normalise( sum_i(click_ui) * feature_norm[i] ) / sum(click_ui)
        # Achieved with a sparse matmul: train @ feature_norm, then row-normalise.
        raw_profiles = np.asarray(train @ self._item_features_norm)
        profile_norms = np.linalg.norm(raw_profiles, axis=1, keepdims=True)
        profile_norms[profile_norms == 0] = 1.0
        self._user_profiles = (raw_profiles / profile_norms).astype(np.float32)

    def score(self, user_row: int) -> np.ndarray:
        """Cosine-similarity score for every item, for a single user."""
        if self._user_profiles is None or self._item_features_norm is None:
            raise RuntimeError("fit() must be called before score()")
        return self._item_features_norm @ self._user_profiles[user_row]

    def recommend(
        self,
        user_row: int,
        k: int,
        candidates: np.ndarray | None = None,
        exclude_seen: bool = True,
    ) -> list[int]:
        if self._train is None:
            raise RuntimeError("fit() must be called before recommend()")
        candidate_pool = (
            np.arange(self._n_items) if candidates is None else np.asarray(candidates)
        )
        if exclude_seen:
            seen = self._train[user_row].indices
            if len(seen) > 0:
                candidate_pool = np.setdiff1d(candidate_pool, seen, assume_unique=False)
        if len(candidate_pool) == 0:
            return []
        scores = self.score(user_row)[candidate_pool]
        order = np.argsort(-scores, kind="stable")
        top = candidate_pool[order][:k]
        return top.tolist()
