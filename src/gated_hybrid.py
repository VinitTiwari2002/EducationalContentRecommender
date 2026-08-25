"""Per-user gated hybrid recommender.

Motivation. Our cold-start analysis surfaced that Popularity beats the
tuned Hybrid on users with fewer than ~10 training clicks, because those
users have no history to build a content profile from. The tuned Hybrid
wins decisively on warm users. A single set of weights cannot exploit
both — so instead of fighting the ablation result, we gate: cold users
are routed to Popularity (or any other cold-safe fallback), warm users
are routed to the tuned Hybrid.

This is a form of the "meta-recommender" or "switching" hybrid pattern
described in Burke's classification of hybrids (Burke 2002); it turns a
weakness that surfaces in the ablation into a designed feature of the
system.

The gate is threshold-based on training clicks (n_clicks < threshold →
cold branch, else warm branch). A future extension could learn the gate
from user demographic features, but the threshold form is directly
testable against the cold-start CSV and easy to reason about.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .hybrid import HybridRecommender


class GatedHybridRecommender:
    """Routes each user to a cold or warm sub-recommender.

    Parameters
    ----------
    cold_model: fallback for users with <threshold training clicks. Any
        recommender exposing fit(train) and recommend(user_row, k, ...).
        Popularity is the natural choice.
    warm_model: recommender for users with >=threshold clicks. A tuned
        HybridRecommender is the intended choice; anything with the
        3-arg fit(train, item_features, outcome_score) signature works.
    threshold: n_clicks cut-off. Default 10 matches the cold-start CSV.
    """

    def __init__(
        self,
        cold_model,
        warm_model,
        threshold: int = 10,
    ) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.cold_model = cold_model
        self.warm_model = warm_model
        self.threshold = int(threshold)
        self._train: sparse.csr_matrix | None = None
        self._n_clicks: np.ndarray | None = None

    def fit(
        self,
        train: sparse.csr_matrix,
        item_features: np.ndarray,
        outcome_score: np.ndarray,
    ) -> None:
        self._train = train
        self._n_clicks = np.asarray((train != 0).sum(axis=1)).ravel()
        # Cold model has the simple fit(train) signature.
        self.cold_model.fit(train)
        # Warm model has the 3-arg hybrid signature.
        self.warm_model.fit(train, item_features, outcome_score)

    def is_cold(self, user_row: int) -> bool:
        if self._n_clicks is None:
            raise RuntimeError("fit() must be called before is_cold()")
        return int(self._n_clicks[user_row]) < self.threshold

    def recommend(
        self,
        user_row: int,
        k: int,
        candidates: np.ndarray | None = None,
        exclude_seen: bool = True,
    ) -> list[int]:
        if self._train is None:
            raise RuntimeError("fit() must be called before recommend()")
        model = self.cold_model if self.is_cold(user_row) else self.warm_model
        return model.recommend(user_row, k, candidates=candidates, exclude_seen=exclude_seen)
