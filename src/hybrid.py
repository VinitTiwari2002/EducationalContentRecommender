"""Hybrid recommender: weighted ensemble of CF + content + outcome.

Combines three scoring signals per (user, candidate item) pair:

    s(u, i) = alpha * s_CF(u, i) + beta * s_content(u, i) + gamma * s_outcome(i)

Each component is min-max normalised across the candidate pool before
combination, so weights can be interpreted as relative importances rather
than absorbing each signal's raw scale. This mirrors the "min-max
normalise then linear combine" recipe used in the Netflix Prize hybrid
literature (Koren, Bell & Volinsky, 2009).

Setting any weight to zero disables that component — this is what powers
the ablation study reported in the evaluation chapter.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .content import ContentRecommender
from .svd import SVDRecommender


class HybridRecommender:
    """Weighted-sum hybrid over CF (SVD), content, and outcome signals.

    The outcome signal is a per-item constant (mean assessment score of
    training-set users who accessed the item) — it does not depend on the
    user. Its role is to bias recommendations towards items that were
    empirically associated with better outcomes in the training window.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        cf=None,
        content: ContentRecommender | None = None,
        svd=None,
    ) -> None:
        """cf: any recommender with a `.score(user_row) -> np.ndarray` and
        `.fit(train)` interface (SVDRecommender, ALSRecommender, …). The
        `svd=` keyword is a back-compat alias for `cf=`.
        """
        if alpha < 0 or beta < 0 or gamma < 0:
            raise ValueError("hybrid weights must be non-negative")
        if alpha + beta + gamma == 0:
            raise ValueError("at least one hybrid weight must be positive")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        cf_model = cf if cf is not None else svd
        self.cf = cf_model if cf_model is not None else SVDRecommender()
        self.content = content if content is not None else ContentRecommender()
        self._train: sparse.csr_matrix | None = None
        self._outcome_score: np.ndarray | None = None
        self._n_items: int = 0

    # Back-compat: some earlier tests / call-sites read `hybrid.svd`.
    @property
    def svd(self):
        return self.cf

    def fit(
        self,
        train: sparse.csr_matrix,
        item_features: np.ndarray,
        outcome_score: np.ndarray,
        refit_components: bool = True,
    ) -> None:
        """Fit the hybrid. Set refit_components=False when self.svd and
        self.content have already been fitted on the same train matrix —
        used by the tuning loop to iterate over hybrid weights without
        re-fitting SVD (which is the expensive step)."""
        if outcome_score.shape[0] != train.shape[1]:
            raise ValueError(
                f"outcome_score has {outcome_score.shape[0]} entries but "
                f"train has {train.shape[1]} items"
            )
        self._train = train
        self._n_items = train.shape[1]
        self._outcome_score = outcome_score.astype(np.float32)
        if refit_components:
            self.cf.fit(train)
            self.content.fit(train, item_features)

    def set_weights(self, alpha: float, beta: float, gamma: float) -> None:
        """Update the hybrid weights without touching the fitted components."""
        if alpha < 0 or beta < 0 or gamma < 0:
            raise ValueError("hybrid weights must be non-negative")
        if alpha + beta + gamma == 0:
            raise ValueError("at least one hybrid weight must be positive")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def score(self, user_row: int) -> np.ndarray:
        """Blended score across the entire catalogue for a single user."""
        if self._outcome_score is None:
            raise RuntimeError("fit() must be called before score()")
        cf = self.cf.score(user_row)
        cn = self.content.score(user_row)
        oc = self._outcome_score
        return (
            self.alpha * _minmax(cf)
            + self.beta * _minmax(cn)
            + self.gamma * _minmax(oc)
        )

    def score_breakdown(
        self,
        user_row: int,
        items: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Return the per-item score decomposition for a user.

        For each item, exposes the raw component scores (unnormalised), the
        min-max-normalised component scores over `items`, each component
        weighted by its hybrid weight, and the summed total. The dashboard
        audit view consumes this to explain *why* an item was ranked where
        it was; the /decompose FastAPI endpoint passes a single-element
        `items` array to get a per-pair decomposition.

        Parameters
        ----------
        user_row: row index in the interaction matrix.
        items: item-column indices to explain (default: full catalogue).

        Returns
        -------
        dict with keys `items`, `cf_raw`, `content_raw`, `outcome_raw`,
        `cf_norm`, `content_norm`, `outcome_norm`, `cf_weighted`,
        `content_weighted`, `outcome_weighted`, `total`. All arrays are
        aligned with `items` in the returned order.
        """
        if self._outcome_score is None:
            raise RuntimeError("fit() must be called before score_breakdown()")
        items_arr = (
            np.arange(self._n_items) if items is None else np.asarray(items).ravel()
        )
        cf_raw = self.cf.score(user_row)[items_arr]
        content_raw = self.content.score(user_row)[items_arr]
        outcome_raw = self._outcome_score[items_arr]
        cf_norm = _minmax(cf_raw)
        content_norm = _minmax(content_raw)
        outcome_norm = _minmax(outcome_raw)
        cf_weighted = self.alpha * cf_norm
        content_weighted = self.beta * content_norm
        outcome_weighted = self.gamma * outcome_norm
        total = cf_weighted + content_weighted + outcome_weighted
        return {
            "items": items_arr,
            "cf_raw": cf_raw,
            "content_raw": content_raw,
            "outcome_raw": outcome_raw,
            "cf_norm": cf_norm,
            "content_norm": content_norm,
            "outcome_norm": outcome_norm,
            "cf_weighted": cf_weighted,
            "content_weighted": content_weighted,
            "outcome_weighted": outcome_weighted,
            "total": total,
        }

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

        # Normalise each component *within the candidate pool* so weights
        # remain interpretable regardless of course-scoping.
        cf = _minmax(self.cf.score(user_row)[candidate_pool])
        cn = _minmax(self.content.score(user_row)[candidate_pool])
        oc = _minmax(self._outcome_score[candidate_pool])
        scores = self.alpha * cf + self.beta * cn + self.gamma * oc

        order = np.argsort(-scores, kind="stable")
        top = candidate_pool[order][:k]
        return top.tolist()


def _minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalise into [0, 1]; zero-range arrays become zero."""
    if x.size == 0:
        return x
    lo = float(x.min())
    hi = float(x.max())
    if hi == lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)
