"""Random and Popularity baseline recommenders.

Both implement the same minimal interface:
    fit(train: csr_matrix) -> None
    recommend(user_row: int, k: int, candidates: np.ndarray | None,
              exclude_seen: bool = True) -> list[int]

`candidates`, if provided, restricts the recommendation pool to that array of
column indices. This is how course-scoping is enforced: the harness passes in
the user's enrolled-presentation item columns. If candidates is None, the
recommender considers all items (used in unit tests with toy data).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


class RandomRecommender:
    """Recommends K items sampled uniformly from the candidate pool.

    Deterministic given a seed so results are reproducible across runs.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._n_items = 0
        self._train: sparse.csr_matrix | None = None

    def fit(self, train: sparse.csr_matrix) -> None:
        self._n_items = train.shape[1]
        self._train = train

    def recommend(
        self,
        user_row: int,
        k: int,
        candidates: np.ndarray | None = None,
        exclude_seen: bool = True,
    ) -> list[int]:
        if self._train is None:
            raise RuntimeError("fit() must be called before recommend()")
        if candidates is None:
            candidate_pool = np.arange(self._n_items)
        else:
            candidate_pool = candidates
        if exclude_seen:
            seen = self._train[user_row].indices
            if len(seen) > 0:
                candidate_pool = np.setdiff1d(candidate_pool, seen, assume_unique=False)
        if len(candidate_pool) == 0:
            return []
        size = min(k, len(candidate_pool))
        return self._rng.choice(candidate_pool, size=size, replace=False).tolist()


class PopularityRecommender:
    """Recommends the top items by training-set click totals, restricted to
    the candidate pool. A separate ranking is computed for each
    recommend() call by filtering the precomputed global totals to the
    candidate columns and sorting."""

    def __init__(self) -> None:
        self._item_totals: np.ndarray | None = None
        self._train: sparse.csr_matrix | None = None

    def fit(self, train: sparse.csr_matrix) -> None:
        self._item_totals = np.asarray(train.sum(axis=0)).ravel()
        self._train = train

    def recommend(
        self,
        user_row: int,
        k: int,
        candidates: np.ndarray | None = None,
        exclude_seen: bool = True,
    ) -> list[int]:
        if self._item_totals is None or self._train is None:
            raise RuntimeError("fit() must be called before recommend()")
        if candidates is None:
            candidate_pool = np.arange(len(self._item_totals))
        else:
            candidate_pool = candidates
        if exclude_seen:
            seen = self._train[user_row].indices
            if len(seen) > 0:
                candidate_pool = np.setdiff1d(candidate_pool, seen, assume_unique=False)
        if len(candidate_pool) == 0:
            return []
        scores = self._item_totals[candidate_pool]
        # Sort by score descending; stable to make ties reproducible
        order = np.argsort(-scores, kind="stable")
        top = candidate_pool[order][:k]
        return top.tolist()
