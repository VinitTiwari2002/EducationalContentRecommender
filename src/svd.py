"""Truncated-SVD collaborative filtering recommender.

Applies scipy.sparse.linalg.svds to a log1p-transformed click matrix,
producing user and item factors whose dot product approximates the log
click intensity. The log1p transform tames the power-law distribution of
click counts (a handful of items dominate raw clicks; log-scaling makes the
factorisation see the long tail).

Design deviation from the preliminary report: the design chapter said
"centred" interaction matrix. In practice, mean-subtracting a sparse click
matrix destroys sparsity and inflates memory to O(users × items); the
standard implicit-feedback recipe (Hu, Koren & Volinsky 2008; Koren, Bell
& Volinsky 2009) is log-weighting without centring, which we follow. This
deviation is documented in the implementation chapter of the draft report.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


class SVDRecommender:
    """Truncated SVD on log1p(train).

    Parameters
    ----------
    n_factors: rank of the truncation, k. Design chapter tunes k in
        {20, 50, 100}; default is 50.
    random_state: seed used to initialise ARPACK's Lanczos iteration
        (via the v0 argument to svds), for reproducibility.
    """

    def __init__(self, n_factors: int = 50, random_state: int = 0) -> None:
        if n_factors <= 0:
            raise ValueError("n_factors must be positive")
        self.n_factors = int(n_factors)
        self.random_state = int(random_state)
        self._train: sparse.csr_matrix | None = None
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._n_items: int = 0

    def fit(self, train: sparse.csr_matrix) -> None:
        self._train = train
        self._n_items = train.shape[1]
        m = train.astype(np.float32)
        m.data = np.log1p(m.data)

        max_k = min(m.shape) - 1
        k = min(self.n_factors, max_k)
        if k <= 0:
            # Degenerate tiny matrix: score everything as zero.
            self._user_factors = np.zeros((m.shape[0], 1), dtype=np.float32)
            self._item_factors = np.zeros((m.shape[1], 1), dtype=np.float32)
            return

        rng = np.random.default_rng(self.random_state)
        v0 = rng.standard_normal(min(m.shape)).astype(np.float32)
        u, s, vt = spla.svds(m, k=k, v0=v0)

        # svds returns singular values in ascending order; flip to descending
        # so factor 0 is the most-important direction (cosmetic — dot product
        # is invariant to column reordering as long as u and vt agree).
        order = np.argsort(-s)
        u = u[:, order]
        s = s[order]
        vt = vt[order]
        self._user_factors = (u * s).astype(np.float32)
        self._item_factors = vt.T.astype(np.float32)

    def score(self, user_row: int) -> np.ndarray:
        """Reconstructed score for every item, for a single user."""
        if self._user_factors is None or self._item_factors is None:
            raise RuntimeError("fit() must be called before score()")
        return self._item_factors @ self._user_factors[user_row]

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
