"""Alternating Least Squares (ALS) collaborative-filtering recommender.

Implements the implicit-feedback ALS method of Hu, Koren & Volinsky (2008),
via the `implicit` library. Unlike truncated SVD, which factorises the
click matrix directly and treats zeros as strong negative signal, ALS-with-
confidence-weighting distinguishes "observed with confidence c(u, i)" from
"no observation" — a much better fit for click data because absence of
interaction is not the same as active dislike.

The confidence weighting is c(u, i) = 1 + alpha * clicks(u, i); the model
minimises

    sum_{u, i} c(u, i) * (p(u, i) - x_u^T y_i)^2 + lambda * (||x_u||^2 + ||y_i||^2)

where p(u, i) = 1 if clicks(u, i) > 0 else 0. This is the standard IF
recommendation objective and is the canonical baseline the design chapter
should have named. We add it here and treat it as the CF backbone of the
hybrid, keeping the plain-SVD implementation for comparison in the
evaluation chapter.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse


class ALSRecommender:
    """Wrap implicit.als.AlternatingLeastSquares behind the project's
    Recommender interface (fit + recommend + score).

    Parameters
    ----------
    n_factors: rank of the factorisation.
    regularization: lambda in the objective.
    iterations: number of ALS sweeps.
    alpha: confidence-weight scale c(u,i) = 1 + alpha * clicks(u,i).
    random_state: seed for reproducibility across runs.
    """

    def __init__(
        self,
        n_factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        alpha: float = 1.0,
        random_state: int = 0,
    ) -> None:
        if n_factors <= 0:
            raise ValueError("n_factors must be positive")
        self.n_factors = int(n_factors)
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self._train: sparse.csr_matrix | None = None
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._n_items: int = 0

    def fit(self, train: sparse.csr_matrix) -> None:
        # Local import so the base package still imports cleanly if the
        # optional dep is missing; users only pay for implicit when they
        # actually instantiate an ALSRecommender.
        from implicit.als import AlternatingLeastSquares

        self._train = train
        self._n_items = train.shape[1]

        confidence = (train * self.alpha).astype(np.float32)
        with warnings.catch_warnings():
            # implicit prints a "OpenBLAS threads" warning we don't need
            warnings.simplefilter("ignore")
            model = AlternatingLeastSquares(
                factors=self.n_factors,
                regularization=self.regularization,
                iterations=self.iterations,
                random_state=self.random_state,
                use_gpu=False,
            )
            model.fit(confidence, show_progress=False)

        self._user_factors = np.asarray(model.user_factors, dtype=np.float32)
        self._item_factors = np.asarray(model.item_factors, dtype=np.float32)

    def score(self, user_row: int) -> np.ndarray:
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
