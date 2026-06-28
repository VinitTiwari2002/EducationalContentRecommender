"""Ranking metrics for top-K recommendation evaluation.

All metrics take a list of recommended item indices and a set of relevant
(held-out) item indices and return a scalar in [0, 1]. The harness averages
per-user scores to produce final numbers.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def precision_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not recommended:
        return 0.0
    top_k = list(recommended)[:k]
    relevant_set = set(relevant)
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = list(recommended)[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant_set)


def hit_rate_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = list(recommended)[:k]
    return 1.0 if any(item in relevant_set for item in top_k) else 0.0


def ndcg_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Binary-relevance NDCG@K. Each relevant hit at rank r contributes
    1/log2(r+2); ideal DCG is the sum over min(k, |relevant|) top ranks."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = list(recommended)[:k]
    dcg = 0.0
    for rank, item in enumerate(top_k):
        if item in relevant_set:
            dcg += 1.0 / np.log2(rank + 2)
    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / np.log2(r + 2) for r in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    recommendations: dict[int, Sequence[int]],
    relevant_per_user: dict[int, Iterable[int]],
    k: int = 10,
) -> dict[str, float]:
    """Aggregate per-user metrics. Users with no relevant items are skipped."""
    users = [u for u, items in relevant_per_user.items() if items]
    if not users:
        return {
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            f"hit_rate@{k}": 0.0,
            "n_users": 0,
        }

    precisions, recalls, ndcgs, hits = [], [], [], []
    for user in users:
        recs = recommendations.get(user, [])
        rel = relevant_per_user[user]
        precisions.append(precision_at_k(recs, rel, k))
        recalls.append(recall_at_k(recs, rel, k))
        ndcgs.append(ndcg_at_k(recs, rel, k))
        hits.append(hit_rate_at_k(recs, rel, k))

    return {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"hit_rate@{k}": float(np.mean(hits)),
        "n_users": len(users),
    }
