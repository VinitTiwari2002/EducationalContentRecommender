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


def outcome_weighted_precision_at_k(
    recommended: Sequence[int],
    relevant: Iterable[int],
    outcome_score: np.ndarray,
    k: int,
    baseline: float | None = None,
) -> float:
    """Outcome-weighted precision@K.

    Formal definition. Let R_u = top-k recommendation list for user u,
    H_u = held-out relevant items, o(i) in [0, 100] = mean assessment score
    of training-set users who accessed item i, and \bar o = train-only global
    mean of o(). Then:

        OWP@K(u) = (1/k) * sum_{i in R_u} 1{i in H_u} * (o(i) / \bar o)

    Interpretation: a plain precision@k hit is worth exactly 1; a hit on an
    item with above-average outcomes contributes >1, and a hit on an item
    with below-average outcomes contributes <1. Averaged across users, OWP
    is a scalar in [0, ~100/\bar o] whose deviation from precision@k
    isolates *whether the recommender is preferentially recommending items
    with better learning outcomes*.

    All the outcome quantities (o and \bar o) are computed strictly from
    training data. This is what makes the metric an offline proxy for
    learning benefit rather than a leaked signal from the test window.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not recommended:
        return 0.0
    top_k = list(recommended)[:k]
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    if baseline is None:
        baseline = float(np.mean(outcome_score)) if outcome_score.size > 0 else 1.0
    if baseline <= 0:
        baseline = 1.0
    total = 0.0
    for item in top_k:
        if item in relevant_set and 0 <= item < outcome_score.size:
            total += float(outcome_score[item]) / baseline
    return total / k


def catalogue_coverage(
    recommendations: dict[int, Sequence[int]], n_items: int
) -> float:
    """Fraction of the catalogue that appears at least once in any
    recommendation list. High coverage = recommender explores the long
    tail; low coverage = recommender fixates on a small popular set."""
    if n_items <= 0:
        return 0.0
    seen: set[int] = set()
    for recs in recommendations.values():
        seen.update(recs)
    return len(seen) / n_items


def recommendation_gini(recommendations: dict[int, Sequence[int]]) -> float:
    """Gini index of the *frequency distribution over items* across all
    recommendations. 0 = perfectly uniform; 1 = one item recommended
    everywhere. Complements catalogue_coverage: high coverage with high
    Gini means many items appear but a few dominate."""
    counts: dict[int, int] = {}
    for recs in recommendations.values():
        for item in recs:
            counts[item] = counts.get(item, 0) + 1
    if not counts:
        return 0.0
    freq = np.array(sorted(counts.values()), dtype=np.float64)
    n = freq.size
    total = freq.sum()
    if total == 0:
        return 0.0
    # Gini via the sorted-frequencies formula: (2 * sum(i*x_i) - (n+1)*sum(x)) / (n * sum(x))
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * freq) - (n + 1) * total) / (n * total))


def evaluate(
    recommendations: dict[int, Sequence[int]],
    relevant_per_user: dict[int, Iterable[int]],
    k: int = 10,
    outcome_score: np.ndarray | None = None,
    n_items: int | None = None,
) -> dict[str, float]:
    """Aggregate per-user metrics. Users with no relevant items are skipped.

    Optional outcome_score (n_items,) unlocks outcome_weighted_precision;
    optional n_items unlocks catalogue_coverage. Recommendation-Gini is
    always computed since it needs only the recommendation dict.
    """
    users = [u for u, items in relevant_per_user.items() if items]
    if not users:
        empty = {
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            f"hit_rate@{k}": 0.0,
            "n_users": 0,
        }
        if outcome_score is not None:
            empty[f"outcome_weighted_precision@{k}"] = 0.0
        if n_items is not None:
            empty["catalogue_coverage"] = 0.0
        empty["recommendation_gini"] = 0.0
        return empty

    precisions, recalls, ndcgs, hits, owp = [], [], [], [], []
    baseline = (
        float(np.mean(outcome_score))
        if outcome_score is not None and outcome_score.size > 0
        else None
    )
    for user in users:
        recs = recommendations.get(user, [])
        rel = relevant_per_user[user]
        precisions.append(precision_at_k(recs, rel, k))
        recalls.append(recall_at_k(recs, rel, k))
        ndcgs.append(ndcg_at_k(recs, rel, k))
        hits.append(hit_rate_at_k(recs, rel, k))
        if outcome_score is not None:
            owp.append(
                outcome_weighted_precision_at_k(
                    recs, rel, outcome_score, k, baseline=baseline
                )
            )

    out: dict[str, float] = {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"hit_rate@{k}": float(np.mean(hits)),
        "n_users": len(users),
    }
    if outcome_score is not None:
        out[f"outcome_weighted_precision@{k}"] = float(np.mean(owp))
    if n_items is not None:
        out["catalogue_coverage"] = catalogue_coverage(recommendations, n_items)
    out["recommendation_gini"] = recommendation_gini(recommendations)
    return out


def per_user_metrics(
    recommendations: dict[int, Sequence[int]],
    relevant_per_user: dict[int, Iterable[int]],
    k: int,
    outcome_score: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return per-user metric arrays for downstream bootstrap CI / paired-t
    testing. Users without relevant items are excluded from the arrays."""
    users = [u for u, items in relevant_per_user.items() if items]
    baseline = (
        float(np.mean(outcome_score))
        if outcome_score is not None and outcome_score.size > 0
        else None
    )
    p, r, n, h, o = [], [], [], [], []
    for user in users:
        recs = recommendations.get(user, [])
        rel = relevant_per_user[user]
        p.append(precision_at_k(recs, rel, k))
        r.append(recall_at_k(recs, rel, k))
        n.append(ndcg_at_k(recs, rel, k))
        h.append(hit_rate_at_k(recs, rel, k))
        if outcome_score is not None:
            o.append(
                outcome_weighted_precision_at_k(
                    recs, rel, outcome_score, k, baseline=baseline
                )
            )
    out = {
        f"precision@{k}": np.asarray(p, dtype=np.float64),
        f"recall@{k}": np.asarray(r, dtype=np.float64),
        f"ndcg@{k}": np.asarray(n, dtype=np.float64),
        f"hit_rate@{k}": np.asarray(h, dtype=np.float64),
    }
    if outcome_score is not None:
        out[f"outcome_weighted_precision@{k}"] = np.asarray(o, dtype=np.float64)
    return out
