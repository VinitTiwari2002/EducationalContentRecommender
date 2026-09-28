"""Robustness check: refit every seeded recommender with a range of
`random_state` values and report the spread of the headline metrics.

This addresses the §5.10-adjacent concern about numerical stability: are
the numbers cited in Table 5.1 driven by the specific `random_state=0`
we happened to pick, or would any reasonable seed produce the same
model ordering?

Output:
    evaluation/robustness.csv  — per (seed, model) row with metrics
    stdout                     — per-model median + (min, max) summary

The check reuses the cached primary split so a single run finishes in a
couple of minutes on top of an already-persisted split. The three seeds
run sequentially rather than in parallel to keep memory usage modest;
change SEEDS below if you want a wider sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import oulad  # noqa: E402
from src.als import ALSRecommender  # noqa: E402
from src.baselines import PopularityRecommender, RandomRecommender  # noqa: E402
from src.content import ContentRecommender  # noqa: E402
from src.features import build_item_features  # noqa: E402
from src.gated_hybrid import GatedHybridRecommender  # noqa: E402
from src.hybrid import HybridRecommender  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.preprocess import Split  # noqa: E402
from src.svd import SVDRecommender  # noqa: E402


SEEDS = [0, 1, 2]
DECAY_RATE = 0.01  # match the primary evaluation run
COLD_START_THRESHOLD = 10
K = 10

# Tuned hyperparameters — these match Table 5.1 in the evaluation chapter.
SVD_N_FACTORS = 100
ALS_N_FACTORS = 32
ALS_REGULARIZATION = 0.01
ALS_ITERATIONS = 15
ALS_ALPHA = 1.0
HYBRID_WEIGHTS = (0.0, 0.8, 0.2)


def _fit_seeded(train, features, seed):
    """Fit every recommender using `seed` wherever a seed is accepted."""
    random = RandomRecommender(seed=seed)
    random.fit(train)
    popularity = PopularityRecommender()
    popularity.fit(train)
    svd = SVDRecommender(n_factors=SVD_N_FACTORS, random_state=seed)
    svd.fit(train)
    als = ALSRecommender(
        n_factors=ALS_N_FACTORS,
        regularization=ALS_REGULARIZATION,
        iterations=ALS_ITERATIONS,
        alpha=ALS_ALPHA,
        random_state=seed,
    )
    als.fit(train)
    content = ContentRecommender()
    content.fit(train, features.matrix)
    alpha, beta, gamma = HYBRID_WEIGHTS
    hybrid = HybridRecommender(
        alpha=alpha, beta=beta, gamma=gamma,
        cf=ALSRecommender(
            n_factors=ALS_N_FACTORS,
            regularization=ALS_REGULARIZATION,
            iterations=ALS_ITERATIONS,
            alpha=ALS_ALPHA,
            random_state=seed + 1,
        ),
        content=ContentRecommender(),
    )
    hybrid.fit(train, features.matrix, features.outcome_score)
    gated = GatedHybridRecommender(
        cold_model=PopularityRecommender(),
        warm_model=HybridRecommender(
            alpha=alpha, beta=beta, gamma=gamma,
            cf=ALSRecommender(
                n_factors=ALS_N_FACTORS,
                regularization=ALS_REGULARIZATION,
                iterations=ALS_ITERATIONS,
                alpha=ALS_ALPHA,
                random_state=seed + 2,
            ),
            content=ContentRecommender(),
        ),
        threshold=COLD_START_THRESHOLD,
    )
    gated.fit(train, features.matrix, features.outcome_score)
    return {
        "Random": random,
        "Popularity": popularity,
        "SVD": svd,
        "ALS": als,
        "Content": content,
        "Hybrid": hybrid,
        "GatedHybrid": gated,
    }


def _relevant_per_user(test):
    out = {}
    for row in range(test.shape[0]):
        items = test[row].indices.tolist()
        if items:
            out[row] = items
    return out


def _recommend_all(model, n_users, k, candidates):
    return {
        u: model.recommend(u, k=k, candidates=candidates[u])
        for u in range(n_users)
    }


def main() -> None:
    print("Loading cached primary split...")
    split = Split.load(PROJECT_ROOT / "data" / "processed")
    print("Loading OULAD for feature construction...")
    data = oulad.load()
    features = build_item_features(data, split.train, split.item_index, split.student_index)
    relevant = _relevant_per_user(split.test)
    n_users = split.train.shape[0]
    n_items = split.train.shape[1]

    rows = []
    for seed in SEEDS:
        print(f"\n=== Refitting with random_state={seed} ===")
        models = _fit_seeded(split.train, features, seed)
        for name, model in models.items():
            recs = _recommend_all(model, n_users, K, split.user_candidates)
            m = evaluate(
                recs, relevant, k=K,
                outcome_score=features.outcome_score,
                n_items=n_items,
            )
            rows.append({"seed": seed, "model": name, **m})
            print(
                f"  {name:12s} P@{K}={m[f'precision@{K}']:.4f}  "
                f"NDCG@{K}={m[f'ndcg@{K}']:.4f}  "
                f"OWP@{K}={m.get(f'outcome_weighted_precision@{K}', float('nan')):.4f}"
            )

    df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "evaluation" / "robustness.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved per-seed metrics to {out_path}")

    # Per-model summary: median + (min, max) across seeds.
    print("\n=== Per-model summary across seeds ===")
    metric_cols = [f"precision@{K}", f"ndcg@{K}", f"hit_rate@{K}",
                   f"outcome_weighted_precision@{K}"]
    summary_rows = []
    for name, sub in df.groupby("model"):
        row = {"model": name, "n_seeds": len(sub)}
        for col in metric_cols:
            values = sub[col].to_numpy()
            row[f"{col}_median"] = float(np.median(values))
            row[f"{col}_min"] = float(values.min())
            row[f"{col}_max"] = float(values.max())
            row[f"{col}_range"] = float(values.max() - values.min())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values(f"ndcg@{K}_median", ascending=False)
    summary_path = PROJECT_ROOT / "evaluation" / "robustness_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
