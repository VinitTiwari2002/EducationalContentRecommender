"""End-to-end pipeline runner.

Usage:
    python -m src.pipeline                     # full run (load, split, evaluate)
    python -m src.pipeline --rebuild-split     # force rebuild of cached split
    python -m src.pipeline --k 5 10 20         # evaluate at multiple K

Outputs:
    data/processed/{train,test}.npz            # cached split
    evaluation/baseline_results.csv            # metrics table per model x K
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from . import oulad
from .baselines import PopularityRecommender, RandomRecommender
from .metrics import evaluate
from .preprocess import Split, build_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_DIR = PROJECT_ROOT / "evaluation"


def _relevant_per_user(test: sparse.csr_matrix) -> dict[int, list[int]]:
    """Map row index -> list of held-out item column indices."""
    out: dict[int, list[int]] = {}
    for user_row in range(test.shape[0]):
        items = test[user_row].indices.tolist()
        if items:
            out[user_row] = items
    return out


def _recommend_all(
    model, n_users: int, k: int, user_candidates: list[np.ndarray]
) -> dict[int, list[int]]:
    return {
        u: model.recommend(u, k=k, candidates=user_candidates[u])
        for u in range(n_users)
    }


def run(rebuild_split: bool, k_values: list[int]) -> pd.DataFrame:
    cache_exists = (PROCESSED_DIR / "train.npz").exists() and (
        PROCESSED_DIR / "user_candidates.npy"
    ).exists()
    if rebuild_split or not cache_exists:
        print("Loading OULAD CSVs...")
        data = oulad.load()
        print("Summary:")
        print(data.summary().to_string(index=False))
        print("Building temporal split with course-scoped candidates...")
        split = build_split(data)
        split.save(PROCESSED_DIR)
        print(f"Split saved to {PROCESSED_DIR}; cutoff_date={split.cutoff_date}")
    else:
        print(f"Loading cached split from {PROCESSED_DIR}")
        split = Split.load(PROCESSED_DIR)

    candidate_sizes = np.array([len(c) for c in split.user_candidates])
    print(
        f"train shape={split.train.shape}, nnz={split.train.nnz:,}; "
        f"test shape={split.test.shape}, nnz={split.test.nnz:,}; "
        f"cutoff_date={split.cutoff_date}"
    )
    print(
        f"user candidate set sizes: "
        f"min={candidate_sizes.min()}, median={int(np.median(candidate_sizes))}, "
        f"mean={candidate_sizes.mean():.1f}, max={candidate_sizes.max()}"
    )

    models = {
        "Random": RandomRecommender(seed=0),
        "Popularity": PopularityRecommender(),
    }

    relevant = _relevant_per_user(split.test)
    n_users = split.train.shape[0]
    rows = []
    for model_name, model in models.items():
        print(f"Fitting {model_name}...")
        model.fit(split.train)
        for k in k_values:
            print(f"  Generating top-{k} recommendations...")
            recs = _recommend_all(
                model, n_users=n_users, k=k, user_candidates=split.user_candidates
            )
            metrics = evaluate(recs, relevant, k=k)
            rows.append({"model": model_name, "k": k, **metrics})
            print(f"  {model_name} @ K={k}: {metrics}")

    results = pd.DataFrame(rows)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "baseline_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    args = parser.parse_args()
    run(rebuild_split=args.rebuild_split, k_values=args.k)


if __name__ == "__main__":
    main()
