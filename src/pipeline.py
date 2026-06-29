"""End-to-end pipeline runner.

Usage:
    python -m src.pipeline                     # full run with course-scoping (default)
    python -m src.pipeline --rebuild-split     # force rebuild of cached split
    python -m src.pipeline --k 5 10 20         # evaluate at multiple K
    python -m src.pipeline --no-course-scoping # diagnostic: skip per-user
                                                  candidate restriction

The --no-course-scoping flag is a diagnostic only. It lets recommenders
propose any item in the global catalogue, ignoring which presentation each
user is enrolled in. Comparing the two runs is what surfaced the
course-scoping methodological finding documented in the prelim report.

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
    model, n_users: int, k: int, user_candidates: list[np.ndarray] | None
) -> dict[int, list[int]]:
    """Generate top-K recommendations for every user.

    When user_candidates is None, the recommender sees the full catalogue
    (the --no-course-scoping diagnostic mode). Otherwise it restricts to
    the user's enrolled-presentation items.
    """
    if user_candidates is None:
        return {u: model.recommend(u, k=k) for u in range(n_users)}
    return {
        u: model.recommend(u, k=k, candidates=user_candidates[u])
        for u in range(n_users)
    }


def run(
    rebuild_split: bool,
    k_values: list[int],
    course_scoping: bool = True,
) -> pd.DataFrame:
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
    if not course_scoping:
        print(
            "\n*** DIAGNOSTIC MODE: course-scoping DISABLED ***\n"
            "Recommenders will see the full catalogue, ignoring per-user enrolment.\n"
            "This reproduces the pathological Popularity@K result reported in the\n"
            "prelim report and is NOT the production configuration.\n"
        )

    models = {
        "Random": RandomRecommender(seed=0),
        "Popularity": PopularityRecommender(),
    }

    relevant = _relevant_per_user(split.test)
    n_users = split.train.shape[0]
    candidates = split.user_candidates if course_scoping else None
    rows = []
    for model_name, model in models.items():
        print(f"Fitting {model_name}...")
        model.fit(split.train)
        for k in k_values:
            print(f"  Generating top-{k} recommendations...")
            recs = _recommend_all(
                model, n_users=n_users, k=k, user_candidates=candidates
            )
            metrics = evaluate(recs, relevant, k=k)
            rows.append({"model": model_name, "k": k, **metrics})
            print(f"  {model_name} @ K={k}: {metrics}")

    results = pd.DataFrame(rows)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if course_scoping else "_no_course_scoping"
    out_path = EVAL_DIR / f"baseline_results{suffix}.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument(
        "--no-course-scoping",
        dest="course_scoping",
        action="store_false",
        help=(
            "Diagnostic: disable per-user course-scoped candidate restriction. "
            "Reproduces the pathological Popularity@K result documented in the "
            "prelim report. NOT the production configuration."
        ),
    )
    parser.set_defaults(course_scoping=True)
    args = parser.parse_args()
    run(
        rebuild_split=args.rebuild_split,
        k_values=args.k,
        course_scoping=args.course_scoping,
    )


if __name__ == "__main__":
    main()
