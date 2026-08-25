"""Unit tests for outcome-weighted precision, catalogue coverage, and Gini."""
import numpy as np
import pytest

from src.metrics import (
    catalogue_coverage,
    evaluate,
    outcome_weighted_precision_at_k,
    per_user_metrics,
    recommendation_gini,
)


def test_owp_reduces_to_precision_when_outcomes_are_uniform():
    # If every item has the same outcome, o(i)/mean(o) = 1, so OWP@K == precision@K.
    outcome = np.full(5, 70.0, dtype=np.float32)
    recs = [0, 1, 2]
    relevant = {0, 2}
    owp = outcome_weighted_precision_at_k(recs, relevant, outcome, k=3)
    # precision@3 = 2/3
    assert abs(owp - 2 / 3) < 1e-6


def test_owp_up_weights_high_outcome_hits():
    outcome = np.array([50.0, 100.0, 50.0, 50.0, 50.0], dtype=np.float32)  # mean=60
    recs = [1, 2, 3]  # only item 1 is a hit, and it's above-average outcome
    relevant = {1}
    owp = outcome_weighted_precision_at_k(recs, relevant, outcome, k=3)
    # OWP = (1/3) * (100 / 60) = 5/9
    assert abs(owp - 5 / 9) < 1e-6


def test_owp_down_weights_low_outcome_hits():
    outcome = np.array([50.0, 100.0, 50.0, 50.0, 50.0], dtype=np.float32)  # mean=60
    recs = [0, 2, 3]  # item 0 is a hit, below-average
    relevant = {0}
    owp = outcome_weighted_precision_at_k(recs, relevant, outcome, k=3)
    # OWP = (1/3) * (50 / 60) = 5/18
    assert abs(owp - 5 / 18) < 1e-6


def test_owp_zero_when_no_hits():
    outcome = np.full(5, 70.0, dtype=np.float32)
    assert outcome_weighted_precision_at_k([1, 2], {4}, outcome, k=2) == 0.0


def test_owp_zero_when_no_relevant():
    outcome = np.full(5, 70.0, dtype=np.float32)
    assert outcome_weighted_precision_at_k([0, 1], set(), outcome, k=2) == 0.0


def test_owp_zero_when_no_recommendations():
    outcome = np.full(5, 70.0, dtype=np.float32)
    assert outcome_weighted_precision_at_k([], {0}, outcome, k=2) == 0.0


def test_owp_handles_zero_baseline():
    # If everyone scores zero, baseline defaults to 1.0 (avoid divide-by-zero).
    outcome = np.zeros(5, dtype=np.float32)
    assert outcome_weighted_precision_at_k([0], {0}, outcome, k=1) == 0.0


def test_catalogue_coverage():
    recs = {0: [0, 1], 1: [1, 2]}
    # 3 unique items out of 10 in the catalogue
    assert abs(catalogue_coverage(recs, n_items=10) - 0.3) < 1e-9


def test_catalogue_coverage_empty():
    assert catalogue_coverage({}, n_items=10) == 0.0
    assert catalogue_coverage({0: [1, 2]}, n_items=0) == 0.0


def test_gini_uniform_is_zero():
    # Every item recommended once → perfectly uniform → Gini = 0.
    recs = {0: [0], 1: [1], 2: [2], 3: [3]}
    assert abs(recommendation_gini(recs)) < 1e-9


def test_gini_concentrated_is_high():
    # Only one item ever recommended → Gini approaches 1 (but with n=1 items
    # in the distribution the formula returns 0; test the n>1 case).
    recs = {0: [0, 0, 0, 1]}  # 1 appears once, 0 appears three times
    g = recommendation_gini(recs)
    assert 0 < g < 1


def test_evaluate_includes_owp_when_outcome_provided():
    outcome = np.full(4, 70.0, dtype=np.float32)
    recs = {0: [0, 1, 2, 3]}
    relevant = {0: [0, 3]}
    m = evaluate(recs, relevant, k=4, outcome_score=outcome, n_items=4)
    assert "outcome_weighted_precision@4" in m
    assert "catalogue_coverage" in m
    assert "recommendation_gini" in m


def test_evaluate_excludes_owp_when_outcome_absent():
    recs = {0: [0, 1]}
    relevant = {0: [0]}
    m = evaluate(recs, relevant, k=2)
    assert "outcome_weighted_precision@2" not in m
    assert "catalogue_coverage" not in m


def test_per_user_metrics_arrays_align():
    outcome = np.full(4, 70.0, dtype=np.float32)
    recs = {0: [0, 1], 1: [2, 3], 2: [0]}
    relevant = {0: [0], 1: [3], 2: []}  # user 2 skipped (no relevant)
    per = per_user_metrics(recs, relevant, k=2, outcome_score=outcome)
    for arr in per.values():
        assert arr.shape == (2,)
