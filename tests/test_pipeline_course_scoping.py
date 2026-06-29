"""Test that the pipeline runner honours the course_scoping flag.

Behavioural contract: when course_scoping=False, the recommendation step
should pass None for the candidate pool (giving the model the full
catalogue). When course_scoping=True, it should pass the per-user
candidate sets from the split.

This locks in the diagnostic mode used to reproduce the Popularity@10 =
0.00003 finding documented in Chapter 4 of the prelim report.
"""
import numpy as np
import pytest
from scipy import sparse

from src.pipeline import _recommend_all


class _RecordingRecommender:
    """Spy that records the candidate argument passed to recommend()."""

    def __init__(self):
        self.calls: list[tuple[int, np.ndarray | None]] = []

    def recommend(self, user_row, k, candidates=None, exclude_seen=True):
        self.calls.append((user_row, candidates))
        return [0, 1, 2]


def test_recommend_all_uses_full_catalogue_when_candidates_none():
    spy = _RecordingRecommender()
    _recommend_all(spy, n_users=3, k=3, user_candidates=None)
    assert len(spy.calls) == 3
    # Every call should have received None for candidates
    assert all(call[1] is None for call in spy.calls)


def test_recommend_all_uses_per_user_candidates_when_provided():
    spy = _RecordingRecommender()
    user_candidates = [
        np.array([10, 20]),
        np.array([30, 40, 50]),
        np.array([60]),
    ]
    _recommend_all(spy, n_users=3, k=3, user_candidates=user_candidates)
    assert len(spy.calls) == 3
    # Each call should have received that user's candidate array
    for (user_row, candidates), expected in zip(spy.calls, user_candidates):
        assert candidates is not None
        np.testing.assert_array_equal(candidates, expected)
