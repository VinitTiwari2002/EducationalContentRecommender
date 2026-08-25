"""Unit tests for the tuning module."""
import numpy as np
import pytest

from src.tuning import simplex_grid


def test_simplex_grid_step_0_5():
    grid = simplex_grid(step=0.5)
    # Points on the simplex at step 0.5: (1,0,0), (0,1,0), (0,0,1),
    # (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5).
    assert len(grid) == 6
    for w in grid:
        assert abs(sum(w) - 1.0) < 1e-9


def test_simplex_grid_step_0_2():
    grid = simplex_grid(step=0.2)
    # Number of compositions of 5 into 3 non-neg parts = C(5+2, 2) = 21.
    assert len(grid) == 21
    for w in grid:
        assert abs(sum(w) - 1.0) < 1e-9
        assert all(0 <= x <= 1 for x in w)


def test_simplex_grid_no_degenerate_triple():
    grid = simplex_grid(step=0.1)
    assert (0.0, 0.0, 0.0) not in grid


def test_simplex_grid_invalid_step_raises():
    with pytest.raises(ValueError):
        simplex_grid(step=0.0)
    with pytest.raises(ValueError):
        simplex_grid(step=1.1)


def test_simplex_grid_covers_corners():
    grid = set(simplex_grid(step=0.5))
    assert (1.0, 0.0, 0.0) in grid
    assert (0.0, 1.0, 0.0) in grid
    assert (0.0, 0.0, 1.0) in grid
