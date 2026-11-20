"""``robustmargin.design.severity_grid``: values, control handling and error cases.

A log axis gets a geometric grid and a linear axis an arithmetic one. The zero-severity control
is prepended when ``control=True``, except on a linear axis whose ``lowest`` is already zero
(a second zero would duplicate the level). Endpoints are exactly the values the caller gave, not
round-off from the spacing function.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robustmargin import Axis
from robustmargin.design import severity_grid

_LOG = Axis(name="severity", unit="unit", scale="log")
_LINEAR = Axis(name="severity", unit="unit", scale="linear")


def test_log_axis_grid_is_geometric_with_a_leading_control() -> None:
    grid = severity_grid(_LOG, lowest=0.1, highest=10.0, n_levels=5)
    assert grid[0] == 0.0
    levels = grid[1:]
    assert len(levels) == 5
    ratios = [b / a for a, b in zip(levels, levels[1:], strict=False)]
    assert ratios == pytest.approx([ratios[0]] * 4, rel=1e-12)
    assert levels == pytest.approx(tuple(np.geomspace(0.1, 10.0, 5)), rel=1e-12)


def test_log_axis_grid_without_control_has_exactly_n_levels() -> None:
    grid = severity_grid(_LOG, lowest=0.1, highest=10.0, n_levels=5, control=False)
    assert len(grid) == 5
    assert 0.0 not in grid


def test_linear_axis_grid_is_arithmetic_with_a_leading_control() -> None:
    grid = severity_grid(_LINEAR, lowest=1.0, highest=5.0, n_levels=5)
    assert grid == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


def test_linear_axis_grid_does_not_duplicate_a_zero_lowest_level() -> None:
    grid = severity_grid(_LINEAR, lowest=0.0, highest=4.0, n_levels=5)
    assert grid == (0.0, 1.0, 2.0, 3.0, 4.0)
    assert grid.count(0.0) == 1


def test_linear_axis_grid_without_control_is_just_the_levels() -> None:
    grid = severity_grid(_LINEAR, lowest=1.0, highest=5.0, n_levels=3, control=False)
    assert grid == (1.0, 3.0, 5.0)


def test_grid_endpoints_equal_the_inputs_exactly() -> None:
    lowest, highest = 0.37, 11.3
    for axis in (_LOG, _LINEAR):
        grid = severity_grid(axis, lowest=lowest, highest=highest, n_levels=7, control=False)
        assert grid[0] == lowest
        assert grid[-1] == highest


def test_grid_is_a_tuple_of_plain_python_floats() -> None:
    for axis in (_LOG, _LINEAR):
        grid = severity_grid(axis, lowest=1.0, highest=8.0, n_levels=4)
        assert isinstance(grid, tuple)
        assert all(type(v) is float for v in grid)


def test_grid_is_strictly_increasing() -> None:
    for axis in (_LOG, _LINEAR):
        grid = severity_grid(axis, lowest=0.5, highest=50.0, n_levels=9)
        assert all(a < b for a, b in zip(grid, grid[1:], strict=False))
        assert all(math.isfinite(v) for v in grid)


def test_two_levels_are_the_two_endpoints() -> None:
    grid = severity_grid(_LOG, lowest=2.0, highest=8.0, n_levels=2, control=False)
    assert grid == (2.0, 8.0)


@pytest.mark.parametrize("lowest", [0.0, -1.0])
def test_log_axis_rejects_a_non_positive_lowest(lowest: float) -> None:
    with pytest.raises(ValueError):
        severity_grid(_LOG, lowest=lowest, highest=10.0, n_levels=5)


@pytest.mark.parametrize("highest", [1.0, 0.5])
def test_highest_not_above_lowest_is_rejected(highest: float) -> None:
    for axis in (_LOG, _LINEAR):
        with pytest.raises(ValueError):
            severity_grid(axis, lowest=1.0, highest=highest, n_levels=5)


@pytest.mark.parametrize("n_levels", [1, 0, -3])
def test_fewer_than_two_levels_is_rejected(n_levels: int) -> None:
    for axis in (_LOG, _LINEAR):
        with pytest.raises(ValueError):
            severity_grid(axis, lowest=1.0, highest=10.0, n_levels=n_levels)
