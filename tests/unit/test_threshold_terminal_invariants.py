"""``Threshold`` applies its ``baseline`` and ``grid`` rules to terminal statuses too.

Only ``FAILS_AT_BASELINE`` may carry a ``baseline``; only ``SEPARATION`` and ``NOT_CONVERGED``
may carry a ``grid``. ``UNREACHABLE``, ``CONTROL_INCOMPATIBLE`` and ``FAILS_AT_BASELINE`` return
early from the other checks, so these two rules must not be skipped on that path.
"""

from __future__ import annotations

import dataclasses

import pytest

from robustmargin import BaselineRate, GridBreakPoint, Status, Threshold
from robustmargin.empirical import grid_break_point
from robustmargin.report import from_dict, to_dict
from robustmargin.testing import fake_threshold

_TERMINAL = (Status.UNREACHABLE, Status.CONTROL_INCOMPATIBLE, Status.FAILS_AT_BASELINE)
_NOT_FAILS_AT_BASELINE = (Status.UNREACHABLE, Status.CONTROL_INCOMPATIBLE)
_GRID_MESSAGE = "Threshold.grid may only be set"
_BASELINE_MESSAGE = "Threshold.baseline may only be set"


def _grid() -> GridBreakPoint:
    return grid_break_point([0.0, 5.0, 10.0], [1.0, 0.95, 0.4], criterion=0.95)


def _baseline() -> BaselineRate:
    return BaselineRate(
        severity=0.0,
        successes=98,
        trials=200,
        rate=0.49,
        lo=0.4188225152054292,
        hi=0.5614782774378427,
        method="per_cell_clopper_pearson",
    )


def _with(status: Status, **fields: object) -> Threshold:
    return dataclasses.replace(fake_threshold(status=status), **fields)


@pytest.mark.parametrize("status", _TERMINAL)
def test_fake_threshold_still_constructs_for_each_terminal_status(status: Status) -> None:
    threshold = fake_threshold(status=status)

    assert threshold.status is status
    assert threshold.grid is None


@pytest.mark.parametrize("status", _TERMINAL)
def test_terminal_status_with_a_grid_is_rejected(status: Status) -> None:
    with pytest.raises(ValueError, match=_GRID_MESSAGE):
        _with(status, grid=_grid())


@pytest.mark.parametrize("status", _NOT_FAILS_AT_BASELINE)
def test_terminal_status_other_than_fails_at_baseline_with_a_baseline_is_rejected(
    status: Status,
) -> None:
    with pytest.raises(ValueError, match=_BASELINE_MESSAGE):
        _with(status, baseline=_baseline())


def test_fails_at_baseline_with_a_baseline_and_no_grid_is_allowed() -> None:
    threshold = _with(Status.FAILS_AT_BASELINE, baseline=_baseline(), grid=None)

    assert threshold.baseline is not None
    assert threshold.grid is None


def test_fails_at_baseline_without_a_baseline_is_allowed() -> None:
    assert _with(Status.FAILS_AT_BASELINE, baseline=None, grid=None).baseline is None


@pytest.mark.parametrize("status", _NOT_FAILS_AT_BASELINE)
def test_unreachable_and_control_incompatible_with_neither_field_are_allowed(
    status: Status,
) -> None:
    threshold = _with(status, baseline=None, grid=None)

    assert threshold.baseline is None
    assert threshold.grid is None


@pytest.mark.parametrize("status", _TERMINAL)
def test_from_dict_rejects_a_serialised_terminal_threshold_carrying_a_grid(
    status: Status,
) -> None:
    payload = to_dict(fake_threshold(status=status))
    payload["grid"] = to_dict(_grid())

    with pytest.raises(ValueError, match=_GRID_MESSAGE):
        from_dict(payload)


@pytest.mark.parametrize("status", _NOT_FAILS_AT_BASELINE)
def test_from_dict_rejects_a_serialised_non_baseline_threshold_carrying_a_baseline(
    status: Status,
) -> None:
    payload = to_dict(fake_threshold(status=status))
    payload["baseline"] = dataclasses.asdict(_baseline())

    with pytest.raises(ValueError, match=_BASELINE_MESSAGE):
        from_dict(payload)
