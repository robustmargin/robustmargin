"""An open-ended profile interval must still contain its own point estimate.

``OPEN_UPPER`` means the upper side of the profile did not close: ``hi`` is ``None`` and ``lo`` is
the closed lower side, so ``lo <= value``. ``OPEN_LOWER`` is the mirror: ``lo`` is ``None`` and
``hi`` is the closed upper side, so ``value <= hi``. A bound on the wrong side of the estimate is
not a bound of an interval around it, and ``Threshold`` refuses to be built that way, so a search
that mislocates a root cannot reach a stored result. (A bounded ``OK`` result already requires
``lo <= value <= hi``.)

Touching is allowed: the constraints are non-strict, matching the bounded case.
"""

from __future__ import annotations

import dataclasses

import pytest

from robustmargin import Censoring, Threshold
from robustmargin.testing import fake_threshold


def _open_upper() -> Threshold:
    t = fake_threshold(censoring=Censoring.OPEN_UPPER)
    assert t.value is not None and t.lo is not None and t.hi is None
    assert t.lo <= t.value
    return t


def _open_lower() -> Threshold:
    t = fake_threshold(censoring=Censoring.OPEN_LOWER)
    assert t.value is not None and t.hi is not None and t.lo is None
    assert t.value <= t.hi
    return t


class TestOpenUpper:
    def test_a_lower_bound_above_the_value_raises(self) -> None:
        t = _open_upper()
        assert t.value is not None

        with pytest.raises(ValueError):
            dataclasses.replace(t, lo=t.value * 1.01)

    def test_a_lower_bound_far_above_the_value_raises(self) -> None:
        t = _open_upper()
        assert t.value is not None

        with pytest.raises(ValueError):
            dataclasses.replace(t, lo=t.value * 100.0)

    def test_a_lower_bound_equal_to_the_value_is_accepted(self) -> None:
        t = _open_upper()

        touching = dataclasses.replace(t, lo=t.value)

        assert touching.lo == touching.value

    def test_a_lower_bound_below_the_value_is_accepted(self) -> None:
        assert _open_upper().lo is not None


class TestOpenLower:
    def test_an_upper_bound_below_the_value_raises(self) -> None:
        t = _open_lower()
        assert t.value is not None

        with pytest.raises(ValueError):
            dataclasses.replace(t, hi=t.value * 0.99)

    def test_an_upper_bound_far_below_the_value_raises(self) -> None:
        t = _open_lower()
        assert t.value is not None

        with pytest.raises(ValueError):
            dataclasses.replace(t, hi=t.value * 0.01)

    def test_an_upper_bound_equal_to_the_value_is_accepted(self) -> None:
        t = _open_lower()

        touching = dataclasses.replace(t, hi=t.value)

        assert touching.hi == touching.value

    def test_an_upper_bound_above_the_value_is_accepted(self) -> None:
        assert _open_lower().hi is not None
