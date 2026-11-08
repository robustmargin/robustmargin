"""A tiny absolute target on an increasing fit is unreachable because of rounding, and says so.

An increasing fit is solved on its mirrored view, whose target is the complement ``1 - a``. For
``0 < a <= 2**-54`` that complement rounds to exactly 1 in double precision, so no curve can be
resolved against it. The target ``a`` itself lies inside ``(0, 1)``, so the warning must name the
rounding and must not claim that the target is outside the open interval.
"""

from __future__ import annotations

import pytest

from robustmargin import Axis, Definition, Fit, Observations, Status, fit_dose_response, threshold

_LOG_AXIS = Axis(name="severity", unit="unit", scale="log")
_SEVERITY = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
_EVENTS = [20, 22, 35, 90, 220, 340, 380]
_TRIALS = [400] * 7


def _increasing_fit() -> Fit:
    obs = Observations.from_counts(
        _LOG_AXIS,
        severity=_SEVERITY,
        successes=_EVENTS,
        trials=_TRIALS,
        outcome="outcome",
        direction="increasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower="estimate")
    assert fit.status is Status.OK
    return fit


@pytest.mark.parametrize("target", [2.0**-54, 2.0**-60, 5e-324])
def test_tiny_target_on_an_increasing_fit_is_unreachable_without_a_value(target: float) -> None:
    assert 1.0 - target == 1.0

    t = threshold(
        _increasing_fit(),
        definition=Definition.absolute(target),
        interval_method="profile",
        level=0.95,
    )

    assert t.status is Status.UNREACHABLE
    assert t.value is None and t.lo is None and t.hi is None


@pytest.mark.parametrize("target", [2.0**-54, 2.0**-60, 5e-324])
def test_tiny_target_on_an_increasing_fit_is_not_called_outside_zero_one(target: float) -> None:
    t = threshold(
        _increasing_fit(),
        definition=Definition.absolute(target),
        interval_method="profile",
        level=0.95,
    )

    assert not any("outside the open interval" in w for w in t.warnings), t.warnings


@pytest.mark.parametrize("target", [2.0**-54, 2.0**-60, 5e-324])
def test_tiny_target_on_an_increasing_fit_warns_that_the_complement_rounds_to_one(
    target: float,
) -> None:
    t = threshold(
        _increasing_fit(),
        definition=Definition.absolute(target),
        interval_method="profile",
        level=0.95,
    )

    assert any("rounds to 1" in w for w in t.warnings), t.warnings


@pytest.mark.parametrize("target", [0.0, 1.5])
def test_out_of_range_target_on_an_increasing_fit_still_says_outside_the_open_interval(
    target: float,
) -> None:
    t = threshold(
        _increasing_fit(),
        definition=Definition.absolute(target),
        interval_method="profile",
        level=0.95,
    )

    assert t.status is Status.UNREACHABLE
    assert any(
        f"the resolved target {target!r} is outside the open interval (0, 1)" in w
        for w in t.warnings
    ), t.warnings
    assert not any("rounds to 1" in w for w in t.warnings), t.warnings
