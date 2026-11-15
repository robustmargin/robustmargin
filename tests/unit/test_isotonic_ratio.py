"""``ratio_interval`` refuses a threshold from an isotonic fit.

A ratio interval is built from each threshold's variance, which the fit's covariance supplies. An
isotonic fit has no covariance, so there is nothing to build it from and the call raises
``ValueError`` rather than inventing a variance.
"""

from __future__ import annotations

import pytest

from robustmargin import (
    Axis,
    Definition,
    Observations,
    Status,
    Threshold,
    fit_dose_response,
    ratio_interval,
    threshold,
)
from robustmargin.testing import fake_threshold

_AXIS = Axis(name="severity", unit="unit", scale="log")


def _isotonic_threshold(scale: float = 1.0) -> Threshold:
    obs = Observations.from_counts(
        _AXIS,
        severity=[0.0, scale, 2.0 * scale, 4.0 * scale, 8.0 * scale],
        successes=[100, 95, 70, 30, 5],
        trials=[100] * 5,
        outcome="outcome",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="isotonic")
    t = threshold(
        fit,
        definition=Definition.absolute(0.5),
        interval_method="cell_bootstrap",
        level=0.95,
        n_resamples=49,
        seed=7,
    )
    assert t.status is Status.OK
    return t


@pytest.mark.parametrize("method", ["fieller", "log_delta"])
def test_two_isotonic_thresholds_raise_value_error(method: str) -> None:
    with pytest.raises(ValueError):
        ratio_interval(
            _isotonic_threshold(), _isotonic_threshold(2.0), method=method, dependence="independent"
        )


@pytest.mark.parametrize("method", ["fieller", "log_delta"])
def test_an_isotonic_numerator_with_a_parametric_denominator_raises_value_error(
    method: str,
) -> None:
    with pytest.raises(ValueError):
        ratio_interval(
            _isotonic_threshold(), fake_threshold(), method=method, dependence="independent"
        )


@pytest.mark.parametrize("method", ["fieller", "log_delta"])
def test_a_parametric_numerator_with_an_isotonic_denominator_raises_value_error(
    method: str,
) -> None:
    with pytest.raises(ValueError):
        ratio_interval(
            fake_threshold(), _isotonic_threshold(), method=method, dependence="independent"
        )


def test_a_fake_isotonic_threshold_is_refused_too() -> None:
    with pytest.raises(ValueError):
        ratio_interval(
            fake_threshold(model="isotonic"),
            fake_threshold(model="isotonic"),
            method="log_delta",
            dependence="independent",
        )


def test_the_same_pair_of_parametric_fakes_is_accepted() -> None:
    ratio = ratio_interval(
        fake_threshold(), fake_threshold(), method="log_delta", dependence="independent"
    )

    assert ratio.status is Status.OK
