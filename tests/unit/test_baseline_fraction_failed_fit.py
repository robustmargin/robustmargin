"""``Definition.baseline_fraction(p)`` on a fit whose status is not ``OK``.

A failed fit has no fitted upper asymptote, and the control's observed rate is only an estimate of
it, so ``baseline_fraction(p)`` has no target and ``threshold`` raises ``ValueError`` (as
``baseline_multiple`` does). The message names the kind and the fit's status and points to
``absolute(a)``, which needs no asymptote. On an ``OK`` fit nothing changes: the target is ``p``
times the fitted upper asymptote.
"""

from __future__ import annotations

import dataclasses

import pytest

from robustmargin import (
    Axis,
    Cell,
    Definition,
    Fit,
    Observations,
    Status,
    Threshold,
    fit_dose_response,
    threshold,
)
from robustmargin.testing import fake_fit

_LEVEL = 0.95
_AXIS = Axis(name="severity", unit="unit", scale="linear")


def _failed_fit(status: Status) -> Fit:
    cells = (
        Cell(severity=0.0, successes=95, trials=100),
        Cell(severity=1.0, successes=80, trials=100),
        Cell(severity=2.0, successes=40, trials=100),
        Cell(severity=3.0, successes=5, trials=100),
    )
    return dataclasses.replace(fake_fit(status=status, monotonicity=None), cells=cells)


@pytest.mark.parametrize(
    "status", [Status.SEPARATION, Status.NOT_CONVERGED, Status.CONTROL_INCOMPATIBLE]
)
class TestRefusalOnAFailedFit:
    def test_baseline_fraction_raises_naming_kind_status_and_the_alternative(
        self, status: Status
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            threshold(
                _failed_fit(status),
                definition=Definition.baseline_fraction(0.5),
                interval_method="profile",
                level=_LEVEL,
            )

        message = str(excinfo.value)
        assert "kind='baseline_fraction'" in message
        assert status.value in message
        assert "absolute(" in message

    def test_the_same_data_with_an_absolute_definition_still_returns_a_threshold(
        self, status: Status
    ) -> None:
        t = threshold(
            _failed_fit(status),
            definition=Definition.absolute(0.5),
            interval_method="profile",
            level=_LEVEL,
        )

        assert isinstance(t, Threshold)
        assert t.fit.status is status


def test_baseline_fraction_on_an_ok_fit_still_targets_p_times_the_fitted_upper() -> None:
    obs = Observations.from_counts(
        _AXIS,
        severity=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        successes=[95, 93, 85, 60, 30, 8, 2],
        trials=[100] * 7,
        outcome="success",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
    assert fit.status is Status.OK
    assert fit.params is not None
    upper = fit.params["upper"].value

    by_fraction = threshold(
        fit, definition=Definition.baseline_fraction(0.5), interval_method="profile", level=_LEVEL
    )
    by_absolute = threshold(
        fit,
        definition=Definition.absolute(0.5 * upper),
        interval_method="profile",
        level=_LEVEL,
    )

    assert by_fraction.status is Status.OK
    assert by_fraction.value == pytest.approx(by_absolute.value, rel=1e-6)
