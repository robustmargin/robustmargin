"""A resolved target outside ``(0, 1)`` is ``UNREACHABLE``, on any fit status.

No binomial curve crosses a target at or below 0 or at or above 1, and the exact one-sided tests
are trivial against one: against a target of 0 every ``pass`` is true, and against a target of 1
``fail`` is true for any control with a failure. So ``threshold()`` decides this before any exact
test, whatever the fit's status, and reports it with no numbers and a warning naming the target.

Two routes reach such a target:

* ``Definition.absolute(a)`` with ``a`` of 0, 1, above 1 or below 0 (``Definition`` accepts any
  finite ``a``; reachability is decided here, not at construction);
* ``Definition.baseline_fraction(p)`` on a failed fit, where the target is ``p`` times the
  control's observed rate and the control is ``0/n``, so the target is exactly 0. The warning that
  records where that target came from names the control count.

This needs ``Threshold`` to accept ``UNREACHABLE`` on a ``SEPARATION`` / ``NOT_CONVERGED`` fit;
the last test checks that it also survives a JSON round trip.
"""

from __future__ import annotations

import dataclasses
import json
import warnings as warnings_module
from collections.abc import Callable

import jsonschema
import pytest

from robustmargin import (
    Axis,
    Cell,
    Censoring,
    Definition,
    Fit,
    Observations,
    Scorecard,
    Status,
    Threshold,
    fit_dose_response,
    threshold,
)
from robustmargin.report import from_dict, load_schema, to_dict
from robustmargin.testing import fake_fit

_LEVEL = 0.95
_LOG_AXIS = Axis(name="severity", unit="unit", scale="log")
_TARGETS = [0.0, 1.0, 1.5, -0.1]


def _observations(
    severity: list[float], successes: list[int], trials: int = 100, axis: Axis = _LOG_AXIS
) -> Observations:
    return Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=[trials] * len(severity),
        outcome="success",
        direction="decreasing",
    )


def _converged_with_a_failing_control() -> Fit:
    """Upper asymptote estimated (about 0.956), control 96/100: the control fails against a
    target of 1 or more, so a rule that tested the control before the target's range would call
    this ``FAILS_AT_BASELINE``."""
    fit = fit_dose_response(
        _observations([0.0, 1.0, 2.0, 4.0, 8.0], [96, 90, 70, 30, 5]),
        model="binomial",
        link="probit",
        upper="estimate",
        lower=0.0,
    )
    assert fit.status is Status.OK
    return fit


def _converged_with_fixed_asymptotes() -> Fit:
    fit = fit_dose_response(
        _observations([0.0, 1.0, 2.0, 4.0, 8.0], [100, 95, 70, 30, 5]),
        model="binomial",
        link="probit",
        upper=1.0,
        lower=0.0,
    )
    assert fit.status is Status.OK
    return fit


def _separation() -> Fit:
    """Complete separation (every level is all-success or all-failure), a deterministic pre-fit
    verdict."""
    fit = fit_dose_response(
        _observations([0.0, 1.0, 2.0, 3.0, 4.0], [100, 100, 100, 0, 0]),
        model="binomial",
        link="probit",
        upper=1.0,
        lower=0.0,
    )
    assert fit.status is Status.SEPARATION
    return fit


def _not_converged() -> Fit:
    """Success rising with severity under ``direction="decreasing"``: the wrong-sign check, a
    documented boundary rather than an optimiser give-up point."""
    fit = fit_dose_response(
        _observations([1.0, 2.0, 4.0, 8.0], [20, 35, 50, 65]),
        model="binomial",
        link="probit",
        upper="estimate",
        lower=0.0,
    )
    assert fit.status is Status.NOT_CONVERGED
    return fit


_FIT_BUILDERS = {
    "converged_estimated_upper": _converged_with_a_failing_control,
    "converged_fixed_asymptotes": _converged_with_fixed_asymptotes,
    "separation": _separation,
    "not_converged": _not_converged,
}


@pytest.fixture(scope="module", params=list(_FIT_BUILDERS))
def any_fit(request: pytest.FixtureRequest) -> Fit:
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        fit: Fit = _FIT_BUILDERS[request.param]()
    return fit


class TestAbsoluteTargetsOutsideZeroOne:
    @pytest.mark.parametrize("target", _TARGETS)
    def test_the_result_is_unreachable_with_no_numbers_and_a_warning_naming_the_target(
        self, any_fit: Fit, target: float
    ) -> None:
        t = threshold(
            any_fit,
            definition=Definition.absolute(target),
            interval_method="profile",
            level=_LEVEL,
        )

        assert t.status is Status.UNREACHABLE
        assert t.censoring is Censoring.NONE
        assert t.value is None and t.lo is None and t.hi is None
        assert t.interval_method == "exact_bound"
        assert t.baseline is None
        assert t.grid is None
        assert t.fit is any_fit
        assert any(
            f"the resolved target {target!r} is outside the open interval (0, 1)" in w
            for w in t.warnings
        ), f"no warning names the target {target!r}: {t.warnings!r}"

    @pytest.mark.parametrize("target", _TARGETS)
    def test_the_requested_interval_method_makes_no_difference(
        self, any_fit: Fit, target: float
    ) -> None:
        profile = threshold(
            any_fit,
            definition=Definition.absolute(target),
            interval_method="profile",
            level=_LEVEL,
        )
        delta = threshold(
            any_fit,
            definition=Definition.absolute(target),
            interval_method="delta",
            level=_LEVEL,
        )

        assert (profile.status, profile.value, profile.interval_method) == (
            delta.status,
            delta.value,
            delta.interval_method,
        )
        assert delta.status is Status.UNREACHABLE


class TestTargetsJustInsideTheRangeAreClassifiedNotRejected:
    """The range is the open interval, so 0.9999 and 0.0001 are ordinary targets on a failed fit:
    the exact tests run and the result keeps the fit's own status."""

    @pytest.mark.parametrize("target", [0.0001, 0.9999])
    @pytest.mark.parametrize("builder", [_separation, _not_converged], ids=["sep", "nc"])
    def test_a_failed_fit_keeps_its_status(self, builder: Callable[[], Fit], target: float) -> None:
        fit = builder()

        t = threshold(
            fit,
            definition=Definition.absolute(target),
            interval_method="profile",
            level=_LEVEL,
        )

        assert t.status is fit.status
        assert t.grid is not None


class TestBaselineFractionOnAFailedFitWithAZeroControl:
    """The target is ``p * (control successes / control trials)``, so a ``0/n`` control makes it
    exactly 0 and every exact ``pass`` trivially true. A ``RIGHT`` bound at the highest severity
    would then be a claim made against a target no curve can fall to, so the result must be
    ``UNREACHABLE`` instead."""

    @staticmethod
    def _zero_control_separation() -> Fit:
        """Nothing succeeded anywhere: a real ``SEPARATION`` fit with a ``0/100`` control."""
        with warnings_module.catch_warnings():
            warnings_module.simplefilter("ignore")
            fit = fit_dose_response(
                _observations([0.0, 1.0, 2.0, 3.0], [0, 0, 0, 0]),
                model="binomial",
                link="probit",
                upper="estimate",
                lower=0.0,
            )
        assert fit.status is Status.SEPARATION
        return fit

    @staticmethod
    def _hand_built(status: Status) -> Fit:
        cells = (
            Cell(severity=0.0, successes=0, trials=100),
            Cell(severity=1.0, successes=60, trials=100),
            Cell(severity=2.0, successes=10, trials=100),
        )
        return dataclasses.replace(fake_fit(status=status, monotonicity=None), cells=cells)

    def test_a_real_separation_fit_with_a_zero_control_is_unreachable_and_names_the_count(
        self,
    ) -> None:
        fit = self._zero_control_separation()

        t = threshold(
            fit,
            definition=Definition.baseline_fraction(0.5),
            interval_method="profile",
            level=_LEVEL,
        )

        assert t.status is Status.UNREACHABLE
        assert t.censoring is Censoring.NONE
        assert t.value is None and t.lo is None and t.hi is None
        assert t.grid is None and t.baseline is None
        assert any("0/100" in w for w in t.warnings), f"no warning names 0/100: {t.warnings!r}"
        # The control's own rate prints as "0.0" too, so the bare substring would match the
        # warning about where the target came from. Match the sentence that names the *target*.
        expected_target = 0.5 * (0 / 100)
        assert any(
            f"the resolved target {expected_target!r} is outside the open interval (0, 1)" in w
            for w in t.warnings
        ), f"no warning names the resolved target {expected_target!r}: {t.warnings!r}"

    @pytest.mark.parametrize("status", [Status.SEPARATION, Status.NOT_CONVERGED])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_every_fraction_of_a_zero_control_is_unreachable_on_a_failed_fit(
        self, status: Status, p: float
    ) -> None:
        t = threshold(
            self._hand_built(status),
            definition=Definition.baseline_fraction(p),
            interval_method="profile",
            level=_LEVEL,
        )

        assert t.status is Status.UNREACHABLE
        assert t.fit.status is status
        assert any("0/100" in w for w in t.warnings)

    def test_an_all_success_control_resolves_to_the_fraction_itself_and_is_not_unreachable(
        self,
    ) -> None:
        """The contrast: with a ``100/100`` control the target is ``p`` and inside ``(0, 1)``."""
        cells = (
            Cell(severity=0.0, successes=100, trials=100),
            Cell(severity=1.0, successes=100, trials=100),
            Cell(severity=2.0, successes=0, trials=100),
        )
        fit = dataclasses.replace(
            fake_fit(status=Status.SEPARATION, monotonicity=None), cells=cells
        )

        t = threshold(
            fit,
            definition=Definition.baseline_fraction(0.5),
            interval_method="profile",
            level=_LEVEL,
        )

        assert t.status is Status.SEPARATION
        assert t.lo == 1.0 and t.hi == 2.0

    def test_relative_on_a_failed_fit_still_refuses_rather_than_resolving_a_target(self) -> None:
        with pytest.raises(ValueError, match="relative"):
            threshold(
                self._hand_built(Status.SEPARATION),
                definition=Definition.relative(0.5),
                interval_method="profile",
                level=_LEVEL,
            )


class TestAControlIncompatibleFitIsNeverReportedAsUnreachable:
    """A fixed ``upper=1.0`` over a control with failures is rejected by the fit itself, before any
    target is examined, and a ``Threshold`` refuses ``UNREACHABLE`` on such a fit. So an
    impossible target, given directly or resolved from a ``0/n`` control, leaves the result
    ``CONTROL_INCOMPATIBLE``."""

    @staticmethod
    def _control_incompatible_fit(control_successes: int) -> Fit:
        fit = fit_dose_response(
            _observations([0.0, 0.01, 0.05, 0.1], [control_successes, 0, 0, 0], trials=200),
            model="binomial",
            link="probit",
            upper=1.0,
            lower=0.0,
        )
        assert fit.status is Status.CONTROL_INCOMPATIBLE
        return fit

    @pytest.mark.parametrize("target", _TARGETS)
    def test_an_absolute_target_outside_zero_one_keeps_the_fits_status(self, target: float) -> None:
        fit = self._control_incompatible_fit(98)

        t = threshold(
            fit,
            definition=Definition.absolute(target),
            interval_method="profile",
            level=_LEVEL,
        )

        assert t.status is Status.CONTROL_INCOMPATIBLE
        assert t.censoring is Censoring.NONE
        assert t.value is None and t.lo is None and t.hi is None

    def test_a_zero_control_fraction_keeps_the_fits_status(self) -> None:
        """The target is ``p * 0/200 = 0``, outside ``(0, 1)``, on a control that is the very
        reason the fit was rejected."""
        fit = self._control_incompatible_fit(0)

        t = threshold(
            fit,
            definition=Definition.baseline_fraction(0.5),
            interval_method="profile",
            level=_LEVEL,
        )

        assert t.status is Status.CONTROL_INCOMPATIBLE
        assert t.censoring is Censoring.NONE
        assert t.value is None and t.lo is None and t.hi is None


class TestUnreachableOnAFailedFitSerialises:
    def _unreachable_on_a_failed_fit(self) -> Threshold:
        return threshold(
            _separation(),
            definition=Definition.absolute(1.5),
            interval_method="profile",
            level=_LEVEL,
        )

    def test_it_round_trips_through_json_with_status_definition_level_and_warnings_intact(
        self,
    ) -> None:
        t = self._unreachable_on_a_failed_fit()
        assert t.status is Status.UNREACHABLE and t.fit.status is Status.SEPARATION

        payload = json.loads(json.dumps(to_dict(t), allow_nan=False))
        back = from_dict(payload)

        assert isinstance(back, Threshold)
        assert back == t
        assert back.status is Status.UNREACHABLE
        assert back.fit.status is Status.SEPARATION
        assert back.definition == Definition.absolute(1.5)
        assert back.level == _LEVEL
        assert back.warnings == t.warnings

    def test_the_serialised_form_validates_against_the_packaged_schema(self) -> None:
        card = Scorecard(results=(self._unreachable_on_a_failed_fit(),), provenance={})
        payload = json.loads(json.dumps(to_dict(card)))

        jsonschema.Draft202012Validator(load_schema()).validate(payload)
