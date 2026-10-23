"""``threshold()`` on an increasing-outcome fit, and ``Definition.baseline_multiple``.

An increasing fit is solved on its mirrored decreasing view with the mirrored target, so the
comparisons here are against that view: ``absolute(a)`` against ``absolute(1 - a)`` and
``relative(p)`` against ``relative(p)``, on the fit of the complement counts with the asymptotes
swapped and complemented. Severities are never mirrored, so ``value``, ``lo``, ``hi``,
``censoring``, ``grid`` and ``status`` carry over; the ``Threshold`` reports the original
(unmirrored) ``Fit``, an original-scale ``BaselineRate`` and original-scale text.

``RIGHT`` reads "still passes at the highest level tested", meaning the rate is confidently
*below* the target there, and ``LEFT`` and ``FAILS_AT_BASELINE`` mirror the same way.

``baseline_multiple(k)`` targets ``k * P(0)`` with ``P(0) = lower``:

* an OK fit whose ``lower`` is fixed at 0 raises (a multiple of zero is not a threshold) and the
  message points at ``lower="estimate"``;
* an OK fit with ``lower`` fixed above 0 resolves to ``k * lower``, which is the same target as
  ``absolute(k * lower)``;
* an OK fit with ``lower`` estimated resolves to ``k * lower_hat``, and the fitted curve passes
  through it at the reported value;
* on a failed fit (``SEPARATION``/``NOT_CONVERGED``) it raises whatever the control holds, naming
  the fit status and pointing to ``absolute(a)``;
* on a linear axis it raises, because the lower asymptote is not the performance at zero severity
  there.

Pairing: ``baseline_fraction`` refuses an increasing fit and ``baseline_multiple`` refuses a
decreasing one.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from robustmargin import (
    Axis,
    Censoring,
    Definition,
    Fit,
    Observations,
    Status,
    Threshold,
    fit_dose_response,
    threshold,
)

_LEVEL = 0.95
_LOG_AXIS = Axis(name="severity", unit="unit", scale="log")
_METHODS = ("profile", "delta")

# Events out of 400, rising from about 5% to about 95%, with a control (see
# test_increasing_fit_mirror.py).
_SEVERITY = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
_EVENTS = [20, 22, 35, 90, 220, 340, 380]
_TRIALS = [400] * 7


def _fit(
    severity: list[float],
    events: list[int],
    trials: list[int],
    *,
    upper: float | str,
    lower: float | str,
    direction: str = "increasing",
    axis: Axis = _LOG_AXIS,
    link: str = "probit",
) -> Fit:
    obs = Observations.from_counts(
        axis,
        severity=severity,
        successes=events,
        trials=trials,
        outcome="outcome",
        direction=direction,
    )
    return fit_dose_response(obs, model="binomial", link=link, upper=upper, lower=lower)


def _flip(value: float | str) -> float | str:
    return value if isinstance(value, str) else 1.0 - value


def _mirror_fit(
    severity: list[float],
    events: list[int],
    trials: list[int],
    *,
    upper: float | str,
    lower: float | str,
    axis: Axis = _LOG_AXIS,
    link: str = "probit",
) -> Fit:
    """The decreasing fit on the complement counts with ``(u', l') = (1 - l, 1 - u)``."""
    return _fit(
        severity,
        [n - k for k, n in zip(events, trials, strict=True)],
        trials,
        upper=_flip(lower),
        lower=_flip(upper),
        direction="decreasing",
        axis=axis,
        link=link,
    )


def _assert_same_result(inc: Threshold, dec: Threshold) -> None:
    assert inc.status is dec.status
    assert inc.censoring is dec.censoring
    assert inc.interval_method == dec.interval_method
    assert inc.grid == dec.grid
    for name in ("value", "lo", "hi"):
        a, b = getattr(inc, name), getattr(dec, name)
        if b is None:
            assert a is None, f"{name}: {a!r} against None"
        else:
            assert a == pytest.approx(b, rel=1e-9, abs=1e-12), name


class TestIncreasingEqualsTheMirroredDecreasingThreshold:
    @pytest.mark.parametrize("method", _METHODS)
    @pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
    @pytest.mark.parametrize("upper_lower", [(1.0, "estimate"), ("estimate", "estimate")])
    def test_absolute_and_relative_on_estimated_asymptotes(
        self, method: str, link: str, upper_lower: tuple[float | str, float | str]
    ) -> None:
        upper, lower = upper_lower
        inc = _fit(_SEVERITY, _EVENTS, _TRIALS, upper=upper, lower=lower, link=link)
        dec = _mirror_fit(_SEVERITY, _EVENTS, _TRIALS, upper=upper, lower=lower, link=link)
        assert inc.status is Status.OK and dec.status is Status.OK

        for definition_inc, definition_dec in [
            (Definition.absolute(0.5), Definition.absolute(0.5)),
            (Definition.absolute(0.3), Definition.absolute(1.0 - 0.3)),
            (Definition.relative(0.5), Definition.relative(0.5)),
            (Definition.relative(0.8), Definition.relative(0.8)),
        ]:
            t_inc = threshold(inc, definition=definition_inc, interval_method=method, level=_LEVEL)
            t_dec = threshold(dec, definition=definition_dec, interval_method=method, level=_LEVEL)

            assert t_inc.status is Status.OK
            _assert_same_result(t_inc, t_dec)
            assert t_inc.direction == "increasing"
            assert t_inc.fit is inc
            assert t_inc.definition == definition_inc

    @pytest.mark.parametrize("method", _METHODS)
    def test_the_value_is_where_the_original_scale_curve_crosses_the_target(
        self, method: str
    ) -> None:
        inc = _fit(_SEVERITY, _EVENTS, _TRIALS, upper=1.0, lower="estimate")

        t = threshold(
            inc, definition=Definition.absolute(0.5), interval_method=method, level=_LEVEL
        )

        assert t.value is not None
        assert inc.predict(t.value).estimate == pytest.approx(0.5, abs=1e-9)
        assert t.lo is not None and t.hi is not None and t.lo <= t.value <= t.hi


class TestBaselineMultipleOnAnOkFit:
    def test_a_fixed_zero_lower_raises_and_suggests_estimating_it(self) -> None:
        fit = _fit([0.0, 1.0, 2.0, 4.0, 8.0], [0, 10, 40, 80, 95], [100] * 5, upper=1.0, lower=0.0)
        assert fit.status is Status.OK

        with pytest.raises(ValueError) as excinfo:
            threshold(
                fit,
                definition=Definition.baseline_multiple(2.0),
                interval_method="profile",
                level=_LEVEL,
            )

        message = str(excinfo.value)
        assert "multiple of zero" in message
        assert 'lower="estimate"' in message

    def test_a_fixed_zero_lower_raises_when_there_is_no_control_cell_too(self) -> None:
        fit = _fit([1.0, 2.0, 4.0, 8.0], [10, 40, 80, 95], [100] * 4, upper=1.0, lower=0.0)
        assert fit.status is Status.OK

        with pytest.raises(ValueError, match="multiple of zero"):
            threshold(
                fit,
                definition=Definition.baseline_multiple(2.0),
                interval_method="delta",
                level=_LEVEL,
            )

    @pytest.mark.parametrize("method", _METHODS)
    def test_a_fixed_positive_lower_resolves_to_k_times_lower_exactly_as_absolute_does(
        self, method: str
    ) -> None:
        fit = _fit(_SEVERITY, _EVENTS, _TRIALS, upper=1.0, lower=0.05)
        assert fit.status is Status.OK

        by_multiple = threshold(
            fit,
            definition=Definition.baseline_multiple(2.0),
            interval_method=method,
            level=_LEVEL,
        )
        by_absolute = threshold(
            fit, definition=Definition.absolute(0.1), interval_method=method, level=_LEVEL
        )

        assert by_multiple.status is Status.OK
        assert by_multiple.definition == Definition.baseline_multiple(2.0)
        assert by_multiple.value == pytest.approx(by_absolute.value, rel=1e-12)
        assert by_multiple.lo == pytest.approx(by_absolute.lo, rel=1e-12)
        assert by_multiple.hi == pytest.approx(by_absolute.hi, rel=1e-12)

    @pytest.mark.parametrize("method", _METHODS)
    def test_a_fixed_positive_lower_equals_the_mirrored_decreasing_threshold(
        self, method: str
    ) -> None:
        inc = _fit(_SEVERITY, _EVENTS, _TRIALS, upper=1.0, lower=0.05)
        dec = _mirror_fit(_SEVERITY, _EVENTS, _TRIALS, upper=1.0, lower=0.05)

        t_inc = threshold(
            inc, definition=Definition.baseline_multiple(2.0), interval_method=method, level=_LEVEL
        )
        t_dec = threshold(
            dec,
            definition=Definition.absolute(1.0 - 2.0 * 0.05),
            interval_method=method,
            level=_LEVEL,
        )

        _assert_same_result(t_inc, t_dec)

    @pytest.mark.parametrize("method", _METHODS)
    @pytest.mark.parametrize("upper", [1.0, "estimate"])
    def test_an_estimated_lower_resolves_to_k_times_the_fitted_lower(
        self, method: str, upper: float | str
    ) -> None:
        fit = _fit(_SEVERITY, _EVENTS, _TRIALS, upper=upper, lower="estimate")
        assert fit.status is Status.OK and fit.params is not None
        lower_hat = fit.params["lower"].value

        for k in (2.0, 3.0):
            t = threshold(
                fit,
                definition=Definition.baseline_multiple(k),
                interval_method=method,
                level=_LEVEL,
            )

            assert t.status is Status.OK
            assert t.value is not None
            assert fit.predict(t.value).estimate == pytest.approx(k * lower_hat, rel=1e-6)
            if t.lo is not None and t.hi is not None:
                assert t.lo <= t.value <= t.hi

    @pytest.mark.parametrize("method", _METHODS)
    def test_a_larger_multiple_is_reached_at_a_larger_severity(self, method: str) -> None:
        fit = _fit(_SEVERITY, _EVENTS, _TRIALS, upper=1.0, lower="estimate")

        values: list[float] = []
        for k in (1.5, 2.0, 3.0):
            value = threshold(
                fit,
                definition=Definition.baseline_multiple(k),
                interval_method=method,
                level=_LEVEL,
            ).value
            assert value is not None
            values.append(value)

        assert values[0] < values[1] < values[2]

    def test_a_multiple_whose_target_reaches_the_upper_asymptote_is_unreachable(self) -> None:
        """``k * lower`` above ``upper`` but still below 1 is outside the fitted curve's range, the
        asymptote-driven ``UNREACHABLE`` rule: ``lower`` is fixed at 0.2 and ``upper`` at 0.9, so
        ``k = 4.75`` targets 0.95."""
        fit = _fit(
            [0.0, 1.0, 2.0, 4.0, 8.0, 16.0],
            [80, 120, 200, 300, 350, 360],
            [400] * 6,
            upper=0.9,
            lower=0.2,
        )
        assert fit.status is Status.OK

        t = threshold(
            fit,
            definition=Definition.baseline_multiple(4.75),
            interval_method="profile",
            level=_LEVEL,
        )

        assert t.status is Status.UNREACHABLE
        assert t.value is None and t.lo is None and t.hi is None

    def test_a_target_at_or_above_one_is_unreachable_even_with_a_high_upper(self) -> None:
        fit = _fit(
            [0.0, 1.0, 2.0, 4.0, 8.0, 16.0],
            [120, 160, 220, 300, 350, 380],
            [400] * 6,
            upper=1.0,
            lower=0.3,
        )
        assert fit.status is Status.OK

        t = threshold(
            fit,
            definition=Definition.baseline_multiple(4.0),
            interval_method="profile",
            level=_LEVEL,
        )

        assert t.status is Status.UNREACHABLE


class TestBaselineMultipleOnAFailedFitRaises:
    """A failed fit has no fitted baseline, and the control's observed rate is an estimate of it,
    not the baseline itself. Exact claims read against a multiple of that estimate would be about
    a target nobody asked for, so ``baseline_multiple`` refuses on every failed fit, whatever the
    control holds. The message names the fit status and points to ``absolute(a)``."""

    @staticmethod
    def _separation(
        severity: list[float],
        events: list[int],
        trials: int = 100,
        lower: float | str = 0.0,
    ) -> Fit:
        fit = _fit(severity, events, [trials] * len(severity), upper=1.0, lower=lower)
        assert fit.status is Status.SEPARATION
        return fit

    @staticmethod
    def _not_converged() -> Fit:
        """Events that fall with severity are the wrong sign for an increasing curve."""
        fit = _fit(
            [0.0, 1.0, 2.0, 4.0, 8.0],
            [40, 35, 25, 12, 5],
            [100] * 5,
            upper=1.0,
            lower="estimate",
        )
        assert fit.status is Status.NOT_CONVERGED
        return fit

    @staticmethod
    def _assert_refused(fit: Fit, k: float = 2.0, method: str = "profile") -> None:
        with pytest.raises(ValueError) as excinfo:
            threshold(
                fit,
                definition=Definition.baseline_multiple(k),
                interval_method=method,
                level=_LEVEL,
            )
        message = str(excinfo.value)
        assert fit.status.value in message, message
        assert "absolute" in message, message

    @pytest.mark.parametrize("method", _METHODS)
    def test_not_converged_with_a_control_that_has_events_raises(self, method: str) -> None:
        self._assert_refused(self._not_converged(), method=method)

    def test_separation_with_a_control_that_has_events_raises(self) -> None:
        self._assert_refused(
            self._separation([0.0, 1.0, 2.0, 3.0, 4.0], [30, 0, 0, 100, 100], lower="estimate")
        )

    @pytest.mark.parametrize("k", [1.5, 2.0, 7.0])
    def test_a_control_with_no_events_raises(self, k: float) -> None:
        self._assert_refused(self._separation([0.0, 1.0, 2.0, 3.0, 4.0], [0, 0, 0, 100, 100]), k)

    def test_the_zero_event_control_that_sends_an_estimated_lower_to_its_boundary_raises(
        self,
    ) -> None:
        fit = _fit(
            [0.0, 1.0, 2.0, 4.0, 8.0, 16.0],
            [0, 1, 3, 8, 15, 19],
            [40, 20, 20, 20, 20, 20],
            upper=1.0,
            lower="estimate",
            link="logit",
        )
        assert fit.status is Status.NOT_CONVERGED

        self._assert_refused(fit)

    def test_a_plug_in_multiple_at_or_above_one_raises_rather_than_reporting_unreachable(
        self,
    ) -> None:
        """Control 11/20, then 14/20, 19/20 and 20/20: ``2 * 11/20 = 1.1`` is not below 1."""
        fit = _fit(
            [0.0, 1.0, 2.0, 4.0],
            [11, 14, 19, 20],
            [20, 20, 20, 20],
            upper="estimate",
            lower="estimate",
        )
        assert fit.status is Status.NOT_CONVERGED

        self._assert_refused(fit)

    def test_no_control_cell_raises_value_error(self) -> None:
        fit = self._separation([1.0, 2.0, 3.0, 4.0, 5.0], [0, 0, 100, 100, 100])

        with pytest.raises(ValueError, match="control"):
            threshold(
                fit,
                definition=Definition.baseline_multiple(2.0),
                interval_method="profile",
                level=_LEVEL,
            )

    def test_absolute_still_answers_on_the_same_failed_fits(self) -> None:
        for fit in (
            self._not_converged(),
            self._separation([0.0, 1.0, 2.0, 3.0, 4.0], [0, 0, 0, 100, 100]),
        ):
            t = threshold(
                fit, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
            )

            assert t.status is fit.status


class TestPairingRefusals:
    def test_baseline_fraction_on_an_increasing_fit_raises(self) -> None:
        fit = _fit(_SEVERITY, _EVENTS, _TRIALS, upper=1.0, lower="estimate")

        with pytest.raises(ValueError, match="baseline_fraction"):
            threshold(
                fit,
                definition=Definition.baseline_fraction(0.5),
                interval_method="profile",
                level=_LEVEL,
            )

    def test_baseline_multiple_on_a_decreasing_fit_raises(self) -> None:
        fit = _fit(
            _SEVERITY,
            [n - k for k, n in zip(_EVENTS, _TRIALS, strict=True)],
            _TRIALS,
            upper="estimate",
            lower=0.0,
            direction="decreasing",
        )
        assert fit.status is Status.OK

        with pytest.raises(ValueError, match="baseline_multiple"):
            threshold(
                fit,
                definition=Definition.baseline_multiple(2.0),
                interval_method="profile",
                level=_LEVEL,
            )

    @pytest.mark.parametrize("direction", ["decreasing", "increasing"])
    def test_the_refusal_applies_to_a_failed_fit_too(self, direction: str) -> None:
        fit = _fit(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0, 0, 0, 100, 100] if direction == "increasing" else [100, 100, 100, 0, 0],
            [100] * 5,
            upper=1.0,
            lower=0.0,
            direction=direction,
        )
        assert fit.status is Status.SEPARATION
        wrong = (
            Definition.baseline_fraction(0.5)
            if direction == "increasing"
            else Definition.baseline_multiple(2.0)
        )

        with pytest.raises(ValueError):
            threshold(fit, definition=wrong, interval_method="profile", level=_LEVEL)


class TestTextIsOnTheOriginalScale:
    @pytest.mark.parametrize("target", [0.0, 1.0, 1.2, 1.5, -0.1])
    def test_an_out_of_range_absolute_target_is_unreachable_and_named_as_given(
        self, target: float
    ) -> None:
        fit = _fit(_SEVERITY, _EVENTS, _TRIALS, upper=1.0, lower="estimate")
        assert fit.status is Status.OK

        t = threshold(
            fit, definition=Definition.absolute(target), interval_method="profile", level=_LEVEL
        )

        assert t.status is Status.UNREACHABLE
        assert t.value is None and t.lo is None and t.hi is None
        assert any(
            f"the resolved target {target!r} is outside the open interval (0, 1)" in w
            for w in t.warnings
        ), t.warnings
        mirrored = repr(1.0 - target)
        assert not any(mirrored in w for w in t.warnings if mirrored != repr(target)), t.warnings

    def test_one_point_two_is_named_as_1_2_and_never_as_minus_0_2(self) -> None:
        fit = _fit(_SEVERITY, _EVENTS, _TRIALS, upper=1.0, lower="estimate")

        t = threshold(
            fit, definition=Definition.absolute(1.2), interval_method="profile", level=_LEVEL
        )

        assert t.status is Status.UNREACHABLE
        assert any("1.2" in w for w in t.warnings)
        assert not any(re.search(r"-0\.(2|19999)", w) for w in t.warnings), t.warnings

    def test_no_threshold_warning_on_an_increasing_fit_names_a_mirrored_target(self) -> None:
        """A bracket on separated data, from the exact fallback: every warning that names a target
        names the caller's ``0.8``, not the mirrored ``0.2``."""
        fit = _fit([0.0, 1.0, 2.0, 3.0, 4.0], [0, 0, 0, 100, 100], [100] * 5, upper=1.0, lower=0.0)

        t = threshold(
            fit, definition=Definition.absolute(0.8), interval_method="profile", level=_LEVEL
        )

        mirrored = repr(1.0 - 0.8)
        assert not any(mirrored in w for w in t.warnings), t.warnings
        assert not any("decreasing" in w for w in t.warnings), t.warnings


_POOLED = Path(__file__).resolve().parents[1] / "data" / "pooled_counts.csv"


def _series(name: str) -> tuple[list[float], list[int], list[int]]:
    severity: list[float] = []
    successes: list[int] = []
    trials: list[int] = []
    with _POOLED.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["series"] == name:
                severity.append(float(row["severity"]))
                successes.append(int(row["successes"]))
                trials.append(int(row["trials"]))
    return severity, successes, trials


# series -> (axis scale, the decreasing view's (upper, lower)). The increasing view of the
# complement counts has the asymptotes swapped and complemented.
_POOLED_CASES = {
    "system_a_axis_1": ("log", (1.0, 0.0)),
    "system_a_axis_2": ("linear", (1.0, 0.0)),
    "system_b_axis_1": ("log", ("estimate", 0.0)),
    "system_c_axis_1": ("log", ("estimate", 0.0)),
}


class TestComplementedPooledCounts:
    """The pooled-counts series read as the outcome that rises: events are ``trials - successes``.
    Every status and censoring edge case must come out as the decreasing result on the original
    counts, with the target mirrored."""

    @pytest.mark.parametrize("series", sorted(_POOLED_CASES))
    @pytest.mark.parametrize("a", [0.05, 0.5])
    def test_status_censoring_and_bounds_match_the_decreasing_series(
        self, series: str, a: float
    ) -> None:
        scale, (upper_dec, lower_dec) = _POOLED_CASES[series]
        axis = Axis(name="axis", unit="unit", scale=scale)
        severity, successes, trials = _series(series)
        events = [n - k for k, n in zip(successes, trials, strict=True)]
        dec = _fit(
            severity,
            successes,
            trials,
            upper=upper_dec,
            lower=lower_dec,
            direction="decreasing",
            axis=axis,
        )
        inc = _fit(
            severity,
            events,
            trials,
            upper=_flip(lower_dec),
            lower=_flip(upper_dec),
            direction="increasing",
            axis=axis,
        )
        assert inc.status is dec.status

        t_dec = threshold(
            dec, definition=Definition.absolute(1.0 - a), interval_method="profile", level=_LEVEL
        )
        t_inc = threshold(
            inc, definition=Definition.absolute(a), interval_method="profile", level=_LEVEL
        )

        _assert_same_result(t_inc, t_dec)
        if t_dec.baseline is None:
            assert t_inc.baseline is None
        else:
            assert t_inc.baseline is not None
            b_inc, b_dec = t_inc.baseline, t_dec.baseline
            assert (b_inc.severity, b_inc.trials) == (b_dec.severity, b_dec.trials)
            assert b_inc.successes == b_dec.trials - b_dec.successes
            assert b_inc.rate == pytest.approx(1.0 - b_dec.rate, abs=1e-12)
            assert b_inc.lo == pytest.approx(1.0 - b_dec.hi, abs=1e-12)
            assert b_inc.hi == pytest.approx(1.0 - b_dec.lo, abs=1e-12)
            assert b_inc.level == b_dec.level and b_inc.method == b_dec.method

    def test_the_identifiable_series_has_a_bracketed_threshold(self) -> None:
        severity, successes, trials = _series("system_a_axis_1")
        events = [n - k for k, n in zip(successes, trials, strict=True)]
        fit = _fit(severity, events, trials, upper=1.0, lower=0.0)

        t = threshold(
            fit, definition=Definition.absolute(0.05), interval_method="profile", level=_LEVEL
        )

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.lo is not None and t.value is not None and t.hi is not None
        assert t.lo <= t.value <= t.hi
        assert 0.01 < t.value < 0.1

    def test_a_series_with_no_event_anywhere_is_right_censored_at_the_highest_level(self) -> None:
        severity, successes, trials = _series("system_a_axis_2")
        events = [n - k for k, n in zip(successes, trials, strict=True)]
        axis = Axis(name="axis", unit="unit", scale="linear")
        fit = _fit(severity, events, trials, upper=1.0, lower=0.0, axis=axis)

        t = threshold(
            fit, definition=Definition.absolute(0.05), interval_method="profile", level=_LEVEL
        )

        assert t.censoring is Censoring.RIGHT
        assert t.value is None and t.hi is None
        assert t.lo == 10.0

    def test_a_control_already_above_the_gate_fails_at_baseline_with_an_original_scale_rate(
        self,
    ) -> None:
        severity, successes, trials = _series("system_b_axis_1")
        events = [n - k for k, n in zip(successes, trials, strict=True)]
        fit = _fit(severity, events, trials, upper=1.0, lower="estimate")

        t = threshold(
            fit, definition=Definition.absolute(0.05), interval_method="profile", level=_LEVEL
        )

        assert t.status is Status.FAILS_AT_BASELINE
        assert t.baseline is not None
        assert (t.baseline.successes, t.baseline.trials) == (102, 200)
        assert t.baseline.rate == pytest.approx(0.51)
        assert t.baseline.lo <= t.baseline.rate <= t.baseline.hi
        assert t.value is None and t.lo is None and t.hi is None


class TestBaselineMultipleIsLogAxisOnly:
    """On a linear axis the lower asymptote is not the performance at zero severity, so a multiple
    of it is not "k times the performance at severity 0"."""

    _LINEAR = Axis(name="severity", unit="unit", scale="linear")

    def _ok_fit(self) -> Fit:
        fit = _fit(_SEVERITY, _EVENTS, _TRIALS, upper=1.0, lower=0.05, axis=self._LINEAR)
        assert fit.status is Status.OK
        return fit

    def test_an_ok_fit_raises_naming_the_axis(self) -> None:
        with pytest.raises(ValueError, match="linear"):
            threshold(
                self._ok_fit(),
                definition=Definition.baseline_multiple(2.0),
                interval_method="profile",
                level=_LEVEL,
            )

    def test_a_failed_fit_raises_too(self) -> None:
        fit = _fit(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0, 0, 0, 100, 100],
            [100] * 5,
            upper=1.0,
            lower=0.0,
            axis=self._LINEAR,
        )
        assert fit.status is Status.SEPARATION

        with pytest.raises(ValueError, match="linear"):
            threshold(
                fit,
                definition=Definition.baseline_multiple(2.0),
                interval_method="profile",
                level=_LEVEL,
            )

    def test_absolute_is_still_available_on_a_linear_axis(self) -> None:
        t = threshold(
            self._ok_fit(),
            definition=Definition.absolute(0.5),
            interval_method="profile",
            level=_LEVEL,
        )

        assert t.status is Status.OK


class TestGridBreakPointOnTheOriginalCounts:
    """The exact-bound fallback's grid rule reads the counts as supplied: a level breaks when
    ``events / trials`` is strictly above the target. The comparison is exact, so a level whose
    rate equals the target to the last digit does not break, whatever floating-point error
    ``1 - a`` carries."""

    @staticmethod
    def _grid_value(events: list[int], trials: int, a: float) -> float | None:
        fit = _fit([0.0, 1.0, 2.0, 4.0, 8.0], events, [trials] * 5, upper=1.0, lower=0.0)
        assert fit.status is Status.SEPARATION
        t = threshold(
            fit, definition=Definition.absolute(a), interval_method="profile", level=_LEVEL
        )
        assert t.grid is not None
        return t.grid.value

    def test_a_level_exactly_at_the_target_does_not_break(self) -> None:
        fit = _fit([0.0, 1.0, 2.0, 4.0], [0, 0, 7, 10], [10] * 4, upper=1.0, lower=0.0)
        assert fit.status is Status.SEPARATION

        t = threshold(
            fit, definition=Definition.absolute(0.7), interval_method="profile", level=_LEVEL
        )

        assert t.grid is not None
        assert t.grid.value == 4.0

    @pytest.mark.parametrize(
        ("a", "n", "tied"),
        [
            (0.1, 10, 1),
            (0.2, 10, 2),
            (0.3, 10, 3),
            (0.6, 10, 6),
            (0.7, 10, 7),
            (0.9, 10, 9),
            (0.35, 20, 7),
            (0.55, 20, 11),
            (0.15, 20, 3),
            (0.45, 20, 9),
            (0.7, 100, 70),
            (0.3, 1000, 300),
        ],
    )
    def test_a_tie_does_not_break_and_one_event_more_does(
        self, a: float, n: int, tied: int
    ) -> None:
        at = self._grid_value([0, 0, tied, n, n], n, a)
        above = self._grid_value([0, 0, tied + 1, n, n], n, a)

        assert at == 4.0, f"{tied}/{n} equals {a}: it must not break"
        assert above == 2.0, f"{tied + 1}/{n} exceeds {a}: it must break"
