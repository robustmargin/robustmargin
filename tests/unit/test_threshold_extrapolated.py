"""``EXTRAPOLATED: `` -- a report-only warning for a model value outside the tested range.

When a converged fit's solve path reports a ``value`` that is positive and lies **strictly** below
the smallest nonzero tested severity ``x_min`` or **strictly** above the largest one ``x_max``,
``threshold()`` appends a warning starting with ``"EXTRAPOLATED: "`` (after the interval method's
own warnings). It exists because exact ``RIGHT`` / ``LEFT`` claims are tests at a stricter
confidence than the model's value needs: where such a claim does not fire, a converged fit reports
the model's value instead, and the value may sit well outside what was measured.

What is pinned here:

* **When it fires.** On the solve path only, with ``value > 0``, ``value < x_min`` or
  ``value > x_max`` strictly; for ``value`` only -- an interval endpoint outside the tested range
  does not trigger it; whatever the interval method and whatever its outcome (bounded, open on one
  side, no bounds); including after a ``NON_MONOTONE: `` contradiction; on a linear axis a value in
  ``(0, x_min)`` counts.
* **When it does not.** On ``RIGHT`` / ``LEFT`` / ``FAILS_AT_BASELINE`` / ``UNREACHABLE`` results
  and on failed fits, none of which carry a model value; a value of 0 or below (which is
  ``UNREACHABLE``); a value exactly at ``x_min`` or ``x_max``.
* **It changes nothing else.** The same fit with the tested range moved gives the same value,
  bounds, status, censoring and method; only the warning differs.
* **It travels.** ``Ratio.warnings`` carries both thresholds' warnings, so it reaches a ratio.

The fits here are ``fake_fit`` OK fits (probit, ``mu = 0.5``, ``s = 0.3``, asymptotes 1 and 0,
``Var(mu) = 0.01``, ``Var(s) = 0.02``) with chosen cells and the delta interval, so the model value
is a closed form (``exp(0.5)`` for a target of 0.5 on a log axis) and can be placed exactly on,
just inside, or just outside the tested range. Only the severities of the cells move; their counts
are the same in every test so the exact tests classify identically.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import re
from collections.abc import Sequence

import pytest

from robustmargin import (
    Axis,
    Cell,
    Censoring,
    Definition,
    Fit,
    Observations,
    Parameter,
    Ratio,
    Status,
    Threshold,
    fit_dose_response,
    ratio_interval,
    threshold,
)
from robustmargin.intervals import IntervalResult, delta_interval, solve_value
from robustmargin.testing import fake_fit

_LEVEL = 0.95
_TOKEN = "EXTRAPOLATED: "
_LOG = Axis(name="severity", unit="unit", scale="log")
_LINEAR = Axis(name="severity", unit="unit", scale="linear")

# The model value for a target of 0.5 on this fit: z* = 0, so theta_hat = mu = 0.5.
_VALUE_AT_HALF = math.exp(0.5)

# Counts that classify as "solve" at target 0.5, level 0.95: the control passes, the other two
# levels neither pass nor fail at the confidence the exact claims use, so none of RIGHT, LEFT or
# FAILS_AT_BASELINE fires, and no pass sits above a fail.
_SOLVE_COUNTS = ((100, 100), (60, 100), (55, 100))
_ALL_PASS = ((100, 100), (100, 100), (100, 100))
_ALL_FAIL_AFTER_A_PASSING_CONTROL = ((100, 100), (0, 100), (0, 100))


def _fit(
    severities: Sequence[float],
    counts: Sequence[tuple[int, int]] = _SOLVE_COUNTS,
    *,
    axis: Axis = _LOG,
) -> Fit:
    cells = tuple(
        Cell(severity=s, successes=k, trials=n)
        for s, (k, n) in zip(severities, counts, strict=True)
    )
    return dataclasses.replace(
        fake_fit(diagnostics=None, monotonicity=None), axis=axis, cells=cells
    )


def _threshold(
    fit: Fit, *, target: float = 0.5, method: str = "delta", kind: str = "absolute"
) -> Threshold:
    definition = (
        Definition.absolute(target) if kind == "absolute" else Definition.baseline_fraction(target)
    )
    return threshold(fit, definition=definition, interval_method=method, level=_LEVEL)


def _tokens(t: Threshold) -> list[str]:
    return [w for w in t.warnings if w.startswith(_TOKEN)]


def _numbers_in(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text)]


class TestItFiresWhenTheValueIsOutsideTheNonzeroTestedRange:
    def test_a_value_below_the_lowest_nonzero_severity_is_flagged(self) -> None:
        t = _threshold(_fit([0.0, 2.0, 5.0]))

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.value == pytest.approx(_VALUE_AT_HALF)
        assert len(_tokens(t)) == 1

    def test_a_value_above_the_highest_severity_is_flagged(self) -> None:
        t = _threshold(_fit([0.0, 0.5, 1.0]))

        assert t.status is Status.OK
        assert t.value == pytest.approx(_VALUE_AT_HALF)
        assert len(_tokens(t)) == 1

    def test_the_message_names_the_value_the_side_and_the_crossed_endpoint(self) -> None:
        below = _tokens(_threshold(_fit([0.0, 2.0, 5.0])))[0]
        above = _tokens(_threshold(_fit([0.0, 0.5, 1.0])))[0]

        for message, endpoint, words in (
            (below, 2.0, ("below", "lower", "under", "less", "smaller")),
            (above, 1.0, ("above", "higher", "upper", "greater", "larger", "beyond", "exceed")),
        ):
            numbers = _numbers_in(message)
            assert any(math.isclose(n, _VALUE_AT_HALF, rel_tol=1e-3) for n in numbers), message
            assert any(math.isclose(n, endpoint, rel_tol=1e-9) for n in numbers), message
            assert any(w in message.lower() for w in words), message

    def test_the_lowest_severity_that_counts_is_the_lowest_nonzero_one_not_the_control(
        self,
    ) -> None:
        """The control sits at severity 0, below every value; it is not the bottom of the tested
        range."""
        t = _threshold(_fit([0.0, 0.5, 5.0]))

        assert not _tokens(t), "the value lies between the control and the highest level only"
        assert t.value is not None and 0.5 < t.value < 5.0

    def test_it_fires_without_a_control(self) -> None:
        cells = ((60, 100), (55, 100))
        t = _threshold(_fit([2.0, 5.0], cells))

        assert len(_tokens(t)) == 1

    def test_on_a_linear_axis_a_value_between_zero_and_the_lowest_severity_counts(self) -> None:
        """Value 0.5, lowest nonzero severity 2.0: positive, so it is a model value below the
        tested range, not an origin crossing."""
        t = _threshold(_fit([0.0, 2.0, 5.0], axis=_LINEAR))

        assert t.status is Status.OK
        assert t.value == pytest.approx(0.5)
        assert len(_tokens(t)) == 1

    def test_it_fires_after_a_contradiction_of_monotonicity(self) -> None:
        """Cells at target 0.96: a level that passes (100/100, severity 2) above one that fails
        (20/100, severity 1) is a ``NON_MONOTONE: `` contradiction, which suppresses the exact
        claims and leaves the solve path. The model value (about 0.975) is below the lowest
        nonzero severity (1), so both warnings are present."""
        counts = ((100, 100), (20, 100), (100, 100), (5, 100))
        t = _threshold(_fit([0.0, 1.0, 2.0, 3.0], counts), target=0.96)

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert any(w.startswith("NON_MONOTONE: ") for w in t.warnings)
        assert t.value is not None and 0.0 < t.value < 1.0
        assert len(_tokens(t)) == 1


class TestItDoesNotFireWhenTheValueIsInsideOrOnTheEdge:
    def test_a_value_between_the_lowest_and_highest_nonzero_severity_is_not_flagged(self) -> None:
        t = _threshold(_fit([0.0, 1.0, 5.0]))

        assert t.value == pytest.approx(_VALUE_AT_HALF)
        assert not _tokens(t)

    def test_a_value_exactly_at_the_lowest_nonzero_severity_is_not_flagged(self) -> None:
        t = _threshold(_fit([0.0, _VALUE_AT_HALF, 5.0]))

        assert t.value == _VALUE_AT_HALF
        assert not _tokens(t), "the comparison is strict"

    def test_a_value_exactly_at_the_highest_severity_is_not_flagged(self) -> None:
        t = _threshold(_fit([0.0, 0.5, _VALUE_AT_HALF]))

        assert t.value == _VALUE_AT_HALF
        assert not _tokens(t), "the comparison is strict"

    def test_a_value_one_float_step_outside_either_edge_is_flagged(self) -> None:
        just_below_lowest = _threshold(_fit([0.0, math.nextafter(_VALUE_AT_HALF, math.inf), 5.0]))
        just_above_highest = _threshold(_fit([0.0, 0.5, math.nextafter(_VALUE_AT_HALF, 0.0)]))

        assert len(_tokens(just_below_lowest)) == 1
        assert len(_tokens(just_above_highest)) == 1

    def test_an_interval_endpoint_outside_the_range_does_not_flag_a_value_inside_it(self) -> None:
        """The delta interval around 1.6487 is about ``[1.355, 2.005]``; the tested range is
        ``[1.5, 1.8]``. Both endpoints are outside it and the value is inside."""
        t = _threshold(_fit([0.0, 1.5, 1.8]))

        assert t.lo is not None and t.hi is not None
        assert t.lo < 1.5 and t.hi > 1.8
        assert t.value is not None and 1.5 < t.value < 1.8
        assert not _tokens(t)


class TestItIsAnnotationOnlyOnTheSolvePath:
    @pytest.mark.parametrize("counts", [_ALL_PASS], ids=["right"])
    def test_a_right_result_has_no_model_value_and_no_token(
        self, counts: Sequence[tuple[int, int]]
    ) -> None:
        t = _threshold(_fit([0.0, 2.0, 5.0], counts))

        assert t.censoring is Censoring.RIGHT and t.value is None
        assert not _tokens(t)

    def test_a_left_result_has_no_model_value_and_no_token(self) -> None:
        t = _threshold(_fit([0.0, 2.0, 5.0], _ALL_FAIL_AFTER_A_PASSING_CONTROL))

        assert t.censoring is Censoring.LEFT and t.value is None
        assert not _tokens(t)

    def test_a_fails_at_baseline_result_has_no_token(self) -> None:
        counts = ((0, 100), (0, 100), (0, 100))
        t = _threshold(_fit([0.0, 2.0, 5.0], counts, axis=_LINEAR))

        assert t.status is Status.FAILS_AT_BASELINE
        assert not _tokens(t)

    def test_a_target_outside_zero_one_has_no_token(self) -> None:
        t = _threshold(_fit([0.0, 2.0, 5.0]), target=1.5)

        assert t.status is Status.UNREACHABLE
        assert not _tokens(t)

    def test_a_target_the_fitted_asymptotes_cannot_reach_has_no_token(self) -> None:
        fit = _fit([0.0, 2.0, 5.0])
        assert fit.params is not None and fit.covariance is not None
        capped = dataclasses.replace(
            fit, params={**fit.params, "upper": Parameter(value=0.9, fixed=True)}
        )

        t = _threshold(capped, target=0.95)

        assert t.status is Status.UNREACHABLE
        assert not _tokens(t)

    def test_a_crossing_at_or_below_zero_is_unreachable_and_has_no_token(self) -> None:
        """On a linear axis a target of 0.99 solves to ``0.5 + 0.3 * z*`` with ``z* = -2.33``,
        about -0.2: below the origin, so ``UNREACHABLE``. That value is below the lowest nonzero
        severity but not positive, and nothing is reported."""
        counts = ((100, 100), (90, 100), (50, 100))
        t = _threshold(_fit([0.0, 1.0, 2.0], counts, axis=_LINEAR), target=0.99)

        assert t.status is Status.UNREACHABLE
        assert t.value is None
        assert not _tokens(t)

    @pytest.mark.parametrize("status", [Status.SEPARATION, Status.NOT_CONVERGED])
    def test_a_failed_fit_has_no_token(self, status: Status) -> None:
        cells = (
            Cell(severity=0.0, successes=100, trials=100),
            Cell(severity=2.0, successes=60, trials=100),
            Cell(severity=5.0, successes=55, trials=100),
        )
        fit = dataclasses.replace(fake_fit(status=status, monotonicity=None), cells=cells)

        t = _threshold(fit)

        assert t.status is status and t.value is None
        assert not _tokens(t)


class TestItDoesNotDependOnTheIntervalOutcome:
    """The token describes the model value, not the interval. Whatever the profile search
    returns -- bounded, open on either side, or no bounds at all -- a value outside the tested
    range is flagged. The search result is supplied directly here, so every outcome is exercised
    on the same fit."""

    _VALUE = _VALUE_AT_HALF

    @pytest.mark.parametrize(
        ("interval", "censoring"),
        [
            (IntervalResult(lo=1.0, hi=2.5, censoring=Censoring.NONE), Censoring.NONE),
            (IntervalResult(lo=None, hi=None, censoring=Censoring.NONE), Censoring.NONE),
            (IntervalResult(lo=1.0, hi=None, censoring=Censoring.OPEN_UPPER), Censoring.OPEN_UPPER),
            (IntervalResult(lo=None, hi=2.5, censoring=Censoring.OPEN_LOWER), Censoring.OPEN_LOWER),
        ],
        ids=["bounded", "no_bounds", "open_upper", "open_lower"],
    )
    @pytest.mark.parametrize(
        "severities", [[0.0, 2.0, 5.0], [0.0, 0.5, 1.0]], ids=["below", "above"]
    )
    def test_every_interval_outcome_still_carries_the_token(
        self,
        monkeypatch: pytest.MonkeyPatch,
        interval: IntervalResult,
        censoring: Censoring,
        severities: list[float],
    ) -> None:
        # `robustmargin.threshold` the attribute is the function; the module is in sys.modules.
        module = importlib.import_module("robustmargin.threshold")
        monkeypatch.setattr(module, "profile_interval", lambda *args, **kwargs: interval)

        t = _threshold(_fit(severities), method="profile")

        assert t.value == pytest.approx(self._VALUE)
        assert t.censoring is censoring
        assert (t.lo, t.hi) == (interval.lo, interval.hi)
        assert len(_tokens(t)) == 1

    def test_interval_warnings_come_before_the_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = importlib.import_module("robustmargin.threshold")
        interval = IntervalResult(
            lo=None, hi=None, censoring=Censoring.NONE, warnings=("intervals.probe: a warning",)
        )
        monkeypatch.setattr(module, "profile_interval", lambda *args, **kwargs: interval)

        t = _threshold(_fit([0.0, 2.0, 5.0]), method="profile")

        assert t.warnings[-2:] == ("intervals.probe: a warning", _tokens(t)[0])


class TestItChangesNoNumber:
    def test_moving_the_tested_range_changes_only_the_warning(self) -> None:
        inside = _threshold(_fit([0.0, 1.0, 5.0]))
        below = _threshold(_fit([0.0, 2.0, 5.0]))
        above = _threshold(_fit([0.0, 0.5, 1.0]))

        assert not _tokens(inside) and _tokens(below) and _tokens(above)
        for flagged in (below, above):
            assert (flagged.value, flagged.lo, flagged.hi) == (inside.value, inside.lo, inside.hi)
            assert flagged.status is inside.status
            assert flagged.censoring is inside.censoring
            assert flagged.interval_method == inside.interval_method
            assert flagged.warnings[:-1] == inside.warnings
            assert flagged.warnings[-1].startswith(_TOKEN)

    def test_the_numbers_are_the_closed_form_solve_and_delta_interval_of_the_fit(self) -> None:
        fit = _fit([0.0, 2.0, 5.0])
        theta_hat, value = solve_value(fit, target=0.5)
        expected = delta_interval(
            fit,
            definition=Definition.absolute(0.5),
            target=0.5,
            level=_LEVEL,
            theta_hat=theta_hat,
            value=value,
        )

        t = _threshold(fit)

        assert _tokens(t)
        assert (t.value, t.lo, t.hi) == (value, expected.lo, expected.hi)

    @pytest.mark.parametrize("method", ["profile", "delta"])
    def test_a_real_fit_reports_the_same_numbers_with_or_without_the_token(
        self, method: str
    ) -> None:
        """A converged fit on real data (the nearly flat curve of the profile-search test): the
        value is far above the tested range and is reported unchanged, with the token."""
        obs = Observations.from_counts(
            _LOG,
            severity=[0.0, 1.0, 10.0, 100.0],
            successes=[100, 90, 89, 88],
            trials=[100, 100, 100, 100],
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        assert fit.status is Status.OK
        theta_hat, value = solve_value(fit, target=0.8)

        t = _threshold(fit, target=0.8, method=method)

        assert t.value == value
        assert value > 100.0
        assert len(_tokens(t)) == 1


class TestItIsLastAndAppearsOnce:
    def test_it_comes_after_the_interval_methods_own_warnings(self) -> None:
        """The profile search for the nearly flat curve warns about the sides it cannot close; the
        token is appended after those."""
        obs = Observations.from_counts(
            _LOG,
            severity=[0.0, 1.0, 10.0, 100.0],
            successes=[100, 90, 89, 88],
            trials=[100, 100, 100, 100],
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        t = _threshold(fit, target=0.8, method="profile")

        interval_warnings = [i for i, w in enumerate(t.warnings) if w.startswith("intervals.")]
        assert interval_warnings, "the premise: the profile search warned"
        assert len(_tokens(t)) == 1
        assert t.warnings[-1].startswith(_TOKEN)
        assert max(interval_warnings) < len(t.warnings) - 1

    def test_it_is_a_plain_string_prefix_for_consumers_to_match(self) -> None:
        t = _threshold(_fit([0.0, 2.0, 5.0]))

        assert _tokens(t)[0].startswith("EXTRAPOLATED: ")


class TestItReachesARatio:
    @staticmethod
    def _pair(
        a_severities: Sequence[float], b_severities: Sequence[float]
    ) -> tuple[Threshold, Threshold]:
        return (
            _threshold(_fit(a_severities)),
            _threshold(_fit(b_severities)),
        )

    def test_both_inputs_warnings_are_carried_so_each_token_reaches_the_ratio(self) -> None:
        t_a, t_b = self._pair([0.0, 2.0, 5.0], [0.0, 0.5, 1.0])
        assert _tokens(t_a) and _tokens(t_b)

        ratio = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert isinstance(ratio, Ratio)
        carried = [w for w in ratio.warnings if w.startswith(_TOKEN)]
        assert sorted(carried) == sorted([*_tokens(t_a), *_tokens(t_b)])

    def test_only_the_extrapolated_input_contributes_a_token(self) -> None:
        t_a, t_b = self._pair([0.0, 2.0, 5.0], [0.0, 1.0, 5.0])
        assert _tokens(t_a) and not _tokens(t_b)

        ratio = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        carried = [w for w in ratio.warnings if w.startswith(_TOKEN)]
        assert carried == _tokens(t_a)

    def test_the_ratios_numbers_do_not_depend_on_the_token(self) -> None:
        flagged_a, flagged_b = self._pair([0.0, 2.0, 5.0], [0.0, 0.5, 1.0])
        plain_a, plain_b = self._pair([0.0, 1.0, 5.0], [0.0, 1.0, 5.0])

        flagged = ratio_interval(flagged_a, flagged_b, method="fieller", dependence="independent")
        plain = ratio_interval(plain_a, plain_b, method="fieller", dependence="independent")

        assert (flagged.estimate, flagged.lo, flagged.hi) == (plain.estimate, plain.lo, plain.hi)
        assert flagged.status is plain.status and flagged.censoring is plain.censoring
