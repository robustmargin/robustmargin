"""The profile-likelihood search when the estimate lies at or beyond a search limit.

The profile search brackets outward from ``theta_hat`` toward a limit on each side (``lower_limit``
below it, ``upper_limit`` above it) and looks for the sign change of
``g(theta) = 2 * (llf_hat - llf_profile(theta)) - chi2_target`` (``g(theta_hat) < 0`` by
construction). The limits are fixed multiples of the *tested* range, so a converged fit whose
crossing lies far outside that range has ``theta_hat`` beyond one of them.

A side whose limit is at or on the wrong side of ``theta_hat`` has nothing to search: stepping
"outward" toward it would walk back across ``theta_hat`` and find the *other* side's root. That
side is therefore open, ``g`` is not evaluated for it, and the warning names both ``theta_hat`` and
the limit. Every root the search reports then lies on its own side of ``theta_hat``, so
``lo < value < hi`` for whichever sides are reported.

The unit tests drive ``_solve_profile_bracket`` directly with a closed-form ``g`` whose roots are
known: ``g(theta) = (theta - 10)**2 - 1`` has roots 9 and 11 around ``theta_hat = 10``.

The end-to-end tests are real fits through ``threshold()`` whose crossing is far outside the tested
range. In two of them the *whole* profile interval lies beyond a search limit, so a search that
walked toward the wrong-side limit would meet a sign change and report the other endpoint as this
side's bound; each checks the bound it does report against a profile likelihood computed here with
``scipy`` alone. In the third only the estimate is beyond the limit, and the interval's lower
endpoint is not reached by the search (a limitation, recorded as an expected failure).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, norm

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
from robustmargin.intervals import IntervalResult, _solve_profile_bracket

_THETA_HAT = 10.0


def _quadratic(theta: float) -> float:
    return (theta - _THETA_HAT) ** 2 - 1.0


def _recording(calls: list[float]) -> Callable[[float], float]:
    def g(theta: float) -> float:
        calls.append(theta)
        return _quadratic(theta)

    return g


def _search(lower_limit: float, upper_limit: float, *, scale: str = "linear") -> IntervalResult:
    return _solve_profile_bracket(
        _quadratic,
        theta_hat=_THETA_HAT,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        scale=scale,
    )


def _numbers_in(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text)]


class TestAnEstimateBeyondTheUpperLimit:
    def test_the_upper_side_is_open_and_the_lower_root_is_kept(self) -> None:
        """Limits ``(-5, 8)`` put ``theta_hat = 10`` above the upper limit. Searching "upward" would
        walk to 8, below the estimate, where ``g`` is positive, and report the lower root (9) as the
        upper bound too: ``lo = hi = 9``."""
        result = _search(-5.0, 8.0)

        assert result.hi is None
        assert result.lo == pytest.approx(9.0, abs=1e-8)
        assert result.censoring is Censoring.OPEN_UPPER

    def test_g_is_never_evaluated_at_the_upper_limit_that_lies_below_the_estimate(self) -> None:
        evaluated: list[float] = []

        _solve_profile_bracket(
            _recording(evaluated),
            theta_hat=_THETA_HAT,
            lower_limit=-5.0,
            upper_limit=8.0,
            scale="linear",
        )

        assert evaluated, "the lower side must still be searched"
        assert 8.0 not in evaluated, "g was evaluated at the upper limit, below the estimate"
        assert all(t <= _THETA_HAT for t in evaluated), (
            "the upper side is open, so nothing above the estimate is searched"
        )

    def test_the_warning_names_the_estimate_and_the_limit(self) -> None:
        result = _search(-5.0, 8.0)

        assert any(
            10.0 in _numbers_in(w) and 8.0 in _numbers_in(w) and "upper" in w
            for w in result.warnings
        ), result.warnings

    def test_an_estimate_exactly_at_the_upper_limit_leaves_the_upper_side_open(self) -> None:
        result = _search(-5.0, _THETA_HAT)

        assert result.hi is None
        assert result.lo == pytest.approx(9.0, abs=1e-8)
        assert result.censoring is Censoring.OPEN_UPPER


class TestAnEstimateBeyondTheLowerLimit:
    def test_the_lower_side_is_open_and_the_upper_root_is_kept(self) -> None:
        """The mirror: limits ``(12, 20)`` put ``theta_hat = 10`` below the lower limit."""
        result = _search(12.0, 20.0)

        assert result.lo is None
        assert result.hi == pytest.approx(11.0, abs=1e-8)
        assert result.censoring is Censoring.OPEN_LOWER

    def test_g_is_never_evaluated_at_the_lower_limit_that_lies_above_the_estimate(self) -> None:
        evaluated: list[float] = []

        _solve_profile_bracket(
            _recording(evaluated),
            theta_hat=_THETA_HAT,
            lower_limit=12.0,
            upper_limit=20.0,
            scale="linear",
        )

        assert evaluated, "the upper side must still be searched"
        assert 12.0 not in evaluated, "g was evaluated at the lower limit, above the estimate"
        assert all(t >= _THETA_HAT for t in evaluated), (
            "the lower side is open, so nothing below the estimate is searched"
        )

    def test_the_warning_names_the_estimate_and_the_limit(self) -> None:
        result = _search(12.0, 20.0)

        assert any(
            10.0 in _numbers_in(w) and 12.0 in _numbers_in(w) and "lower" in w
            for w in result.warnings
        ), result.warnings

    def test_an_estimate_exactly_at_the_lower_limit_leaves_the_lower_side_open(self) -> None:
        result = _search(_THETA_HAT, 20.0)

        assert result.lo is None
        assert result.hi == pytest.approx(11.0, abs=1e-8)
        assert result.censoring is Censoring.OPEN_LOWER


class TestAnEstimateInsideBothLimitsIsUnchanged:
    def test_both_roots_are_found_on_their_own_sides(self) -> None:
        result = _search(-50.0, 50.0)

        assert result.lo == pytest.approx(9.0, abs=1e-8)
        assert result.hi == pytest.approx(11.0, abs=1e-8)
        assert result.censoring is Censoring.NONE
        assert result.warnings == ()

    def test_on_a_log_axis_the_roots_are_mapped_back_to_the_severity_scale(self) -> None:
        result = _search(-50.0, 50.0, scale="log")

        assert result.lo == pytest.approx(math.exp(9.0), rel=1e-8)
        assert result.hi == pytest.approx(math.exp(11.0), rel=1e-8)

    def test_a_limit_closer_than_the_root_leaves_that_side_open_with_the_limit_named(self) -> None:
        """The limit is above the estimate but below the root (11): the profile has not crossed
        the target by the limit, so that side is open -- a different outcome from the estimate
        being beyond the limit, and it keeps its own warning."""
        result = _search(-50.0, 10.5)

        assert result.hi is None
        assert result.lo == pytest.approx(9.0, abs=1e-8)
        assert any("search limit" in w for w in result.warnings)


@given(
    theta_hat=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False),
    below=st.floats(min_value=-30.0, max_value=30.0, allow_nan=False),
    above=st.floats(min_value=-30.0, max_value=30.0, allow_nan=False),
    curvature=st.floats(min_value=0.05, max_value=20.0, allow_nan=False),
)
# Both limits on the wrong side of the estimate, one on the wrong side, and the straddling case.
@example(theta_hat=10.0, below=-15.0, above=-2.0, curvature=1.0)
@example(theta_hat=10.0, below=2.0, above=10.0, curvature=1.0)
@example(theta_hat=0.0, below=-5.0, above=5.0, curvature=1.0)
@settings(max_examples=200, deadline=None)
def test_every_reported_root_lies_on_its_own_side_of_the_estimate(
    theta_hat: float, below: float, above: float, curvature: float
) -> None:
    """Whatever the limits, ``lo < theta_hat < hi`` for the sides that are reported, a side whose
    limit is at or beyond the estimate is reported open, and the roots that are found are the
    true ones."""
    lower_limit, upper_limit = sorted((theta_hat + below, theta_hat + above))
    if not lower_limit < upper_limit:
        return
    root_distance = 1.0 / math.sqrt(curvature)

    def g(theta: float) -> float:
        return curvature * (theta - theta_hat) ** 2 - 1.0

    result = _solve_profile_bracket(
        g,
        theta_hat=theta_hat,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        scale="linear",
    )

    if result.lo is not None:
        assert result.lo < theta_hat
        assert result.lo == pytest.approx(theta_hat - root_distance, abs=1e-6)
    if result.hi is not None:
        assert result.hi > theta_hat
        assert result.hi == pytest.approx(theta_hat + root_distance, abs=1e-6)
    if upper_limit <= theta_hat:
        assert result.hi is None
    if lower_limit >= theta_hat:
        assert result.lo is None
    # A root that lies comfortably inside its limit is found.
    if theta_hat < upper_limit and theta_hat + root_distance < upper_limit - 1e-3:
        assert result.hi is not None
    if lower_limit < theta_hat and theta_hat - root_distance > lower_limit + 1e-3:
        assert result.lo is not None


_LEVEL = 0.95
_LOG_AXIS = Axis(name="severity", unit="unit", scale="log")


def _fit(severity: list[float], successes: list[int], trials: list[int]) -> Fit:
    """A converged probit fit with both asymptotes fixed (upper 1, lower 0) on a log axis."""
    obs = Observations.from_counts(
        _LOG_AXIS,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
    assert fit.status is Status.OK
    return fit


def _deviance_excess(fit: Fit, *, target: float, severity: float) -> float:
    """``2 * (llf_hat - llf_profile(theta)) - chi2.ppf(level, 1)`` at ``theta = log(severity)``,
    from the binomial-probit likelihood written out here, independently of ``intervals``.

    The curve is ``P(x) = 1 - Phi((log x - mu) / s)``. Requiring it to cross ``target`` at
    ``theta`` fixes ``mu = theta - s * z*`` with ``z* = Phi^-1(1 - target)``, so the profile
    likelihood at ``theta`` is the likelihood maximised over ``s > 0`` alone. A profile endpoint is
    where this is zero. The control (severity 0) is all-success, so with the upper asymptote at 1 it
    contributes ``log(1) = 0`` and is left out.
    """
    assert fit.params is not None
    control, *tested = fit.cells
    assert control.severity == 0.0 and control.successes == control.trials
    log_x = np.log([c.severity for c in tested])
    k = np.array([c.successes for c in tested], dtype=float)
    n = np.array([c.trials for c in tested], dtype=float)
    z_star = float(norm.ppf(1.0 - target))

    def log_likelihood(mu: float, s: float) -> float:
        z = (log_x - mu) / s
        return float(np.sum(k * norm.logsf(z) + (n - k) * norm.logcdf(z)))

    llf_hat = log_likelihood(fit.params["mu"].value, fit.params["s"].value)
    theta = math.log(severity)
    profile = minimize_scalar(
        lambda log_s: -log_likelihood(theta - math.exp(log_s) * z_star, math.exp(log_s)),
        bounds=(-10.0, 15.0),
        method="bounded",
        options={"xatol": 1.0e-10},
    )
    return 2.0 * (llf_hat + float(profile.fun)) - float(chi2.ppf(_LEVEL, 1))


class TestAnIntervalWhollyBelowTheTestedRange:
    """Severities 1, 10, 100 and 1000 with 20 trials each and 7, 5, 3 and 1 successes, a 20/20
    control, against ``absolute(0.95)``. Success falls from 0.35 to 0.05 over the tested range, so
    the curve crosses 0.95 far below severity 1: the estimate is near 9e-6, and even the interval's
    upper endpoint (near 0.0062) lies below the lowest search limit, severity 0.01 (a hundredth of
    the lowest tested severity).

    With the limit above the estimate there is nothing to search below it. A search that stepped
    "downward" toward it would walk back up across the estimate and find the upper endpoint, then
    report it as the lower bound as well: ``lo == hi > value``, which no ``Threshold`` can hold.
    """

    _SEVERITY = [0.0, 1.0, 10.0, 100.0, 1000.0]
    _SUCCESSES = [20, 7, 5, 3, 1]
    _TRIALS = [20, 20, 20, 20, 20]
    _TARGET = 0.95
    _LOWEST_SEARCH_LIMIT = 1.0 / 100.0

    def _result(self) -> tuple[Fit, Threshold]:
        fit = _fit(self._SEVERITY, self._SUCCESSES, self._TRIALS)
        t = threshold(
            fit,
            definition=Definition.absolute(self._TARGET),
            interval_method="profile",
            level=_LEVEL,
        )
        return fit, t

    def test_the_design_puts_the_whole_interval_below_the_lowest_search_limit(self) -> None:
        """Not vacuous: this is the situation the section above describes, and it is not the one
        where the walk to the limit simply finds no sign change."""
        _, t = self._result()

        assert t.value is not None and t.hi is not None
        assert t.value < t.hi < self._LOWEST_SEARCH_LIMIT

    def test_the_upper_bound_is_reported_and_the_lower_side_is_open(self) -> None:
        _, t = self._result()

        assert t.status is Status.OK
        assert t.censoring is Censoring.OPEN_LOWER
        assert t.lo is None
        assert t.value is not None and t.hi is not None
        assert t.value < t.hi

    def test_the_upper_bound_is_the_profile_likelihood_endpoint(self) -> None:
        """The deviance at ``hi`` is the chi-square critical value, and at the estimate it is
        zero, so ``hi`` is the endpoint on the upper side of ``value`` and not some other root."""
        fit, t = self._result()

        assert t.value is not None and t.hi is not None
        excess_at_hi = _deviance_excess(fit, target=self._TARGET, severity=t.hi)
        excess_at_value = _deviance_excess(fit, target=self._TARGET, severity=t.value)
        assert excess_at_hi == pytest.approx(0.0, abs=1e-6)
        assert excess_at_value == pytest.approx(-float(chi2.ppf(_LEVEL, 1)), abs=1e-6)

    def test_the_lower_side_warning_names_the_estimate_and_the_search_limit(self) -> None:
        _, t = self._result()
        assert t.value is not None

        side_warnings = [w for w in t.warnings if "nothing to search" in w]
        assert len(side_warnings) == 1 and "lower" in side_warnings[0], t.warnings
        numbers = _numbers_in(side_warnings[0])
        assert any(abs(x - math.log(t.value)) < 1e-6 for x in numbers), side_warnings
        assert any(abs(x - math.log(self._LOWEST_SEARCH_LIMIT)) < 1e-6 for x in numbers)


class TestAnIntervalWhollyAboveTheTestedRange:
    """The mirror. Severities 1, 10 and 100 with 1000, 1000 and 10 trials and 500, 480 and 5
    successes, a 1000/1000 control, against ``absolute(0.2)``. The two large cells are nearly flat
    near one half and the 10-trial top cell is too small to settle anything, so the curve crosses
    0.2 far above severity 100: the estimate is near 6e17, and even the interval's lower endpoint
    (near 5.8e5) lies above the highest search limit, severity 1e4 (a hundred times the highest
    tested severity).

    A search that stepped "upward" toward a limit below the estimate would find the lower endpoint
    and report it as the upper bound too: ``lo == hi < value``.
    """

    _SEVERITY = [0.0, 1.0, 10.0, 100.0]
    _SUCCESSES = [1000, 500, 480, 5]
    _TRIALS = [1000, 1000, 1000, 10]
    _TARGET = 0.2
    _HIGHEST_SEARCH_LIMIT = 100.0 * 100.0

    def _result(self) -> tuple[Fit, Threshold]:
        fit = _fit(self._SEVERITY, self._SUCCESSES, self._TRIALS)
        t = threshold(
            fit,
            definition=Definition.absolute(self._TARGET),
            interval_method="profile",
            level=_LEVEL,
        )
        return fit, t

    def test_the_design_puts_the_whole_interval_above_the_highest_search_limit(self) -> None:
        _, t = self._result()

        assert t.value is not None and t.lo is not None
        assert self._HIGHEST_SEARCH_LIMIT < t.lo < t.value

    def test_the_lower_bound_is_reported_and_the_upper_side_is_open(self) -> None:
        _, t = self._result()

        assert t.status is Status.OK
        assert t.censoring is Censoring.OPEN_UPPER
        assert t.hi is None
        assert t.value is not None and t.lo is not None
        assert t.lo < t.value

    def test_the_lower_bound_is_the_profile_likelihood_endpoint(self) -> None:
        fit, t = self._result()

        assert t.value is not None and t.lo is not None
        excess_at_lo = _deviance_excess(fit, target=self._TARGET, severity=t.lo)
        excess_at_value = _deviance_excess(fit, target=self._TARGET, severity=t.value)
        assert excess_at_lo == pytest.approx(0.0, abs=1e-6)
        assert excess_at_value == pytest.approx(-float(chi2.ppf(_LEVEL, 1)), abs=1e-6)

    def test_the_upper_side_warning_names_the_estimate_and_the_search_limit(self) -> None:
        _, t = self._result()
        assert t.value is not None

        side_warnings = [w for w in t.warnings if "nothing to search" in w]
        assert len(side_warnings) == 1 and "upper" in side_warnings[0], t.warnings
        numbers = _numbers_in(side_warnings[0])
        assert any(abs(x - math.log(t.value)) < 1e-6 for x in numbers), side_warnings
        assert any(abs(x - math.log(self._HIGHEST_SEARCH_LIMIT)) < 1e-6 for x in numbers)


class TestARealFitWhoseEstimateIsFarPastTheTestedRange:
    """Cells at log severities ``[0, 1, 10, 100]`` with 100, 90, 89 and 88 successes of 100: a
    nearly flat curve whose probit crossing of ``absolute(0.8)`` is at ``theta_hat = 19.0`` on the
    log axis, well above the upper search limit (``log(100) + log(100) = 9.21``).

    The top cell is 88/100 against a target of 0.8: its lower bound at the confidence the exact
    claims use (0.9875 at level 0.95) is 0.79, just under the target (at level 0.95 itself it is
    0.81, which a test at `level` would take as ``RIGHT``), so ``RIGHT`` does not fire and the
    model's value is reported (flagged as extrapolated). That puts the estimate beyond the upper
    search limit.

    Only the estimate is out there: the profile's lower endpoint is at severity 161 (``theta``
    5.08), inside the upper limit, so the walk to the upper limit finds no sign change whether or
    not the limit is on the right side of the estimate. This input therefore pins the behaviour
    below, and does not tell a search that checks the limit's side from one that does not; the two
    classes above are the inputs that do.
    """

    @staticmethod
    def _fit() -> Fit:
        return _fit([0.0, 1.0, 10.0, 100.0], [100, 90, 89, 88], [100, 100, 100, 100])

    @classmethod
    def _threshold(cls) -> Threshold:
        return threshold(
            cls._fit(), definition=Definition.absolute(0.8), interval_method="profile", level=0.95
        )

    @classmethod
    def _profile_lower_endpoint(cls) -> float:
        """The severity where the profile deviance first reaches the critical value below the
        estimate, from the independent profile above. ``g(5)`` is positive and ``g(6)`` negative."""
        fit = cls._fit()

        def excess(theta: float) -> float:
            return _deviance_excess(fit, target=0.8, severity=math.exp(theta))

        assert excess(5.0) > 0.0 > excess(6.0)
        return math.exp(brentq(excess, 5.0, 6.0, xtol=1.0e-10))

    def test_the_model_value_is_reported_beyond_the_upper_search_limit(self) -> None:
        t = self._threshold()

        assert t.status is Status.OK
        assert t.censoring is not Censoring.RIGHT, "the top cell does not pass at the exact level"
        assert t.value is not None
        # theta_hat = log(value) is beyond the upper search limit, log(100) + log(100).
        assert math.log(t.value) > 2.0 * math.log(100.0)

    def test_the_upper_side_is_open_and_no_reported_bound_crosses_the_value(self) -> None:
        """What the search returns today. ``hi`` is open (the estimate is beyond the upper limit);
        ``lo`` is whatever the lower walk finds, which must lie below the value if it is reported
        at all."""
        t = self._threshold()

        assert t.value is not None
        assert t.hi is None
        assert t.lo is None or t.lo < t.value
        assert any("nothing to search" in w and "upper" in w for w in t.warnings), t.warnings

    def test_the_profile_has_a_lower_endpoint_near_severity_161(self) -> None:
        """The interval's lower endpoint exists: it is where an independently computed profile
        deviance crosses the chi-square critical value."""
        assert self._profile_lower_endpoint() == pytest.approx(161.0, rel=5e-3)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=(
            "The lower walk steps outward in growing strides and stops at the first stride that "
            "lands where the inner fit fails, without trying a shorter stride. Here the stride "
            "from theta 7.5 lands near -0.5, past the point (theta about 2.1) where the inner "
            "fit fails, so a real lower endpoint at theta 5.08 is never reached and the lower "
            "side is reported open. This is a known limitation, and conservative: a missing "
            "bound is never a wrong one."
        ),
    )
    def test_the_lower_bound_is_found_near_severity_161(self) -> None:
        t = self._threshold()

        assert t.lo is not None, "the lower side was left open"
        assert t.lo == pytest.approx(self._profile_lower_endpoint(), rel=1e-3)
