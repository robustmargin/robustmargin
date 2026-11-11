"""``threshold()`` on a Gaussian fit: target resolution, interval calibration, and what it refuses.

* **Targets are in response units.** ``absolute(a)`` takes any finite ``a``. ``relative(p)`` is a
  fraction of the fitted range, and ``baseline_fraction(p)`` and ``baseline_multiple(k)`` are
  multiples of the fitted zero-severity level, so they refuse a level that is not positive.
* **Intervals are calibrated on the residual df.** The delta interval is ``theta +- t(df) * se``
  and the profile cut-off is ``F(1, df)`` on ``(RSS_theta - RSS) / (RSS / df)``. Neither uses the
  normal or chi-square quantile.
* **No exact claims.** A Gaussian threshold is ``OK``, ``UNREACHABLE`` (the target lies outside
  the fitted ``(l, u)``) or ``NOT_CONVERGED`` (the fit failed, and nothing is reported). Never
  ``FAILS_AT_BASELINE``, ``RIGHT`` or ``LEFT``.
* **Beyond the tested range** the value is reported with an ``EXTRAPOLATED:`` warning.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import chi2, norm
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

from robustmargin import (
    Axis,
    Censoring,
    ContinuousObservations,
    Definition,
    Fit,
    Status,
    Threshold,
    fit_dose_response,
    threshold,
)

_LOG = Axis(name="severity", unit="unit", scale="log")
_LINEAR = Axis(name="severity", unit="unit", scale="linear")
_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
_MU, _S, _U, _L = math.log(0.8), 0.5, 10.0, 1.0
_ESTIMATE = {"upper": "estimate", "lower": "estimate"}


def _cdf(z: np.ndarray, link: str) -> np.ndarray:
    if link == "logit":
        return np.asarray(1.0 / (1.0 + np.exp(-z)))
    if link == "probit":
        return np.asarray(norm.cdf(z))
    return np.asarray(1.0 - np.exp(-np.exp(z)))


def _quantile(q: float, link: str) -> float:
    if link == "logit":
        return math.log(q / (1.0 - q))
    if link == "probit":
        return float(norm.ppf(q))
    return math.log(-math.log(1.0 - q))


def _series(
    *, seed: int = 20261108, reps: int = 4, link: str = "logit", sigma: float = 0.3,
    linear: bool = False,
) -> tuple[list[float], list[float]]:  # fmt: skip
    rng = np.random.default_rng(seed)
    levels = np.linspace(0.0, 10.0, 8).tolist() if linear else _LEVELS
    severity = [x for x in levels for _ in range(reps)]
    x = np.asarray(severity)
    if linear:
        f = _cdf((x - 4.0) / 1.2, link)
    else:
        with np.errstate(divide="ignore"):
            z = (np.log(x) - _MU) / _S
        f = np.where(x == 0.0, 0.0, _cdf(z, link))
    return severity, (_U - (_U - _L) * f + rng.normal(0.0, sigma, size=x.size)).tolist()


def _fit(
    *,
    link: str = "logit",
    direction: str = "decreasing",
    shift: float = 0.0,
    axis: Axis = _LOG,
    reps: int = 4,
    seed: int = 20261108,
    **asymptotes: object,
) -> Fit:
    severity, value = _series(link=link, reps=reps, seed=seed, linear=axis.scale == "linear")
    sign = 1.0 if direction == "decreasing" else -1.0
    obs = ContinuousObservations.from_observations(
        axis,
        severity=severity,
        value=[sign * v + shift for v in value],
        outcome="response",
        direction=direction,
    )
    kw = {**_ESTIMATE, **asymptotes}
    fit = fit_dose_response(obs, model="gaussian", link=link, **kw)  # type: ignore[arg-type]
    assert fit.status is Status.OK
    return fit


def _solve(
    fit: Fit, definition: Definition, method: str = "delta", level: float = 0.95
) -> Threshold:
    return threshold(fit, definition=definition, interval_method=method, level=level)


def _theta_and_se(fit: Fit, p: float, link: str) -> tuple[float, float]:
    """``theta`` and its delta-method SE for ``relative(p)``: ``theta = mu + s F^-1(p)``,
    which depends on the asymptotes not at all."""
    assert fit.params is not None and fit.covariance is not None
    names = list(fit.covariance.names)
    sigma = np.array(fit.covariance.matrix)
    idx = [names.index("mu"), names.index("s")]
    grad = np.array([1.0, _quantile(p, link)])
    theta = fit.params["mu"].value + fit.params["s"].value * grad[1]
    return theta, float(math.sqrt(grad @ sigma[np.ix_(idx, idx)] @ grad))


# ---------------------------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------------------------


class TestTargetResolution:
    @pytest.mark.parametrize("link", ["logit", "probit", "cloglog"])
    def test_relative_is_the_curve_at_that_fraction_of_the_fitted_range(self, link: str) -> None:
        fit = _fit(link=link)
        assert fit.params is not None

        t = _solve(fit, Definition.relative(0.3))

        expected = math.exp(fit.params["mu"].value + fit.params["s"].value * _quantile(0.3, link))
        assert t.status is Status.OK and t.value == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("link", ["logit", "probit", "cloglog"])
    def test_absolute_is_in_response_units(self, link: str) -> None:
        fit = _fit(link=link)
        assert fit.params is not None
        u, low = fit.params["upper"].value, fit.params["lower"].value

        t = _solve(fit, Definition.absolute(4.0))

        q = (u - 4.0) / (u - low)
        expected = math.exp(fit.params["mu"].value + fit.params["s"].value * _quantile(q, link))
        assert t.value == pytest.approx(expected, rel=1e-9)
        assert t.definition == Definition.absolute(4.0)

    def test_absolute_takes_a_negative_level(self) -> None:
        fit = _fit(shift=-50.0)
        assert fit.params is not None
        u, low = fit.params["upper"].value, fit.params["lower"].value

        t = _solve(fit, Definition.absolute(-45.0))

        q = (u + 45.0) / (u - low)
        expected = math.exp(fit.params["mu"].value + fit.params["s"].value * _quantile(q, "logit"))
        assert t.status is Status.OK and t.value == pytest.approx(expected, rel=1e-9)

    def test_absolute_on_a_linear_axis_returns_a_severity_not_its_log(self) -> None:
        fit = _fit(axis=_LINEAR)
        assert fit.params is not None
        u, low = fit.params["upper"].value, fit.params["lower"].value

        t = _solve(fit, Definition.absolute(5.0))

        q = (u - 5.0) / (u - low)
        expected = fit.params["mu"].value + fit.params["s"].value * _quantile(q, "logit")
        assert t.value == pytest.approx(expected, rel=1e-9)

    def test_baseline_fraction_is_a_fraction_of_the_fitted_upper_asymptote(self) -> None:
        fit = _fit()
        assert fit.params is not None
        u, low = fit.params["upper"].value, fit.params["lower"].value

        t = _solve(fit, Definition.baseline_fraction(0.5))

        q = (u - 0.5 * u) / (u - low)
        expected = math.exp(fit.params["mu"].value + fit.params["s"].value * _quantile(q, "logit"))
        assert t.status is Status.OK and t.value == pytest.approx(expected, rel=1e-9)

    def test_baseline_multiple_is_a_multiple_of_the_fitted_lower_asymptote(self) -> None:
        fit = _fit(direction="increasing", shift=11.0)  # rises from about 1 to about 10
        assert fit.params is not None
        u, low = fit.params["upper"].value, fit.params["lower"].value

        t = _solve(fit, Definition.baseline_multiple(2.0))

        q = (2.0 * low - low) / (u - low)
        expected = math.exp(fit.params["mu"].value + fit.params["s"].value * _quantile(q, "logit"))
        assert t.status is Status.OK and t.value == pytest.approx(expected, rel=1e-9)
        assert t.fit is fit

    def test_the_fit_is_at_the_target_where_the_threshold_says(self) -> None:
        fit = _fit()
        t = _solve(fit, Definition.absolute(6.0))
        assert t.value is not None

        assert float(fit.predict(t.value).estimate) == pytest.approx(6.0, abs=1e-9)


class TestBaselineSignRefusals:
    def test_baseline_fraction_raises_when_the_fitted_upper_asymptote_is_not_positive(self) -> None:
        fit = _fit(shift=-12.0)  # u about -2
        assert fit.params is not None and fit.params["upper"].value <= 0.0

        with pytest.raises(ValueError):
            _solve(fit, Definition.baseline_fraction(0.5))

    def test_baseline_multiple_raises_when_the_fitted_lower_asymptote_is_not_positive(self) -> None:
        fit = _fit(direction="increasing", shift=3.0)  # rises from about -2
        assert fit.params is not None and fit.params["lower"].value <= 0.0

        with pytest.raises(ValueError):
            _solve(fit, Definition.baseline_multiple(2.0))

    def test_a_positive_fitted_level_is_not_refused(self) -> None:
        assert _solve(_fit(), Definition.baseline_fraction(0.5)).status is Status.OK

    def test_baseline_multiple_still_needs_an_increasing_outcome(self) -> None:
        with pytest.raises(ValueError):
            _solve(_fit(), Definition.baseline_multiple(2.0))

    def test_baseline_multiple_still_needs_a_log_axis(self) -> None:
        fit = _fit(direction="increasing", shift=11.0, axis=_LINEAR)

        with pytest.raises(ValueError):
            _solve(fit, Definition.baseline_multiple(2.0))

    def test_baseline_fraction_still_needs_a_decreasing_outcome(self) -> None:
        fit = _fit(direction="increasing", shift=11.0)

        with pytest.raises(ValueError):
            _solve(fit, Definition.baseline_fraction(0.5))


# ---------------------------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------------------------


class TestDeltaIntervalUsesStudentT:
    @pytest.mark.parametrize("link", ["logit", "probit", "cloglog"])
    @pytest.mark.parametrize("level", [0.90, 0.95, 0.99])
    def test_the_interval_is_theta_hat_plus_or_minus_t_times_se_on_the_log_axis(
        self, link: str, level: float
    ) -> None:
        fit = _fit(link=link)
        assert fit.residual is not None
        theta, se = _theta_and_se(fit, 0.8, link)

        t = _solve(fit, Definition.relative(0.8), "delta", level)

        quantile = float(t_dist.ppf(0.5 + level / 2.0, fit.residual.df))
        assert t.value == pytest.approx(math.exp(theta), rel=1e-9)
        assert t.lo == pytest.approx(math.exp(theta - quantile * se), rel=1e-9)
        assert t.hi == pytest.approx(math.exp(theta + quantile * se), rel=1e-9)
        assert t.interval_method == "delta" and t.level == level

    def test_the_interval_on_a_linear_axis_is_symmetric_on_the_log_scale(self) -> None:
        fit = _fit(axis=_LINEAR)
        assert fit.residual is not None
        theta, se = _theta_and_se(fit, 0.8, "logit")

        t = _solve(fit, Definition.relative(0.8))

        assert t.value is not None and t.lo is not None and t.hi is not None
        assert t.value == pytest.approx(theta, rel=1e-9)  # x* = theta on a linear axis
        quantile = float(t_dist.ppf(0.975, fit.residual.df))
        se_log = se / t.value  # Var(log x*) = Var(x*) / x*^2
        assert t.lo == pytest.approx(t.value * math.exp(-quantile * se_log), rel=1e-9)
        assert t.hi == pytest.approx(t.value * math.exp(quantile * se_log), rel=1e-9)
        assert math.log(t.hi) - math.log(t.value) == pytest.approx(
            math.log(t.value) - math.log(t.lo), rel=1e-9
        )

    def test_the_degrees_of_freedom_are_those_of_the_residual(self) -> None:
        fixed = _fit(upper=_U, lower=_L)  # two parameters, so more df than the four-parameter fit
        free = _fit()
        assert fixed.residual is not None and free.residual is not None
        assert fixed.residual.df == free.residual.df + 2

        t = _solve(fixed, Definition.relative(0.8))

        theta, se = _theta_and_se(fixed, 0.8, "logit")
        assert t.hi == pytest.approx(
            math.exp(theta + float(t_dist.ppf(0.975, fixed.residual.df)) * se), rel=1e-9
        )

    def test_it_is_wider_than_the_normal_interval_at_a_small_df(self) -> None:
        fit = _fit(reps=1, upper=_U, lower=_L)
        assert fit.residual is not None and fit.residual.df == 6
        theta, se = _theta_and_se(fit, 0.8, "logit")

        t = _solve(fit, Definition.relative(0.8))

        assert t.hi is not None and t.hi > math.exp(theta + float(norm.ppf(0.975)) * se) * 1.01


def _profile_oracle(link: str, p: float, reps: int):  # type: ignore[no-untyped-def]
    """``RSS`` at the minimum and a function giving the constrained minimum at a ``theta``.

    This file's own code: the asymptotes are fixed at their generating values, so once ``theta``
    pins ``mu = theta - s * F^-1(p)`` only ``s`` is free, and a dense grid in ``log s`` with a
    bounded polish finds the minimum.
    """
    severity, value = _series(link=link, reps=reps)
    x, y = np.asarray(severity), np.asarray(value)
    zero = x == 0.0
    lx = np.log(np.where(zero, 1.0, x))

    def residuals(mu: float, s: float) -> np.ndarray:
        f = np.where(zero, 0.0, _cdf((lx - mu) / s, link))
        return np.asarray(_U - (_U - _L) * f - y)

    best = least_squares(
        lambda v: residuals(v[0], math.exp(v[1])),
        [_MU, math.log(_S)],
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
    )

    def rss_at(theta: float, log_s: float) -> float:
        s = math.exp(log_s)
        return float(np.sum(residuals(theta - s * _quantile(p, link), s) ** 2))

    def constrained(theta: float) -> float:
        grid = np.linspace(math.log(0.02), math.log(20.0), 600)
        values = [rss_at(theta, g) for g in grid]
        k = int(np.argmin(values))
        polish = minimize_scalar(
            lambda g: rss_at(theta, g),
            bounds=(grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]),
            method="bounded",
            options={"xatol": 1e-12},
        )
        return float(min(values[k], polish.fun))

    return 2.0 * best.cost, constrained


class TestProfileIntervalUsesTheFDistribution:
    @pytest.mark.parametrize("link", ["logit", "probit"])
    def test_each_bound_sits_where_the_f_statistic_reaches_the_f_quantile(self, link: str) -> None:
        fit = _fit(link=link, upper=_U, lower=_L)
        assert fit.residual is not None
        rss_hat, constrained = _profile_oracle(link, 0.8, reps=4)

        t = _solve(fit, Definition.relative(0.8), "profile")

        assert fit.residual.rss == pytest.approx(rss_hat, rel=1e-6)
        cutoff = float(f_dist.ppf(0.95, 1, fit.residual.df))
        assert t.status is Status.OK and t.lo is not None and t.hi is not None
        for bound in (t.lo, t.hi):
            statistic = (constrained(math.log(bound)) - rss_hat) / (rss_hat / fit.residual.df)
            assert statistic == pytest.approx(cutoff, rel=5e-3)
        assert t.interval_method == "profile"

    def test_the_cut_off_is_not_the_chi_square_quantile(self) -> None:
        fit = _fit(upper=_U, lower=_L, reps=1)
        assert fit.residual is not None and fit.residual.df == 6
        rss_hat, constrained = _profile_oracle("logit", 0.8, reps=1)

        t = _solve(fit, Definition.relative(0.8), "profile")

        assert t.hi is not None
        statistic = (constrained(math.log(t.hi)) - rss_hat) / (rss_hat / fit.residual.df)
        # F(1, 6; .95) = 5.99 against chi2(1; .95) = 3.84.
        assert statistic > 1.5 * float(chi2.ppf(0.95, 1))

    def test_the_profile_contains_the_point_estimate_and_widens_with_the_level(self) -> None:
        fit = _fit()

        narrow = _solve(fit, Definition.relative(0.5), "profile", 0.90)
        wide = _solve(fit, Definition.relative(0.5), "profile", 0.99)

        assert narrow.lo is not None and narrow.hi is not None and narrow.value is not None
        assert wide.lo is not None and wide.hi is not None
        assert narrow.lo < narrow.value < narrow.hi
        assert wide.lo < narrow.lo and narrow.hi < wide.hi

    def test_the_profile_agrees_with_the_delta_interval_to_first_order(self) -> None:
        fit = _fit(reps=8)

        prof = _solve(fit, Definition.relative(0.5), "profile")
        delta = _solve(fit, Definition.relative(0.5), "delta")

        assert prof.lo == pytest.approx(delta.lo, rel=0.05)
        assert prof.hi == pytest.approx(delta.hi, rel=0.05)


# ---------------------------------------------------------------------------------------------
# Censoring, reachability, extrapolation, failure
# ---------------------------------------------------------------------------------------------


class TestUnreachable:
    @pytest.mark.parametrize("level", [50.0, 11.0, 0.0, -3.0])
    def test_a_level_outside_the_fitted_range_is_unreachable_with_no_number(
        self, level: float
    ) -> None:
        fit = _fit()  # fitted range is about (1, 10)

        t = _solve(fit, Definition.absolute(level))

        assert t.status is Status.UNREACHABLE
        assert (t.value, t.lo, t.hi, t.grid) == (None, None, None, None)

    def test_a_baseline_fraction_below_the_fitted_lower_asymptote_is_unreachable(self) -> None:
        fit = _fit()  # 0.05 * u is about 0.5, below l is about 1

        t = _solve(fit, Definition.baseline_fraction(0.05))

        assert t.status is Status.UNREACHABLE and t.value is None

    def test_a_baseline_multiple_above_the_fitted_upper_asymptote_is_unreachable(self) -> None:
        fit = _fit(direction="increasing", shift=11.0)  # l about 1, u about 10

        t = _solve(fit, Definition.baseline_multiple(30.0))

        assert t.status is Status.UNREACHABLE and t.value is None

    def test_reachability_is_judged_on_the_fitted_range_not_the_unit_interval(self) -> None:
        fit = _fit()

        assert _solve(fit, Definition.absolute(5.0)).status is Status.OK
        assert _solve(fit, Definition.absolute(1.5)).status is Status.OK  # above 1: fine here


class TestExtrapolation:
    def test_a_value_beyond_the_highest_nonzero_level_carries_the_extrapolated_warning(
        self,
    ) -> None:
        t = _solve(_fit(), Definition.relative(0.999))  # about 26, tested up to 6.4

        assert t.status is Status.OK and t.value is not None and t.value > 6.4
        assert any(w.startswith("EXTRAPOLATED:") for w in t.warnings)

    def test_a_value_below_the_lowest_nonzero_level_carries_the_extrapolated_warning(self) -> None:
        t = _solve(_fit(), Definition.relative(0.001))  # about 0.025, lowest nonzero level 0.1

        assert t.status is Status.OK and t.value is not None and t.value < 0.1
        assert any(w.startswith("EXTRAPOLATED:") for w in t.warnings)

    def test_a_value_inside_the_tested_range_has_no_such_warning(self) -> None:
        t = _solve(_fit(), Definition.relative(0.5))

        assert not any(w.startswith("EXTRAPOLATED:") for w in t.warnings)

    def test_an_extrapolated_value_is_still_a_value_and_never_a_censoring_label(self) -> None:
        t = _solve(_fit(), Definition.relative(0.999))

        assert t.censoring is Censoring.NONE and t.value is not None


class TestNoExactClaims:
    @pytest.mark.parametrize("method", ["delta", "profile"])
    @pytest.mark.parametrize(
        "definition",
        [Definition.relative(0.5), Definition.absolute(5.0), Definition.absolute(500.0)],
    )
    def test_a_converged_fit_gives_ok_or_unreachable_and_never_an_exact_label(
        self, method: str, definition: Definition
    ) -> None:
        t = _solve(_fit(), definition, method)

        assert t.status in (Status.OK, Status.UNREACHABLE)
        assert t.censoring in (Censoring.NONE, Censoring.OPEN_UPPER, Censoring.OPEN_LOWER)
        assert t.baseline is None and t.grid is None


class TestAFailedFit:
    def _failed(self) -> Fit:
        obs = ContinuousObservations.from_observations(
            _LOG,
            severity=[0.5, 1.0, 2.0, 4.0],
            value=[9.0, 7.0, 3.0, 1.5],
            outcome="response",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="gaussian", link="logit", **_ESTIMATE)  # type: ignore[arg-type]
        assert fit.status is Status.NOT_CONVERGED
        return fit

    @pytest.mark.parametrize("method", ["delta", "profile"])
    @pytest.mark.parametrize("definition", [Definition.relative(0.5), Definition.absolute(5.0)])
    def test_the_threshold_is_not_converged_with_no_number_no_bound_and_no_grid(
        self, method: str, definition: Definition
    ) -> None:
        t = _solve(self._failed(), definition, method)

        assert t.status is Status.NOT_CONVERGED
        assert (t.value, t.lo, t.hi, t.grid) == (None, None, None, None)
        assert t.censoring is Censoring.NONE

    def test_no_x_max_masquerades_as_the_estimate(self) -> None:
        t = _solve(self._failed(), Definition.absolute(0.0))

        assert t.value is None and t.hi is None

    def test_a_baseline_definition_on_a_failed_fit_is_refused(self) -> None:
        with pytest.raises(ValueError):
            _solve(self._failed(), Definition.baseline_fraction(0.5))
