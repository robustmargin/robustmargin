"""Failure paths of the Gaussian family that the other continuous test files do not reach.

* **Welch.** An adjacent pair with a one-observation level, or with zero spread at both levels,
  is untestable (p = 1, no statistic, not flagged).
* **NOT_CONVERGED reasons.** ``optimizer``, ``zero residual`` and constant level means, each
  with no number anywhere.
* **Delta interval.** An overflowing half-width reports the estimate with no bounds; on a linear
  axis the interval is still log-scale, closed and asymmetric.
* **Invariants.** A Gaussian ``Fit`` and ``Threshold`` refuse the binomial-only statuses,
  censoring and fields.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.stats import t as t_dist

import robustmargin.models as models
from robustmargin import (
    Axis,
    Censoring,
    ContinuousObservations,
    Covariance,
    Definition,
    Fit,
    Residual,
    Status,
    check_monotonicity,
    fit_dose_response,
    threshold,
)
from robustmargin.testing import fake_fit, fake_threshold

_LOG = Axis(name="severity", unit="unit", scale="log")
_LINEAR = Axis(name="severity", unit="unit", scale="linear")
_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
_ESTIMATE: dict[str, Any] = {"upper": "estimate", "lower": "estimate"}


def _summaries(
    mean: list[float], sd: list[float], n: list[int], *, direction: str = "decreasing"
) -> ContinuousObservations:
    return ContinuousObservations.from_summaries(
        _LOG,
        severity=[float(i + 1) for i in range(len(mean))],
        mean=mean,
        sd=sd,
        n=n,
        outcome="response",
        direction=direction,
    )


def _fit(axis: Axis, seed: int = 1) -> Fit:
    rng = np.random.default_rng(seed)
    levels = _LEVELS if axis.scale == "log" else np.linspace(0.0, 10.0, 8).tolist()
    severity = [x for x in levels for _ in range(4)]
    x = np.asarray(severity)
    if axis.scale == "log":
        with np.errstate(divide="ignore"):
            z = (np.log(x) - math.log(0.8)) / 0.5
        f = np.where(x == 0.0, 0.0, 1.0 / (1.0 + np.exp(-z)))
    else:
        f = 1.0 / (1.0 + np.exp(-(x - 4.0) / 1.2))
    y = 10.0 - 9.0 * f + rng.normal(0.0, 0.3, size=x.size)
    obs = ContinuousObservations.from_observations(
        axis, severity=severity, value=y.tolist(), outcome="response", direction="decreasing"
    )
    fit = fit_dose_response(obs, model="gaussian", link="logit", **_ESTIMATE)
    assert fit.status is Status.OK
    return fit


def _inflated(fit: Fit, factor: float) -> Fit:
    """The same fit with every covariance entry multiplied by ``factor``: a design that does not
    constrain the parameters."""
    assert fit.covariance is not None
    matrix = tuple(tuple(v * factor for v in row) for row in fit.covariance.matrix)
    return replace(fit, covariance=Covariance(names=fit.covariance.names, matrix=matrix))


def _log_scale_delta_at_median(fit: Fit, value: float, level: float) -> tuple[float, float]:
    """Independent log-scale delta interval for ``relative(0.5)`` on a logit link, where
    ``x* = mu`` (the logit median), so ``grad x* = e_mu`` and ``se(x*) = sqrt(Var(mu))``."""
    assert fit.covariance is not None and fit.residual is not None
    names = list(fit.covariance.names)
    var_mu = fit.covariance.matrix[names.index("mu")][names.index("mu")]
    se_log = math.sqrt(var_mu) / value
    quantile = float(t_dist.ppf((1.0 + level) / 2.0, fit.residual.df))
    return value * math.exp(-quantile * se_log), value * math.exp(quantile * se_log)


class TestWelchMonotonicityPairs:
    def test_a_pair_with_a_one_observation_level_is_untestable_with_p_one_and_a_warning(
        self,
    ) -> None:
        obs = _summaries([5.0, 6.0, 3.0, 3.0], [0.0, 0.0, 0.0, 0.0], [1, 3, 3, 3])

        check = check_monotonicity(obs)

        first = check.pairs[0]
        assert (first.statistic, first.df, first.p_value) == (None, None, 1.0)
        assert first.holm_adjusted_p == 1.0 and not first.flagged
        assert any("single observation" in w for w in check.warnings)
        assert check.method == "welch_t_one_sided_holm"

    def test_untestable_pairs_are_named_in_the_warning_and_testable_ones_are_not(self) -> None:
        obs = _summaries([5.0, 6.0, 3.0], [0.0, 0.0, 0.0], [3, 1, 3])

        (warning,) = check_monotonicity(obs).warnings

        assert "(1.0, 2.0)" in warning and "(2.0, 3.0)" in warning

    @pytest.mark.parametrize("direction", ["decreasing", "increasing"])
    def test_zero_spread_at_both_levels_is_untestable_and_not_flagged_even_for_a_reversal(
        self, direction: str
    ) -> None:
        # Welch's statistic divides by the pooled standard error, which is zero here: there is
        # nothing to test. A reversal of the means is not a significant one.
        obs = _summaries([5.0, 6.0, 3.0], [0.0, 0.0, 0.0], [3, 3, 3], direction=direction)

        check = check_monotonicity(obs)

        for pair in check.pairs:
            assert (pair.statistic, pair.df, pair.p_value) == (None, None, 1.0)
            assert pair.holm_adjusted_p == 1.0
            assert not pair.flagged
        assert not check.any_flagged
        text = " ".join(check.warnings)
        assert "(1.0, 2.0)" in text and "(2.0, 3.0)" in text
        assert "single observation or no spread at either level" in text

    def test_zero_spread_at_one_level_only_is_still_testable(self) -> None:
        obs = _summaries([5.0, 6.0], [0.0, 0.5], [3, 3])

        (pair,) = check_monotonicity(obs).pairs

        assert pair.statistic is not None and pair.df is not None
        assert 0.0 < pair.p_value < 1.0


class TestNotConvergedReasons:
    _EXACT_MEAN = [
        10.0 - 9.0 / (1.0 + math.exp(-(math.log(x) - math.log(0.8)) / 0.5)) for x in _LEVELS[1:]
    ]

    def _exact_summaries(self) -> ContinuousObservations:
        return ContinuousObservations.from_summaries(
            _LOG,
            severity=_LEVELS[1:],
            mean=self._EXACT_MEAN,
            sd=[0.0] * 7,
            n=[3] * 7,
            outcome="response",
            direction="decreasing",
        )

    @staticmethod
    def _assert_no_number(fit: Fit, phrase: str) -> None:
        assert fit.status is Status.NOT_CONVERGED
        assert fit.params is None and fit.covariance is None and fit.log_likelihood is None
        assert fit.residual is None and fit.diagnostics is None
        assert any(phrase in w for w in fit.warnings), fit.warnings

    def test_the_exact_summaries_do_fit_when_nothing_is_patched(self) -> None:
        fit = fit_dose_response(
            self._exact_summaries(), model="gaussian", link="logit", **_ESTIMATE
        )

        assert fit.status is Status.OK  # so the two failures below come from the patch

    def test_an_optimizer_that_raises_from_every_start_is_the_optimizer_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raising(*args: object, **kwargs: object) -> object:
            raise ValueError("no solution")

        monkeypatch.setattr(models, "least_squares", raising)

        fit = fit_dose_response(
            self._exact_summaries(), model="gaussian", link="logit", **_ESTIMATE
        )

        self._assert_no_number(fit, "the optimizer did not converge")
        assert fit.model == "gaussian"

    def test_an_optimizer_that_reports_failure_is_the_optimizer_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = models.least_squares

        def unsuccessful(*args: Any, **kwargs: Any) -> object:
            run = real(*args, **kwargs)
            return SimpleNamespace(x=run.x, cost=run.cost, status=0)

        monkeypatch.setattr(models, "least_squares", unsuccessful)

        fit = fit_dose_response(
            self._exact_summaries(), model="gaussian", link="logit", **_ESTIMATE
        )

        self._assert_no_number(fit, "the optimizer did not converge")

    def test_a_curve_through_every_observation_is_the_zero_residual_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Level means exactly on the curve and no within-level spread leave a residual sum of
        # squares of zero up to rounding; the optimiser's cost is forced to exactly zero so the
        # exact-zero branch, which rounding would otherwise hide, is the one exercised.
        real = models.least_squares

        def exact(*args: Any, **kwargs: Any) -> object:
            run = real(*args, **kwargs)
            return SimpleNamespace(x=run.x, cost=0.0, status=run.status)

        monkeypatch.setattr(models, "least_squares", exact)

        fit = fit_dose_response(
            self._exact_summaries(), model="gaussian", link="logit", **_ESTIMATE
        )

        self._assert_no_number(fit, "zero residual")

    @pytest.mark.parametrize(
        "value", [[5.0] * 24, [5.0, 5.1, 4.9] * 8], ids=["no-spread", "spread"]
    )
    def test_level_means_that_are_all_equal_are_the_constant_boundary_reason(
        self, value: list[float]
    ) -> None:
        severity = [x for x in _LEVELS for _ in range(3)]
        obs = ContinuousObservations.from_observations(
            _LOG, severity=severity, value=value, outcome="response", direction="decreasing"
        )

        fit = fit_dose_response(obs, model="gaussian", link="logit", **_ESTIMATE)

        self._assert_no_number(fit, "every level has the same mean")
        assert any("boundary" in w for w in fit.warnings)


class TestDeltaIntervalEdges:
    def test_a_delta_half_width_beyond_the_float_range_gives_the_estimate_with_no_bounds(
        self,
    ) -> None:
        fit = _fit(_LOG)
        honest = threshold(
            fit, definition=Definition.relative(0.5), interval_method="delta", level=0.95
        )

        t = threshold(
            _inflated(fit, 1.0e12),
            definition=Definition.relative(0.5),
            interval_method="delta",
            level=0.95,
        )

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.value == honest.value and t.value is not None
        assert (t.lo, t.hi) == (None, None)
        assert any("too large for a finite interval" in w for w in t.warnings)

    def test_a_linear_axis_delta_interval_is_built_on_the_log_scale_so_it_is_never_open(
        self,
    ) -> None:
        """The delta interval is ``x* * exp(-/+ t * se_log)`` with ``se_log = se(x*) / x*`` on
        every axis: never open, never clipped, asymmetric on a linear axis. This fit's symmetric
        interval would have put the lower end below the origin."""
        rng = np.random.default_rng(1)
        x = np.asarray([0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])
        y = 10.0 - 9.0 / (1.0 + np.exp(-(x - 0.8) / 0.5)) + rng.normal(0.0, 0.6, size=x.size)
        obs = ContinuousObservations.from_observations(
            _LINEAR, severity=x.tolist(), value=y.tolist(), outcome="response",
            direction="decreasing",
        )  # fmt: skip
        fit = fit_dose_response(obs, model="gaussian", link="logit", **_ESTIMATE)
        assert fit.status is Status.OK

        t = threshold(fit, definition=Definition.relative(0.5), interval_method="delta", level=0.95)

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.value is not None and t.lo is not None and t.hi is not None
        assert 0.0 < t.lo < t.value < t.hi
        lo, hi = _log_scale_delta_at_median(fit, t.value, 0.95)
        assert t.lo == pytest.approx(lo, rel=1e-9)
        assert t.hi == pytest.approx(hi, rel=1e-9)

    def test_an_inflated_covariance_on_a_linear_axis_still_gives_a_closed_log_scale_interval(
        self,
    ) -> None:
        fit = _inflated(_fit(_LINEAR), 1.0e3)

        t = threshold(fit, definition=Definition.relative(0.5), interval_method="delta", level=0.95)

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.value is not None and t.lo is not None and t.hi is not None
        assert 0.0 < t.lo < t.value < t.hi
        lo, hi = _log_scale_delta_at_median(fit, t.value, 0.95)
        assert t.lo == pytest.approx(lo, rel=1e-9)
        assert t.hi == pytest.approx(hi, rel=1e-9)

    def test_a_linear_axis_delta_half_width_beyond_the_float_range_gives_the_estimate_alone(
        self,
    ) -> None:
        """As on a log axis (and as the binomial path): overflowing ``exp`` reports the estimate
        with no bounds, ``censoring is NONE`` and a warning."""
        t = threshold(
            _inflated(_fit(_LINEAR), 1.0e12),
            definition=Definition.relative(0.5),
            interval_method="delta",
            level=0.95,
        )

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.value is not None and (t.lo, t.hi) == (None, None)
        assert any("too large for a finite interval" in w for w in t.warnings)

    def test_a_linear_axis_delta_interval_above_the_origin_is_not_clipped(self) -> None:
        t = threshold(
            _fit(_LINEAR),
            definition=Definition.relative(0.5),
            interval_method="delta",
            level=0.95,
        )

        assert t.lo is not None and t.value is not None and t.hi is not None
        assert 0.0 < t.lo < t.value < t.hi
        assert t.warnings == ()


class TestGaussianFitInvariants:
    def test_a_separation_status_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no separation"):
            replace(
                fake_fit(model="gaussian", status=Status.NOT_CONVERGED), status=Status.SEPARATION
            )

    @pytest.mark.parametrize(
        "status", [Status.UNREACHABLE, Status.FAILS_AT_BASELINE, Status.CONTROL_INCOMPATIBLE]
    )
    def test_a_status_other_than_ok_or_not_converged_is_refused(self, status: Status) -> None:
        with pytest.raises(ValueError):
            replace(fake_fit(model="gaussian"), status=status)

    def test_binomial_cells_on_a_gaussian_fit_are_refused(self) -> None:
        binomial = fake_fit()

        with pytest.raises(ValueError, match="ContinuousCell"):
            replace(fake_fit(model="gaussian"), cells=binomial.cells)

    def test_continuous_cells_on_a_binomial_fit_are_refused(self) -> None:
        gaussian = fake_fit(model="gaussian")

        with pytest.raises(ValueError, match="Cell"):
            replace(fake_fit(), cells=gaussian.cells)

    def test_an_ok_gaussian_fit_without_a_residual_is_refused(self) -> None:
        with pytest.raises(ValueError, match="residual"):
            replace(fake_fit(model="gaussian"), residual=None)

    def test_a_binomial_fit_with_a_residual_is_refused(self) -> None:
        residual = fake_fit(model="gaussian").residual
        assert isinstance(residual, Residual)

        with pytest.raises(ValueError, match="residual"):
            replace(fake_fit(), residual=residual)

    def test_a_failed_gaussian_fit_with_a_residual_is_refused(self) -> None:
        residual = fake_fit(model="gaussian").residual

        with pytest.raises(ValueError, match="residual"):
            replace(fake_fit(model="gaussian", status=Status.NOT_CONVERGED), residual=residual)

    def test_binomial_diagnostics_on_a_gaussian_fit_are_refused(self) -> None:
        with pytest.raises(ValueError, match="ContinuousDiagnostics"):
            replace(fake_fit(model="gaussian"), diagnostics=fake_fit().diagnostics)

    def test_a_gaussian_fit_with_the_asymptotes_the_wrong_way_round_is_refused(self) -> None:
        fit = fake_fit(model="gaussian")
        assert fit.params is not None
        params = dict(fit.params)
        params["upper"], params["lower"] = params["lower"], params["upper"]

        with pytest.raises(ValueError, match="lower.value < upper.value"):
            replace(fit, params=params)

    def test_a_binomial_asymptote_outside_the_unit_interval_is_still_refused(self) -> None:
        fit = fake_fit()
        assert fit.params is not None
        params = dict(fit.params)
        params["upper"] = replace(params["upper"], value=2.0)

        with pytest.raises(ValueError):
            replace(fit, params=params)


class TestGaussianThresholdInvariants:
    @pytest.mark.parametrize(
        "status", [Status.SEPARATION, Status.FAILS_AT_BASELINE, Status.CONTROL_INCOMPATIBLE]
    )
    def test_a_binomial_only_status_is_refused(self, status: Status) -> None:
        with pytest.raises(ValueError, match="no exact claims"):
            replace(fake_threshold(model="gaussian"), status=status)

    @pytest.mark.parametrize("censoring", [Censoring.RIGHT, Censoring.LEFT])
    def test_exact_bound_censoring_is_refused(self, censoring: Censoring) -> None:
        with pytest.raises(ValueError, match="no exact bound"):
            replace(fake_threshold(model="gaussian"), censoring=censoring)

    def test_a_grid_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be None for a Gaussian fit"):
            replace(fake_threshold(model="gaussian"), grid=(1.0, 2.0))

    def test_a_baseline_is_refused(self) -> None:
        baseline = fake_threshold().baseline
        with pytest.raises(ValueError):
            replace(fake_threshold(model="gaussian"), baseline=baseline or 0.5)

    def test_a_non_converged_threshold_with_a_bound_is_refused(self) -> None:
        with pytest.raises(ValueError):
            replace(fake_threshold(model="gaussian", status=Status.NOT_CONVERGED), lo=0.5, hi=1.0)

    def test_a_value_on_a_failed_gaussian_fit_is_refused(self) -> None:
        failed = fake_fit(model="gaussian", status=Status.NOT_CONVERGED)
        ok = fake_threshold(model="gaussian")

        with pytest.raises(ValueError):
            replace(ok, fit=failed)


def test_welch_pair_with_a_spread_that_underflows_when_squared_does_not_divide_by_zero() -> None:
    """A level whose spread is ~1e-121 squares to a variance that underflows to zero in the
    Welch degrees of freedom. Found by ``tests/property/test_continuous_properties.py``
    (``test_no_number_accompanies_a_failure...``); the check must return a result, not raise."""
    severity = [x for x in _LEVELS for _ in range(2)]
    value = [0.0] * 15 + [8.78457530618143e-121]
    obs = ContinuousObservations.from_observations(
        _LOG, severity=severity, value=value, outcome="response", direction="decreasing"
    )

    check = check_monotonicity(obs)

    assert len(check.pairs) == len(_LEVELS) - 1
    assert all(0.0 <= p.p_value <= 1.0 for p in check.pairs)


def _sparse_profile(*, mu: float, sigma: float, seed: int) -> Fit:
    """Seven levels, one observation each, with noise large enough that the profile F statistic
    does not reach its cut-off on one or both sides before the search limit."""
    levels = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
    rng = np.random.default_rng(seed)
    x = np.asarray(levels)
    f = 1.0 / (1.0 + np.exp(-(np.log(x) - math.log(mu)) / 0.5))
    y = 10.0 - 9.0 * f + rng.normal(0.0, sigma, size=x.size)
    obs = ContinuousObservations.from_observations(
        _LOG, severity=levels, value=y.tolist(), outcome="response", direction="decreasing"
    )
    fit = fit_dose_response(obs, model="gaussian", link="logit", **_ESTIMATE)
    assert fit.status is Status.OK
    return fit


class TestOpenProfileSides:
    @staticmethod
    def _profile(fit: Fit) -> Any:
        return threshold(
            fit, definition=Definition.relative(0.5), interval_method="profile", level=0.95
        )

    def test_a_profile_that_never_crosses_on_the_upper_side_is_open_upper_with_a_lower_bound(
        self,
    ) -> None:
        t = self._profile(_sparse_profile(mu=0.8, sigma=1.5, seed=5))

        assert t.status is Status.OK and t.censoring is Censoring.OPEN_UPPER
        assert t.value is not None and t.lo is not None and t.hi is None
        assert 0.0 < t.lo < t.value
        assert any("upper side left open" in w for w in t.warnings)

    def test_a_profile_that_never_crosses_on_the_lower_side_is_open_lower_with_an_upper_bound(
        self,
    ) -> None:
        t = self._profile(_sparse_profile(mu=0.15, sigma=1.0, seed=0))

        assert t.status is Status.OK and t.censoring is Censoring.OPEN_LOWER
        assert t.value is not None and t.hi is not None and t.lo is None
        assert t.value < t.hi
        assert any("lower side left open" in w for w in t.warnings)

    def test_a_profile_open_on_both_sides_keeps_the_estimate_and_reports_no_bounds(self) -> None:
        t = self._profile(_sparse_profile(mu=0.8, sigma=1.0, seed=1))

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.value is not None and (t.lo, t.hi) == (None, None)
        assert any("left open" in w for w in t.warnings)

    def test_an_open_side_is_never_filled_with_the_search_limit(self) -> None:
        t = self._profile(_sparse_profile(mu=0.8, sigma=1.5, seed=5))

        assert t.hi is None  # not x_max, not a multiple of the value
