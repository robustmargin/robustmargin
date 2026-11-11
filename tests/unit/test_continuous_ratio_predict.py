"""The ratio of two Gaussian thresholds, the predict band, and the Gaussian test fakes.

* **Ratio.** Two Gaussian thresholds are compared with a Welch-Satterthwaite Student-t, not the
  normal quantile. ``log_delta`` uses the df of ``Va + Vb`` (``Va``, ``Vb`` the variances of the
  log-scale thresholds). ``fieller`` uses the t quantile at the df evaluated at the ratio
  estimate, ``(Va + rho^2 Vb)^2 / (Va^2 / df_a + rho^4 Vb^2 / df_b)`` on natural-scale
  variances. A binomial and a Gaussian threshold are not comparable and raise.
* **Predict.** A Gaussian band is on the response scale (no link transform, not clipped to the
  unit interval) and is ``estimate +- t(df) * se`` by the delta method.
* **Fakes.** ``fake_fit`` and ``fake_threshold`` build schema-valid Gaussian results.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import t as t_dist

from robustmargin import (
    Axis,
    Censoring,
    ContinuousCell,
    ContinuousDiagnostics,
    ContinuousObservations,
    Definition,
    Fit,
    IntervalShape,
    Observations,
    Residual,
    Status,
    Threshold,
    fit_dose_response,
    ratio_interval,
    threshold,
)
from robustmargin.report import to_dict
from robustmargin.testing import fake_fit, fake_threshold

_LOG = Axis(name="severity", unit="unit", scale="log")
_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
_ESTIMATE = {"upper": "estimate", "lower": "estimate"}


def _logistic(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-z)))


def _fit(*, seed: int, reps: int, mu: float = math.log(0.8), **asymptotes: object) -> Fit:
    rng = np.random.default_rng(seed)
    severity = [x for x in _LEVELS for _ in range(reps)]
    x = np.asarray(severity)
    with np.errstate(divide="ignore"):
        z = (np.log(x) - mu) / 0.5
    f = np.where(x == 0.0, 0.0, _logistic(z))
    y = 10.0 - 9.0 * f + rng.normal(0.0, 0.3, size=x.size)
    obs = ContinuousObservations.from_observations(
        _LOG, severity=severity, value=y.tolist(), outcome="response", direction="decreasing"
    )
    fit = fit_dose_response(obs, model="gaussian", link="logit", **{**_ESTIMATE, **asymptotes})  # type: ignore[arg-type]
    assert fit.status is Status.OK
    return fit


def _delta(fit: Fit, p: float = 0.5, level: float = 0.95) -> Threshold:
    t = threshold(fit, definition=Definition.relative(p), interval_method="delta", level=level)
    assert t.status is Status.OK
    return t


def _log_variance(fit: Fit, p: float) -> float:
    """``Var(log x*)`` for ``relative(p)`` under a logit link: gradient ``(1, logit(p))`` in
    ``(mu, s)``."""
    assert fit.covariance is not None
    names = list(fit.covariance.names)
    cov = np.array(fit.covariance.matrix)
    idx = [names.index("mu"), names.index("s")]
    grad = np.array([1.0, math.log(p / (1.0 - p))])
    return float(grad @ cov[np.ix_(idx, idx)] @ grad)


def _df(fit: Fit) -> int:
    assert fit.residual is not None
    return fit.residual.df


class TestWelchSatterthwaiteRatio:
    def test_log_delta_uses_t_at_the_welch_df_of_the_summed_log_variances(self) -> None:
        fit_a, fit_b = _fit(seed=1, reps=3), _fit(seed=2, reps=7, mu=math.log(0.4))
        t_a, t_b = _delta(fit_a), _delta(fit_b)
        assert t_a.value is not None and t_b.value is not None
        va, vb = _log_variance(fit_a, 0.5), _log_variance(fit_b, 0.5)
        df_a, df_b = _df(fit_a), _df(fit_b)
        assert df_a != df_b
        welch = (va + vb) ** 2 / (va**2 / df_a + vb**2 / df_b)

        r = ratio_interval(t_a, t_b, method="log_delta", dependence="independent")

        half = float(t_dist.ppf(0.975, welch)) * math.sqrt(va + vb)
        estimate = t_a.value / t_b.value
        assert r.status is Status.OK and r.shape is IntervalShape.BOUNDED
        assert r.estimate == pytest.approx(estimate, rel=1e-12)
        assert r.lo == pytest.approx(estimate * math.exp(-half), rel=1e-8)
        assert r.hi == pytest.approx(estimate * math.exp(half), rel=1e-8)

    def test_it_is_not_the_normal_interval_nor_the_smaller_df_interval(self) -> None:
        fit_a, fit_b = _fit(seed=1, reps=3), _fit(seed=2, reps=7, mu=math.log(0.4))
        t_a, t_b = _delta(fit_a), _delta(fit_b)
        va, vb = _log_variance(fit_a, 0.5), _log_variance(fit_b, 0.5)
        sigma = math.sqrt(va + vb)

        r = ratio_interval(t_a, t_b, method="log_delta", dependence="independent")

        assert r.estimate is not None and r.hi is not None
        half = math.log(r.hi / r.estimate)
        assert half > float(norm.ppf(0.975)) * sigma * 1.001
        assert half != pytest.approx(
            float(t_dist.ppf(0.975, min(_df(fit_a), _df(fit_b)))) * sigma, rel=1e-4
        )

    def test_two_independent_identical_fits_have_df_twice_their_own(self) -> None:
        fit = _fit(seed=3, reps=3)
        t, t2 = _delta(fit), _delta(_fit(seed=3, reps=3))
        v = _log_variance(fit, 0.5)

        r = ratio_interval(t, t2, method="log_delta", dependence="independent")

        half = float(t_dist.ppf(0.975, 2 * _df(fit))) * math.sqrt(2.0 * v)
        assert r.estimate == pytest.approx(1.0, rel=1e-12)
        assert r.hi == pytest.approx(math.exp(half), rel=1e-8)
        assert r.lo == pytest.approx(math.exp(-half), rel=1e-8)

    def test_fieller_uses_t_at_the_df_evaluated_at_the_ratio_estimate(self) -> None:
        fit_a, fit_b = _fit(seed=4, reps=3), _fit(seed=5, reps=7, mu=math.log(0.4))
        t_a, t_b = _delta(fit_a), _delta(fit_b)
        assert t_a.value is not None and t_b.value is not None
        va = t_a.value**2 * _log_variance(fit_a, 0.5)
        vb = t_b.value**2 * _log_variance(fit_b, 0.5)
        rho = t_a.value / t_b.value
        df_a, df_b = _df(fit_a), _df(fit_b)
        welch = (va + rho**2 * vb) ** 2 / (va**2 / df_a + rho**4 * vb**2 / df_b)
        t2 = float(t_dist.ppf(0.975, welch)) ** 2
        a_coef = t_b.value**2 - t2 * vb
        b_coef = t_a.value * t_b.value
        k_coef = t_a.value**2 - t2 * va
        disc = b_coef**2 - a_coef * k_coef
        assert a_coef > 0.0 and disc > 0.0

        r = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert r.shape is IntervalShape.BOUNDED
        assert r.lo == pytest.approx((b_coef - math.sqrt(disc)) / a_coef, rel=1e-8)
        assert r.hi == pytest.approx((b_coef + math.sqrt(disc)) / a_coef, rel=1e-8)

    def test_the_ratio_records_its_method_level_and_both_thresholds(self) -> None:
        t_a = _delta(_fit(seed=1, reps=3), level=0.9)
        t_b = _delta(_fit(seed=2, reps=3, mu=math.log(0.4)), level=0.9)

        r = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=0.9)

        assert (r.method, r.level, r.dependence) == ("fieller", 0.9, "independent")
        assert r.threshold_a is t_a and r.threshold_b is t_b


class TestAMixedKindRatioIsRefused:
    def _binomial(self) -> Threshold:
        obs = Observations.from_counts(
            _LOG,
            severity=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
            successes=[100, 96, 80, 45, 12, 2],
            trials=[100] * 6,
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        t = threshold(fit, definition=Definition.relative(0.5), interval_method="delta", level=0.95)
        assert t.status is Status.OK
        return t

    @pytest.mark.parametrize("method", ["fieller", "log_delta"])
    @pytest.mark.parametrize("gaussian_first", [True, False])
    def test_a_binomial_and_a_gaussian_threshold_raise(
        self, method: str, gaussian_first: bool
    ) -> None:
        gaussian, binomial = _delta(_fit(seed=1, reps=3)), self._binomial()
        pair = (gaussian, binomial) if gaussian_first else (binomial, gaussian)

        with pytest.raises(ValueError):
            ratio_interval(*pair, method=method, dependence="independent")


class TestPredictBand:
    @staticmethod
    def _gradient(fit: Fit, x: float) -> dict[str, float]:
        assert fit.params is not None
        mu, s = fit.params["mu"].value, fit.params["s"].value
        u, low = fit.params["upper"].value, fit.params["lower"].value
        if x == 0.0:
            return {"mu": 0.0, "s": 0.0, "upper": 1.0, "lower": 0.0}
        z = (math.log(x) - mu) / s
        cdf = 1.0 / (1.0 + math.exp(-z))
        pdf = cdf * (1.0 - cdf)
        return {
            "mu": (u - low) * pdf / s,
            "s": (u - low) * pdf * z / s,
            "upper": 1.0 - cdf,
            "lower": cdf,
        }

    def _expected(self, fit: Fit, x: float, level: float) -> tuple[float, float, float]:
        assert fit.params is not None and fit.covariance is not None and fit.residual is not None
        grad = self._gradient(fit, x)
        names = list(fit.covariance.names)
        g = np.array([grad[n] for n in names])
        se = math.sqrt(float(g @ np.array(fit.covariance.matrix) @ g))
        mu, s = fit.params["mu"].value, fit.params["s"].value
        u, low = fit.params["upper"].value, fit.params["lower"].value
        z = 0.0 if x == 0.0 else (math.log(x) - mu) / s
        cdf = 0.0 if x == 0.0 else 1.0 / (1.0 + math.exp(-z))
        estimate = u - (u - low) * cdf
        half = float(t_dist.ppf(0.5 + level / 2.0, fit.residual.df)) * se
        return estimate, estimate - half, estimate + half

    @pytest.mark.parametrize(
        "asymptotes", [{}, {"upper": 10.0, "lower": 1.0}], ids=["free", "fixed"]
    )
    @pytest.mark.parametrize("level", [0.90, 0.95])
    def test_the_band_is_the_delta_method_on_the_response_scale_with_t_quantiles(
        self, asymptotes: dict[str, float], level: float
    ) -> None:
        fit = _fit(seed=7, reps=4, **asymptotes)
        grid = [0.0, 0.15, 0.5, 0.8, 2.0, 6.0]

        band = fit.predict(grid, level=level)

        for i, x in enumerate(grid):
            estimate, lo, hi = self._expected(fit, x, level)
            assert band.estimate[i] == pytest.approx(estimate, rel=1e-9)
            assert band.lo[i] == pytest.approx(lo, rel=1e-8, abs=1e-12)
            assert band.hi[i] == pytest.approx(hi, rel=1e-8, abs=1e-12)
        assert band.level == level

    def test_the_band_is_symmetric_about_the_estimate_and_not_squeezed_into_the_unit_interval(
        self,
    ) -> None:
        fit = _fit(seed=7, reps=4)

        band = fit.predict([0.0, 0.4, 0.8, 1.6])

        np.testing.assert_allclose(band.hi - band.estimate, band.estimate - band.lo, rtol=1e-9)
        assert float(band.estimate.max()) > 5.0  # response units, not a probability

    def test_a_fixed_asymptote_band_has_no_width_at_the_control(self) -> None:
        fit = _fit(seed=7, reps=4, upper=10.0, lower=1.0)

        band = fit.predict(0.0)

        assert float(band.lo) == float(band.hi) == pytest.approx(10.0, rel=1e-12)


class TestGaussianFakes:
    @pytest.mark.parametrize("direction", ["decreasing", "increasing"])
    def test_fake_fit_is_a_gaussian_fit_with_continuous_parts(self, direction: str) -> None:
        fit = fake_fit(model="gaussian", direction=direction)

        assert fit.model == "gaussian" and fit.direction == direction
        assert fit.status is Status.OK and fit.params is not None
        assert isinstance(fit.residual, Residual)
        assert fit.cells and all(isinstance(c, ContinuousCell) for c in fit.cells)
        assert isinstance(fit.diagnostics, ContinuousDiagnostics)
        to_dict(fit)

    def test_the_default_fake_is_still_binomial_without_a_residual(self) -> None:
        fit = fake_fit()

        assert fit.model == "binomial" and fit.residual is None

    @pytest.mark.parametrize("status", [Status.NOT_CONVERGED])
    def test_a_failed_gaussian_fake_has_no_numbers(self, status: Status) -> None:
        fit = fake_fit(model="gaussian", status=status)

        assert fit.status is status
        assert fit.params is None and fit.covariance is None and fit.log_likelihood is None

    @pytest.mark.parametrize("direction", ["decreasing", "increasing"])
    def test_fake_threshold_is_a_gaussian_threshold(self, direction: str) -> None:
        t = fake_threshold(model="gaussian", direction=direction)

        assert t.fit.model == "gaussian" and t.fit.residual is not None
        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.lo is not None and t.value is not None and t.hi is not None
        assert t.lo <= t.value <= t.hi
        to_dict(t)

    def test_fake_threshold_for_a_failed_gaussian_fit_has_no_numbers(self) -> None:
        t = fake_threshold(model="gaussian", status=Status.NOT_CONVERGED)

        assert t.status is Status.NOT_CONVERGED
        assert (t.value, t.lo, t.hi) == (None, None, None)

    def test_an_unknown_fake_model_raises(self) -> None:
        with pytest.raises(ValueError):
            fake_fit(model="poisson")
        with pytest.raises(ValueError):
            fake_threshold(model="poisson")
