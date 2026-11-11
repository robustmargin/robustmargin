"""Fitting the Gaussian LL.4 family: the fit, its residual, its failures.

The model is ``P(x) = u - (u - l) * F(z)`` with Gaussian homoscedastic errors, fitted by
nonlinear least squares. Failures are explicit: a fit that cannot be identified is
``NOT_CONVERGED`` with no parameter, covariance or likelihood, and says why in ``warnings``.
There is no separation, no exact bound and no fallback to another model.

Reasons are named by one lower-case word each in ``warnings``: ``flat``, ``step``, ``boundary``
and ``df``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import least_squares

from robustmargin import (
    Axis,
    ContinuousDiagnostics,
    ContinuousObservations,
    Observations,
    Residual,
    Status,
    fit_dose_response,
)

_LOG = Axis(name="severity", unit="unit", scale="log")
_LINEAR = Axis(name="severity", unit="unit", scale="linear")
_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
_TRUTH = {"mu": math.log(0.8), "s": 0.5, "u": 10.0, "l": 1.0}
_SIGMA = 0.3


def _logistic(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-z)))


def _series(
    seed: int = 20261108, reps: int = 4, sign: float = 1.0
) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    severity = [x for x in _LEVELS for _ in range(reps)]
    x = np.asarray(severity)
    with np.errstate(divide="ignore"):
        z = (np.log(x) - _TRUTH["mu"]) / _TRUTH["s"]
    f = np.where(x == 0.0, 0.0, _logistic(z))
    mean = _TRUTH["u"] - (_TRUTH["u"] - _TRUTH["l"]) * f
    return severity, (sign * (mean + rng.normal(0.0, _SIGMA, size=x.size))).tolist()


def _obs(
    severity: list[float],
    value: list[float],
    *,
    direction: str = "decreasing",
    axis: Axis = _LOG,
) -> ContinuousObservations:
    return ContinuousObservations.from_observations(
        axis, severity=severity, value=value, outcome="response", direction=direction
    )


def _standard() -> ContinuousObservations:
    return _obs(*_series())


class TestAConvergedFit:
    def test_it_is_ok_and_recovers_the_generating_parameters(self) -> None:
        fit = fit_dose_response(
            _standard(), model="gaussian", link="logit", upper="estimate", lower="estimate"
        )

        assert fit.status is Status.OK
        assert fit.model == "gaussian" and fit.link == "logit"
        assert fit.params is not None
        assert fit.params["mu"].value == pytest.approx(_TRUTH["mu"], abs=0.1)
        assert fit.params["s"].value == pytest.approx(_TRUTH["s"], rel=0.2)
        assert fit.params["upper"].value == pytest.approx(_TRUTH["u"], abs=0.3)
        assert fit.params["lower"].value == pytest.approx(_TRUTH["l"], abs=0.3)
        assert not any(p.fixed for p in fit.params.values())

    def test_it_matches_an_independent_least_squares_fit(self) -> None:
        severity, value = _series()
        x, y = np.asarray(severity), np.asarray(value)

        def residuals(p: np.ndarray) -> np.ndarray:
            mu, log_s, low, log_gap = p
            with np.errstate(divide="ignore"):
                z = (np.log(x) - mu) / math.exp(log_s)
            f = np.where(x == 0.0, 0.0, _logistic(z))
            u = low + math.exp(log_gap)
            return np.asarray(u - (u - low) * f - y)

        ref = least_squares(
            residuals, [-0.2, math.log(0.5), 1.0, math.log(9.0)], xtol=1e-14, ftol=1e-14, gtol=1e-14
        )
        fit = fit_dose_response(
            _obs(severity, value),
            model="gaussian",
            link="logit",
            upper="estimate",
            lower="estimate",
        )

        assert fit.params is not None and fit.residual is not None
        assert fit.params["mu"].value == pytest.approx(ref.x[0], rel=1e-5)
        assert fit.params["s"].value == pytest.approx(math.exp(ref.x[1]), rel=1e-5)
        assert fit.params["lower"].value == pytest.approx(ref.x[2], rel=1e-5)
        assert fit.params["upper"].value == pytest.approx(ref.x[2] + math.exp(ref.x[3]), rel=1e-5)
        assert fit.residual.rss == pytest.approx(2.0 * ref.cost, rel=1e-6)

    def test_the_residual_counts_every_row_not_every_cell(self) -> None:
        severity, value = _series(reps=4)
        fit = fit_dose_response(
            _obs(severity, value),
            model="gaussian",
            link="logit",
            upper="estimate",
            lower="estimate",
        )

        assert isinstance(fit.residual, Residual)
        assert fit.residual.df == len(value) - 4
        assert fit.residual.sigma == pytest.approx(
            math.sqrt(fit.residual.rss / fit.residual.df), rel=1e-12
        )
        assert 0.15 < fit.residual.sigma < 0.5

    def test_each_fixed_asymptote_costs_no_degree_of_freedom_and_is_marked_fixed(self) -> None:
        severity, value = _series()
        n = len(value)

        both = fit_dose_response(
            _obs(severity, value), model="gaussian", link="logit", upper=10.0, lower=1.0
        )
        one = fit_dose_response(
            _obs(severity, value), model="gaussian", link="logit", upper=10.0, lower="estimate"
        )

        assert both.residual is not None and one.residual is not None
        assert both.residual.df == n - 2 and one.residual.df == n - 3
        assert both.params is not None and one.params is not None
        assert both.params["upper"].fixed and both.params["lower"].fixed
        assert both.params["upper"].value == 10.0 and both.params["lower"].value == 1.0
        assert one.params["upper"].fixed and not one.params["lower"].fixed
        assert both.covariance is not None and set(both.covariance.names) == {"mu", "s"}

    def test_the_covariance_is_the_residual_variance_times_the_inverse_observed_hessian(
        self,
    ) -> None:
        """``sigma-hat^2 (1/2 d2 RSS)^-1`` at the optimum, with ``sigma-hat^2 = RSS / df``: the
        observed information, which keeps the curvature term that ``sigma^2 (J'J)^-1`` drops. The
        Hessian is taken by central differences of the RSS, independently of the fit."""
        severity, value = _series()
        x = np.asarray(severity)
        y = np.asarray(value)
        fit = fit_dose_response(
            _obs(severity, value), model="gaussian", link="logit", upper=10.0, lower=1.0
        )
        assert fit.params is not None and fit.covariance is not None and fit.residual is not None

        def rss(theta: np.ndarray) -> float:
            with np.errstate(divide="ignore"):
                z = (np.log(x) - theta[0]) / theta[1]
            mean = 10.0 - 9.0 * np.where(x == 0.0, 0.0, _logistic(z))
            return float(((y - mean) ** 2).sum())

        theta0 = np.array([fit.params["mu"].value, fit.params["s"].value])
        h = 1e-4
        hess = np.empty((2, 2))
        for i in range(2):
            for j in range(2):
                ei, ej = np.eye(2)[i] * h, np.eye(2)[j] * h
                hess[i, j] = (
                    rss(theta0 + ei + ej) - rss(theta0 + ei - ej)
                    - rss(theta0 - ei + ej) + rss(theta0 - ei - ej)
                ) / (4.0 * h * h)  # fmt: skip
        sigma2 = fit.residual.rss / fit.residual.df
        expected = sigma2 * np.linalg.inv(hess / 2.0)

        names = list(fit.covariance.names)
        idx = np.ix_([names.index("mu"), names.index("s")], [names.index("mu"), names.index("s")])
        got = np.array(fit.covariance.matrix)[idx]
        np.testing.assert_allclose(got, expected, rtol=1e-3)

    def test_the_gaussian_log_likelihood_is_reported_at_the_mle(self) -> None:
        fit = fit_dose_response(
            _standard(), model="gaussian", link="logit", upper="estimate", lower="estimate"
        )
        assert fit.residual is not None and fit.log_likelihood is not None
        n = len(_series()[1])
        sigma2_mle = fit.residual.rss / n

        expected = -0.5 * n * (math.log(2.0 * math.pi * sigma2_mle) + 1.0)

        assert fit.log_likelihood == pytest.approx(expected, rel=1e-9)

    def test_cells_are_continuous_cells_and_diagnostics_are_continuous(self) -> None:
        fit = fit_dose_response(
            _standard(), model="gaussian", link="logit", upper="estimate", lower="estimate"
        )

        assert [c.severity for c in fit.cells] == _LEVELS
        assert all(c.n == 4 for c in fit.cells)  # type: ignore[union-attr]
        assert isinstance(fit.diagnostics, ContinuousDiagnostics)
        assert fit.diagnostics.rss == pytest.approx(fit.residual.rss, rel=1e-12)  # type: ignore[union-attr]
        assert fit.diagnostics.df == fit.residual.df  # type: ignore[union-attr]
        assert fit.monotonicity is not None

    @pytest.mark.parametrize("link", ["logit", "probit", "cloglog"])
    def test_every_link_converges_on_a_well_identified_series(self, link: str) -> None:
        fit = fit_dose_response(
            _standard(), model="gaussian", link=link, upper="estimate", lower="estimate"
        )

        assert fit.status is Status.OK and fit.link == link

    def test_the_links_are_never_chosen_for_the_caller(self) -> None:
        a = fit_dose_response(
            _standard(), model="gaussian", link="logit", upper="estimate", lower="estimate"
        )
        b = fit_dose_response(
            _standard(), model="gaussian", link="probit", upper="estimate", lower="estimate"
        )

        assert a.link == "logit" and b.link == "probit"
        assert a.params is not None and b.params is not None
        assert a.params["s"].value != b.params["s"].value

    def test_a_linear_axis_fits_with_mu_on_the_severity_scale(self) -> None:
        rng = np.random.default_rng(5)
        severity = [x for x in np.linspace(0.0, 10.0, 8) for _ in range(3)]
        x = np.asarray(severity)
        y = 10.0 - 9.0 * _logistic((x - 5.0) / 1.2) + rng.normal(0.0, 0.1, size=x.size)

        fit = fit_dose_response(
            _obs(severity, y.tolist(), axis=_LINEAR),
            model="gaussian",
            link="logit",
            upper="estimate",
            lower="estimate",
        )

        assert fit.status is Status.OK and fit.params is not None
        assert fit.params["mu"].value == pytest.approx(5.0, abs=0.3)
        assert fit.params["s"].value == pytest.approx(1.2, rel=0.25)

    def test_the_response_may_be_negative(self) -> None:
        severity, value = _series()
        fit = fit_dose_response(
            _obs(severity, [v - 50.0 for v in value]),
            model="gaussian",
            link="logit",
            upper="estimate",
            lower="estimate",
        )

        assert fit.status is Status.OK and fit.params is not None
        assert fit.params["upper"].value == pytest.approx(-40.0, abs=0.3)
        assert fit.params["lower"].value == pytest.approx(-49.0, abs=0.3)


class TestAnIncreasingFit:
    def test_it_is_the_negation_of_the_decreasing_fit(self) -> None:
        severity, value = _series()
        dec = fit_dose_response(
            _obs(severity, value),
            model="gaussian",
            link="logit",
            upper="estimate",
            lower="estimate",
        )
        inc = fit_dose_response(
            _obs(severity, [-v for v in value], direction="increasing"),
            model="gaussian", link="logit", upper="estimate", lower="estimate",
        )  # fmt: skip

        assert inc.direction == "increasing" and inc.status is Status.OK
        assert dec.params is not None and inc.params is not None
        assert inc.params["mu"].value == dec.params["mu"].value
        assert inc.params["lower"].value == -dec.params["upper"].value
        assert inc.params["upper"].value == -dec.params["lower"].value

    def test_lower_is_the_zero_severity_end_and_upper_the_high_severity_end(self) -> None:
        severity, value = _series()
        # Rising from about 1 to about 10: the mirror image, shifted to be positive.
        rising = [11.0 - v for v in value]
        fit = fit_dose_response(
            _obs(severity, rising, direction="increasing"),
            model="gaussian",
            link="logit",
            upper="estimate",
            lower="estimate",
        )

        assert fit.params is not None
        assert fit.params["lower"].value == pytest.approx(1.0, abs=0.3)
        assert fit.params["upper"].value == pytest.approx(10.0, abs=0.3)

    def test_a_fixed_asymptote_follows_the_increasing_convention(self) -> None:
        severity, value = _series()
        fit = fit_dose_response(
            _obs(severity, [11.0 - v for v in value], direction="increasing"),
            model="gaussian", link="logit", upper=10.0, lower="estimate",
        )  # fmt: skip

        assert fit.params is not None
        assert fit.params["upper"].fixed and fit.params["upper"].value == 10.0
        assert not fit.params["lower"].fixed


class TestArgumentChecks:
    def test_continuous_observations_with_a_binomial_model_raise(self) -> None:
        with pytest.raises(ValueError):
            fit_dose_response(_standard(), model="binomial", link="probit", upper=1.0, lower=0.0)

    def test_binomial_observations_with_a_gaussian_model_raise(self) -> None:
        counts = Observations.from_counts(
            _LOG,
            severity=[0.0, 1.0, 2.0, 4.0, 8.0],
            successes=[100, 90, 60, 20, 5],
            trials=[100] * 5,
            outcome="success",
            direction="decreasing",
        )

        with pytest.raises(ValueError):
            fit_dose_response(
                counts, model="gaussian", link="logit", upper="estimate", lower="estimate"
            )

    def test_an_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError):
            fit_dose_response(
                _standard(), model="poisson", link="logit", upper="estimate", lower="estimate"
            )  # type: ignore[call-overload]

    @pytest.mark.parametrize(
        ("upper", "lower"),
        [
            (1.0, 1.0),
            (0.0, 1.0),
            (float("nan"), 1.0),
            (10.0, float("inf")),
            (float("inf"), 1.0),
            (True, 0.0),
        ],
        ids=["equal", "inverted", "nan_upper", "inf_lower", "inf_upper", "bool_upper"],
    )
    def test_fixed_asymptotes_must_be_finite_with_upper_above_lower(
        self, upper: float, lower: float
    ) -> None:
        with pytest.raises(ValueError):
            fit_dose_response(_standard(), model="gaussian", link="logit", upper=upper, lower=lower)

    def test_fixed_asymptotes_outside_the_unit_interval_are_valid_in_response_units(self) -> None:
        fit = fit_dose_response(_standard(), model="gaussian", link="logit", upper=10.0, lower=-3.0)

        assert fit.params is not None and fit.params["lower"].value == -3.0

    def test_an_unknown_link_raises(self) -> None:
        with pytest.raises(ValueError):
            fit_dose_response(
                _standard(), model="gaussian", link="loglog", upper="estimate", lower="estimate"
            )


class TestFailuresAreNamedAndCarryNoNumber:
    def _assert_failed(self, fit: object, reason: tuple[str, ...]) -> None:
        from robustmargin import Fit

        assert isinstance(fit, Fit)
        assert fit.status is Status.NOT_CONVERGED
        assert fit.params is None and fit.covariance is None and fit.log_likelihood is None
        assert fit.diagnostics is None
        text = " ".join(fit.warnings).lower()
        assert any(word in text for word in reason), fit.warnings

    @staticmethod
    def _line_with_replicate_spread() -> tuple[list[float], list[float]]:
        """Level means exactly linear in log severity (so the best logistic is the limit of an
        ever wider one), with a symmetric replicate spread that leaves each level mean intact."""
        severity = [x for x in np.logspace(-1.0, 1.0, 8) for _ in range(4)]
        spread = [0.05, -0.05, 0.02, -0.02] * 8
        value = [10.0 - 0.8 * math.log(x) + d for x, d in zip(severity, spread, strict=True)]
        return severity, value

    @staticmethod
    def _rss_at_scale(severity: list[float], value: list[float], s: float) -> float:
        """Least squares at a fixed scale ``s`` (logit, location and both asymptotes free), by
        an independent multi-start solve. At fixed ``s`` the asymptotes are linear, so they are
        profiled out exactly and only the location is searched."""
        x = np.log(np.asarray(severity))
        y = np.asarray(value)
        best = math.inf
        for mu in np.linspace(-3.0, 3.0, 61) * max(1.0, s / 10.0):
            f = _logistic((x - mu) / s)
            design = np.column_stack([1.0 - f, f])  # y = u (1 - f) + l f
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            best = min(best, float(((y - design @ coef) ** 2).sum()))
        return best

    def test_the_rss_of_an_exact_line_keeps_falling_as_the_scale_grows(self) -> None:
        """The precondition for ``flat``: the supremum is at ``s -> infinity``, not at a finite
        scale. Checked here independently of the fit."""
        severity, value = self._line_with_replicate_spread()
        span = math.log(10.0) - math.log(0.1)

        rss = [self._rss_at_scale(severity, value, span * k) for k in (1.0, 10.0, 100.0, 1000.0)]

        assert all(a > b for a, b in zip(rss, rss[1:], strict=False)), rss

    def test_a_line_exactly_linear_in_log_severity_is_flat(self) -> None:
        severity, value = self._line_with_replicate_spread()

        fit = fit_dose_response(
            _obs(severity, value),
            model="gaussian",
            link="logit",
            upper="estimate",
            lower="estimate",
        )

        self._assert_failed(fit, ("flat",))

    def test_noisy_data_with_a_finite_rss_minimum_in_the_scale_fit_ok_not_flat(self) -> None:
        """A noisy line has a finite best scale (about 12 spans); that is a wide but identified
        curve, left to the profile's open sides, not a ``flat`` failure."""
        rng = np.random.default_rng(1)
        severity = [x for x in np.logspace(-1.0, 1.0, 8) for _ in range(3)]
        value = (10.0 - 0.8 * np.log(severity) + rng.normal(0.0, 0.01, size=len(severity))).tolist()

        fit = fit_dose_response(
            _obs(severity, value),
            model="gaussian",
            link="logit",
            upper="estimate",
            lower="estimate",
        )

        assert fit.status is Status.OK

    def test_a_jump_between_adjacent_levels_is_a_step(self) -> None:
        severity = [x for x in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] for _ in range(3)]
        wobble = [0.05, -0.05, 0.0] * 7
        value = [(10.0 if x <= 3.0 else 2.0) + w for x, w in zip(severity, wobble, strict=True)]

        fit = fit_dose_response(
            _obs(severity, value),
            model="gaussian",
            link="logit",
            upper="estimate",
            lower="estimate",
        )

        self._assert_failed(fit, ("step",))

    def test_a_series_that_rises_against_its_stated_direction_runs_to_the_boundary(self) -> None:
        severity, value = _series()
        rising = [11.0 - v for v in value]

        fit = fit_dose_response(
            _obs(severity, rising),
            model="gaussian",
            link="logit",
            upper="estimate",
            lower="estimate",
        )

        self._assert_failed(fit, ("boundary",))

    def test_a_fixed_upper_asymptote_far_below_the_data_runs_to_the_boundary(self) -> None:
        severity, value = _series()

        fit = fit_dose_response(
            _obs(severity, value), model="gaussian", link="logit", upper=-5.0, lower="estimate"
        )

        self._assert_failed(fit, ("boundary",))

    def test_as_many_free_parameters_as_observations_leaves_no_degrees_of_freedom(self) -> None:
        obs = _obs([0.5, 1.0, 2.0, 4.0], [9.0, 7.0, 3.0, 1.5])

        fit = fit_dose_response(
            obs, model="gaussian", link="logit", upper="estimate", lower="estimate"
        )

        self._assert_failed(fit, ("df", "degrees of freedom"))

    def test_a_failure_is_never_replaced_by_a_different_model(self) -> None:
        obs = _obs([0.5, 1.0, 2.0, 4.0], [9.0, 7.0, 3.0, 1.5])

        fit = fit_dose_response(
            obs, model="gaussian", link="logit", upper="estimate", lower="estimate"
        )

        assert fit.model == "gaussian" and fit.link == "logit"
        assert fit.residual is None or fit.residual.df <= 0

    def test_monotonicity_is_still_computed_on_a_failed_fit(self) -> None:
        obs = _obs([0.5, 1.0, 2.0, 4.0], [9.0, 7.0, 3.0, 1.5])

        fit = fit_dose_response(
            obs, model="gaussian", link="logit", upper="estimate", lower="estimate"
        )

        assert fit.monotonicity is not None
