"""Unit tests for ``Fit.predict()`` (the ``predict`` bands).

Covers band containment, the log-axis zero-severity control, raising on a non-``OK`` fit,
``level`` validation, a hand-computed curve check, and the invariants that apply at the
``Fit``/``Prediction`` level (interval containment and "a wider confidence level never produces
a narrower interval") -- monotonicity is included too, since it is cheap to check directly on a
real fit. The other two invariants (scale invariance of ``value``/``lo``/``hi``, and "censoring
never yields a finite point estimate") are about ``Threshold``, so they are not applicable to
``Prediction`` and live in ``tests/property/test_threshold_properties.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from robustmargin import Axis, Fit, Observations, Prediction, Status, fit_dose_response
from robustmargin.testing import fake_fit

_LOG_AXIS = Axis(name="axis_1", unit="m", scale="log")

# A well-identified decreasing series (the ``system_a_axis_1`` shape from the pooled-counts
# fixture), reused across most of this file.
_SEVERITY = [0.0, 0.01, 0.05, 0.1]
_SUCCESSES = [200, 299, 122, 0]
_TRIALS = [200, 300, 300, 300]


def _fixed_trivial_fit() -> Fit:
    obs = Observations.from_counts(
        _LOG_AXIS,
        severity=_SEVERITY,
        successes=_SUCCESSES,
        trials=_TRIALS,
        outcome="success",
        direction="decreasing",
    )
    return fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)


def _fixed_nontrivial_fit() -> Fit:
    obs = Observations.from_counts(
        _LOG_AXIS,
        severity=[0.0, 1.0, 2.0, 3.0, 4.0],
        successes=[94, 85, 60, 25, 6],
        trials=[100, 100, 100, 100, 100],
        outcome="success",
        direction="decreasing",
    )
    return fit_dose_response(obs, model="binomial", link="probit", upper=0.95, lower=0.05)


def _estimated_asymptote_fit() -> Fit:
    """An identified design (a control plus 7 log-spaced nonzero levels, both asymptotes
    estimated): the band at ``predict(0.0)`` and ``predict(1e-12)`` must agree, since
    both are, to floating-point purposes, the zero-severity control on a log axis.
    """
    obs = Observations.from_counts(
        _LOG_AXIS,
        severity=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
        successes=[180, 170, 150, 110, 60, 30, 22, 20],
        trials=[200, 200, 200, 200, 200, 200, 200, 200],
        outcome="success",
        direction="decreasing",
    )
    return fit_dose_response(
        obs, model="binomial", link="probit", upper="estimate", lower="estimate"
    )


class TestPredictionShape:
    def test_returns_a_prediction_with_numpy_arrays_and_the_given_level(self) -> None:
        fit = _fixed_trivial_fit()

        prediction = fit.predict([0.0, 0.01, 0.05], level=0.9)

        assert isinstance(prediction, Prediction)
        assert isinstance(prediction.severity, np.ndarray)
        assert isinstance(prediction.estimate, np.ndarray)
        assert isinstance(prediction.lo, np.ndarray)
        assert isinstance(prediction.hi, np.ndarray)
        assert prediction.level == 0.9
        assert prediction.severity.shape == prediction.estimate.shape == (3,)


class TestBandContainment:
    def test_lo_le_estimate_le_hi_across_severities(self) -> None:
        fit = _fixed_trivial_fit()

        prediction = fit.predict([0.0, 0.005, 0.01, 0.03, 0.05, 0.08, 0.1], level=0.95)

        assert np.all(prediction.lo <= prediction.estimate)
        assert np.all(prediction.estimate <= prediction.hi)

    def test_band_lies_within_fixed_asymptote_bounds(self) -> None:
        fit = _fixed_nontrivial_fit()

        prediction = fit.predict([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 10.0], level=0.95)

        assert np.all(prediction.lo >= 0.05 - 1e-9)
        assert np.all(prediction.hi <= 0.95 + 1e-9)

    def test_wider_confidence_level_gives_wider_or_equal_band(self) -> None:
        fit = _fixed_trivial_fit()
        severities = [0.005, 0.02, 0.03, 0.07]

        narrow = fit.predict(severities, level=0.80)
        wide = fit.predict(severities, level=0.99)

        assert np.all((wide.hi - wide.lo) >= (narrow.hi - narrow.lo) - 1e-12)


class TestZeroSeverityOnLogAxis:
    def test_estimate_equals_upper_at_zero_severity(self) -> None:
        fit = _fixed_trivial_fit()

        prediction = fit.predict([0.0], level=0.95)

        assert fit.params is not None
        assert prediction.estimate[0] == pytest.approx(fit.params["upper"].value)

    def test_zero_width_band_at_zero_severity_when_upper_is_fixed(self) -> None:
        fit = _fixed_trivial_fit()

        prediction = fit.predict([0.0], level=0.95)

        assert prediction.hi[0] - prediction.lo[0] == pytest.approx(0.0, abs=1e-9)


class TestPredictRaises:
    @pytest.mark.parametrize(
        "status",
        [Status.SEPARATION, Status.NOT_CONVERGED, Status.CONTROL_INCOMPATIBLE],
    )
    def test_raises_value_error_on_a_non_ok_fit(self, status: Status) -> None:
        fit = fake_fit(status=status)

        with pytest.raises(ValueError):
            fit.predict([0.01, 0.05], level=0.95)

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
    def test_raises_value_error_on_invalid_level(self, level: float) -> None:
        fit = _fixed_trivial_fit()

        with pytest.raises(ValueError):
            fit.predict([0.01, 0.05], level=level)


class TestPredictMatchesHandComputedCurve:
    def test_estimate_matches_u_minus_u_l_times_probit_cdf(self) -> None:
        fit = _fixed_trivial_fit()
        assert fit.params is not None
        mu = fit.params["mu"].value
        s = fit.params["s"].value
        upper = fit.params["upper"].value
        lower = fit.params["lower"].value

        severities = np.array([0.005, 0.02, 0.04, 0.09])
        prediction = fit.predict(severities, level=0.95)

        z = (np.log(severities) - mu) / s
        expected = upper - (upper - lower) * norm.cdf(z)

        np.testing.assert_allclose(prediction.estimate, expected, rtol=1e-6, atol=1e-9)


class TestPredictIsMonotone:
    def test_estimate_is_non_increasing_in_severity(self) -> None:
        fit = _fixed_trivial_fit()

        severities = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.5]
        prediction = fit.predict(severities, level=0.95)

        diffs = np.diff(prediction.estimate)
        assert np.all(diffs <= 1e-9)


class TestZeroVsNearZeroSeverityOnLogAxis:
    """On a log axis with ``upper="estimate"`` (so ``P(0) = upper`` is itself an estimated
    quantity, not a fixed one), ``predict(0.0)`` and ``predict(1e-12)`` must agree to within
    about ``1e-6`` -- ``1e-12`` is indistinguishable from the exact zero-severity control at
    floating-point precision (``log(1e-12)`` is a large but finite negative number, and the
    fitted curve's value there is already extremely close to its ``severity=0`` limit for any
    reasonably identified ``(mu, s)``).
    """

    _ABS_TOL = 1e-6

    def test_estimate_agrees_between_zero_and_near_zero_severity(self) -> None:
        fit = _estimated_asymptote_fit()
        assert fit.status is Status.OK

        at_zero = fit.predict([0.0], level=0.95)
        at_near_zero = fit.predict([1e-12], level=0.95)

        assert at_zero.estimate[0] == pytest.approx(at_near_zero.estimate[0], abs=self._ABS_TOL)

    def test_lo_and_hi_agree_between_zero_and_near_zero_severity(self) -> None:
        fit = _estimated_asymptote_fit()
        assert fit.status is Status.OK

        at_zero = fit.predict([0.0], level=0.95)
        at_near_zero = fit.predict([1e-12], level=0.95)

        assert at_zero.lo[0] == pytest.approx(at_near_zero.lo[0], abs=self._ABS_TOL)
        assert at_zero.hi[0] == pytest.approx(at_near_zero.hi[0], abs=self._ABS_TOL)


class TestPredictionArraysAreReadOnly:
    """``Prediction``'s numpy arrays must be read-only, so a caller cannot mutate a
    result in place and mistake it for a fresh prediction.
    """

    def test_assigning_into_estimate_raises_value_error(self) -> None:
        fit = _fixed_trivial_fit()
        prediction = fit.predict([0.0, 0.01], level=0.95)

        with pytest.raises(ValueError):
            prediction.estimate[0] = 1.0

    @pytest.mark.parametrize("field", ["severity", "estimate", "lo", "hi"])
    def test_assigning_into_any_field_raises_value_error(self, field: str) -> None:
        fit = _fixed_trivial_fit()
        prediction = fit.predict([0.0, 0.01], level=0.95)

        array = getattr(prediction, field)
        with pytest.raises(ValueError):
            array[0] = 1.0
