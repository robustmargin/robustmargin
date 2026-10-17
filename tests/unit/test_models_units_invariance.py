"""Units invariance for ``fit_dose_response`` (a regression test on the linear axis, plus the
estimated-asymptote path's version of the log-axis property already covered for the
fixed-asymptote path in ``tests/property/test_models_properties.py``).

**Linear axis.** Rescaling severity by ``x -> c*x`` is a pure reparametrisation: ``z = (x -
mu)/s`` is unchanged if ``mu`` and ``s`` both scale by exactly ``c``. This differs from the log
axis, where the same rescaling only *shifts* ``mu`` by ``log(c)`` and leaves ``s`` unchanged
(``log(c*x) = log(c) + log(x)``) -- the two axes are genuinely different invariances, not the
same check with a different label.

``c=1e6`` on the linear axis guards against a ``Covariance`` positive-definiteness rule that is
sensitive to a matrix's *raw* eigenvalue scale rather than its correlation structure, which
would turn a perfectly good fit into ``NOT_CONVERGED`` on both the fixed and the
estimated-asymptote paths. The rule is units-independent, as covered in
``tests/unit/test_results.py``'s ``TestCovarianceMixedScaleAcceptance`` and
``TestCovarianceAcceptsEveryMatrixTheRawEigenvalueRuleAccepts``. ``c=1e-6`` on the linear-axis
estimated-asymptote path is exercised here for the same reason.
"""

from __future__ import annotations

import math

import pytest

from robustmargin import Axis, Fit, Observations, Status, fit_dose_response

_LINEAR_AXIS = Axis(name="severity", unit="unit", scale="linear")
_LOG_AXIS = Axis(name="severity", unit="unit", scale="log")

# 7 levels, c * {0, 1, ..., 6}: a well-identified decreasing curve under fixed upper=1, lower=0.
_FIXED_LEVELS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
_FIXED_SUCCESSES = (190, 185, 160, 100, 45, 25, 20)
_FIXED_TRIALS = (200,) * 7

# 10 levels, c * {0, ..., 9}: a well-identified decreasing curve with both a ceiling (~0.9, not
# 1.0) and a floor (~0.1, not 0.0), so both asymptotes are actually identifiable from the data
# rather than resting entirely on the parametrisation's own boundaries.
_ESTIMATED_LEVELS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
_ESTIMATED_SUCCESSES = (180, 175, 160, 130, 95, 60, 35, 25, 21, 20)
_ESTIMATED_TRIALS = (200,) * 10

_REL_TOL_FIXED = 1e-8
_REL_TOL_ESTIMATED = 1e-6


def _linear_fixed_fit(c: float) -> Fit:
    obs = Observations.from_counts(
        _LINEAR_AXIS,
        severity=[c * level for level in _FIXED_LEVELS],
        successes=_FIXED_SUCCESSES,
        trials=_FIXED_TRIALS,
        outcome="success",
        direction="decreasing",
    )
    return fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)


def _linear_estimated_fit(c: float) -> Fit:
    obs = Observations.from_counts(
        _LINEAR_AXIS,
        severity=[c * level for level in _ESTIMATED_LEVELS],
        successes=_ESTIMATED_SUCCESSES,
        trials=_ESTIMATED_TRIALS,
        outcome="success",
        direction="decreasing",
    )
    return fit_dose_response(
        obs, model="binomial", link="probit", upper="estimate", lower="estimate"
    )


class TestLinearAxisUnitsInvarianceFixedAsymptotes:
    """``upper=1.0, lower=0.0`` fixed (the GLM path): rescaling by ``c`` must scale both ``mu``
    and ``s`` by exactly ``c``, to a tight ``1e-8`` relative tolerance -- this is an exact
    reparametrisation of a converged GLM fit, not an approximation.
    """

    @pytest.mark.parametrize("c", [1e-4, 1.0, 1e6])
    def test_status_is_ok(self, c: float) -> None:
        fit = _linear_fixed_fit(c)

        assert fit.status is Status.OK

    @pytest.mark.parametrize("c", [1e-4, 1e6])
    def test_mu_and_s_scale_by_exactly_c(self, c: float) -> None:
        reference = _linear_fixed_fit(1.0)
        scaled = _linear_fixed_fit(c)

        assert reference.params is not None
        assert scaled.params is not None
        assert scaled.params["mu"].value == pytest.approx(
            c * reference.params["mu"].value, rel=_REL_TOL_FIXED
        )
        assert scaled.params["s"].value == pytest.approx(
            c * reference.params["s"].value, rel=_REL_TOL_FIXED
        )


class TestLinearAxisUnitsInvarianceEstimatedAsymptotes:
    """``upper="estimate", lower="estimate"`` (the ``GenericLikelihoodModel`` path): the same
    scale invariance, at a looser ``1e-6`` relative tolerance since two different optimizer
    runs (one per ``c``) are being compared rather than a closed-form GLM reparametrisation.
    ``upper``/``lower`` themselves are dimensionless probabilities and must come out identical
    (not merely close) across ``c``, since severity scaling cannot change a rate.

    ``c=1e-6`` is included alongside ``1e-4, 1, 1e6``: the ``Covariance`` positive-definiteness
    rule that would make ``c=1e6`` fail if it were scale-sensitive is units-independent
    (``tests/unit/test_results.py``), and this checks that this also covers the small-c end.
    """

    @pytest.mark.parametrize("c", [1e-6, 1e-4, 1.0, 1e6])
    def test_status_is_ok(self, c: float) -> None:
        fit = _linear_estimated_fit(c)

        assert fit.status is Status.OK

    @pytest.mark.parametrize("c", [1e-6, 1e-4, 1e6])
    def test_mu_and_s_scale_by_exactly_c(self, c: float) -> None:
        reference = _linear_estimated_fit(1.0)
        scaled = _linear_estimated_fit(c)

        assert reference.status is Status.OK
        assert scaled.status is Status.OK
        assert reference.params is not None
        assert scaled.params is not None
        assert scaled.params["mu"].value == pytest.approx(
            c * reference.params["mu"].value, rel=_REL_TOL_ESTIMATED
        )
        assert scaled.params["s"].value == pytest.approx(
            c * reference.params["s"].value, rel=_REL_TOL_ESTIMATED
        )

    @pytest.mark.parametrize("c", [1e-6, 1e-4, 1e6])
    def test_upper_and_lower_are_unchanged_by_severity_scaling(self, c: float) -> None:
        reference = _linear_estimated_fit(1.0)
        scaled = _linear_estimated_fit(c)

        assert reference.params is not None
        assert scaled.params is not None
        assert scaled.params["upper"].value == pytest.approx(
            reference.params["upper"].value, rel=_REL_TOL_ESTIMATED
        )
        assert scaled.params["lower"].value == pytest.approx(
            reference.params["lower"].value, rel=_REL_TOL_ESTIMATED
        )


class TestLogAxisUnitsInvarianceEstimatedAsymptotes:
    """The log-axis property (``x -> c*x`` shifts ``mu`` by ``log(c)`` and leaves ``s``
    unchanged) is already covered for the fixed-asymptote path by
    ``tests/property/test_models_properties.py``'s
    ``test_rescaling_severity_shifts_mu_by_log_c_and_leaves_s_unchanged``. This is the
    estimated-asymptote path's version of the same check, at ``c=1e6`` on an identified design
    (a zero control plus 7 log-spaced nonzero levels, both asymptotes estimated).
    """

    _SEVERITY = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
    _SUCCESSES = (180, 170, 150, 110, 60, 30, 22, 20)
    _TRIALS = (200,) * 8
    _C = 1e6

    @staticmethod
    def _fit(severity: tuple[float, ...]) -> Fit:
        obs = Observations.from_counts(
            _LOG_AXIS,
            severity=list(severity),
            successes=list(TestLogAxisUnitsInvarianceEstimatedAsymptotes._SUCCESSES),
            trials=list(TestLogAxisUnitsInvarianceEstimatedAsymptotes._TRIALS),
            outcome="success",
            direction="decreasing",
        )
        return fit_dose_response(
            obs, model="binomial", link="probit", upper="estimate", lower="estimate"
        )

    def test_mu_shifts_by_log_c_and_s_is_unchanged(self) -> None:
        # severity=0.0 (the log-axis control) is left at exactly 0.0 under rescaling: 0 * c ==
        # 0, and it is still the same control row either way.
        original = self._fit(self._SEVERITY)
        rescaled = self._fit(
            tuple(0.0 if level == 0.0 else self._C * level for level in self._SEVERITY)
        )

        assert original.status is Status.OK
        assert rescaled.status is Status.OK
        assert original.params is not None
        assert rescaled.params is not None

        assert rescaled.params["mu"].value == pytest.approx(
            original.params["mu"].value + math.log(self._C), rel=1e-6, abs=1e-9
        )
        assert rescaled.params["s"].value == pytest.approx(
            original.params["s"].value, rel=1e-6, abs=1e-9
        )
