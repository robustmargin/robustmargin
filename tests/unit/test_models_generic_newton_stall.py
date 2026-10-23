"""Two defects in the estimated-asymptote path's log-likelihood and convergence verdict.

1. ``_curve_log_success_and_failure_probability`` took ``math.log(upper - lower)``, which raises
   ``ValueError: math domain error`` when float rounding makes ``upper == lower`` (the
   parametrisation ``u = l + (1 - l) * expit(b)`` collapses when ``expit`` saturates), as BFGS's
   line search does when it reaches ``s = inf, upper = lower = 1.0`` on an exact step.
   ``u == l`` is a constant curve ``p = u``; the log-likelihood is evaluated, never raised.

2. Newton's step-size criterion (``tol=1e-10``) is a coin flip at the maximum, where the numerical
   score and Hessian carry roundoff near 1e-9, so the optimizer can stall at the maximum and
   report ``converged=False``. The verdict must come from the optimality of the final point, not
   from that flag. The stall is simulated here, so the tests are deterministic on every platform:
   the Newton stage's result is made to report ``converged=False`` and emit a
   ``ConvergenceWarning`` while its parameters are the true maximum.
"""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from robustmargin import Axis, Observations, Status, fit_dose_response
from robustmargin import models as models_module

_DID_NOT_CONVERGE = "the optimizer did not converge"


@pytest.mark.parametrize("value", [1.0, 0.6])
@pytest.mark.parametrize("s", [0.5, math.inf])
@pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
def test_equal_asymptotes_give_constant_curve_log_probabilities(
    value: float, s: float, link: str
) -> None:
    severity = np.array([1.0, 10.0, 100.0])

    log_p, log_q = models_module._curve_log_success_and_failure_probability(
        severity, mu=-0.124, s=s, upper=value, lower=value, scale="log", link=link
    )

    with np.errstate(divide="ignore"):
        expected_p = np.full(3, math.log(value))
        expected_q = np.full(3, math.log(1.0 - value) if value < 1.0 else -math.inf)
    np.testing.assert_allclose(log_p, expected_p, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(log_q, expected_q)


@pytest.mark.parametrize("value", [1.0, 0.6])
@pytest.mark.parametrize("s", [0.5, math.inf])
def test_equal_asymptotes_log_likelihood_is_a_float_not_an_error(value: float, s: float) -> None:
    severity = np.array([1.0, 10.0, 100.0])
    successes = np.array([5.0, 6.0, 7.0])
    trials = np.array([10.0, 10.0, 10.0])

    value_ll = models_module._full_binomial_log_likelihood(
        severity,
        successes,
        trials,
        mu=-0.124,
        s=s,
        upper=value,
        lower=value,
        scale="log",
        link="probit",
    )

    assert isinstance(value_ll, float)
    # Every cell has failures, so p = 1 contradicts the data; p = 0.6 is merely a poor fit.
    if value == 1.0:
        assert value_ll == -math.inf
    else:
        assert math.isfinite(value_ll)


def test_equal_asymptotes_log_likelihood_is_finite_when_data_agree_with_the_constant() -> None:
    severity = np.array([1.0, 10.0])
    successes = np.array([10.0, 10.0])
    trials = np.array([10.0, 10.0])

    value_ll = models_module._full_binomial_log_likelihood(
        severity,
        successes,
        trials,
        mu=0.0,
        s=math.inf,
        upper=1.0,
        lower=1.0,
        scale="log",
        link="probit",
    )

    assert value_ll == 0.0


_AXIS = Axis(name="concentration", unit="ug/L", scale="log")


def _form_b() -> Observations:
    trials = [141, 153, 142, 139, 154, 155]
    dead = [2, 30, 59, 82, 62, 85]
    return Observations.from_counts(
        _AXIS,
        severity=[0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        successes=[n - d for n, d in zip(trials, dead, strict=True)],
        trials=trials,
        outcome="survival",
        direction="decreasing",
    )


def _fit(obs: Observations, *, lower: Any) -> Any:
    return fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=lower)


def _patch_newton(monkeypatch: pytest.MonkeyPatch, transform: Any) -> None:
    """Replace only the ``method="newton"`` call; BFGS goes through untouched."""
    real_fit = GenericLikelihoodModel.fit

    def fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = real_fit(self, *args, **kwargs)
        if kwargs.get("method") != "newton":
            return result
        return transform(self, result)

    monkeypatch.setattr(GenericLikelihoodModel, "fit", fit)


def _stalled_at_maximum(self: Any, result: Any) -> Any:
    result.mle_retvals["converged"] = False
    warnings.warn("Maximum Likelihood optimization failed to converge.", ConvergenceWarning)
    return result


def _stalled_off_maximum(self: Any, result: Any) -> Any:
    shifted = np.asarray(result.params, dtype=np.float64).copy()
    shifted[0] += 3.0 * float(np.asarray(result.bse)[0])
    return SimpleNamespace(
        params=shifted,
        llf=result.llf,
        mle_retvals={"converged": False},
    )


@pytest.mark.parametrize("lower", [0.0, "estimate"], ids=["upper_estimated", "both_estimated"])
def test_newton_stall_at_the_maximum_is_accepted_as_ok(
    monkeypatch: pytest.MonkeyPatch, lower: Any
) -> None:
    obs = _form_b()
    reference = _fit(obs, lower=lower)
    assert reference.status is Status.OK

    _patch_newton(monkeypatch, _stalled_at_maximum)
    stalled = _fit(obs, lower=lower)

    assert stalled.status is Status.OK
    assert not any(_DID_NOT_CONVERGE in w for w in stalled.warnings)
    for name, param in reference.params.items():
        assert stalled.params[name].value == pytest.approx(param.value, rel=1e-9, abs=1e-12)
    assert stalled.log_likelihood == pytest.approx(reference.log_likelihood, rel=0, abs=1e-9)


@pytest.mark.parametrize("lower", [0.0, "estimate"], ids=["upper_estimated", "both_estimated"])
def test_newton_stall_far_from_the_maximum_stays_not_converged(
    monkeypatch: pytest.MonkeyPatch, lower: Any
) -> None:
    obs = _form_b()

    _patch_newton(monkeypatch, _stalled_off_maximum)
    fit = _fit(obs, lower=lower)

    assert fit.status is Status.NOT_CONVERGED
    assert fit.params is None
    assert any(_DID_NOT_CONVERGE in w for w in fit.warnings)
