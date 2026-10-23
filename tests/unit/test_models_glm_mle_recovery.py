"""A binomial fit with ``upper=1, lower=0`` lands on the true maximum-likelihood estimate.

Regression for a defect in the fixed-asymptote fit of the 0.2.0 release. Two failures shared one
cause family:

* the log-likelihood formed ``p = u - (u - l) * F(z)`` by subtraction, so in a tail it was
  exactly ``0`` (log-likelihood ``-inf``) or lost its relative precision, even at the true
  maximum;
* statsmodels' clipped IRLS reports "converged" at points that are not the maximum (released
  numbers up to about 0.28 SE off in ``mu``), and reports non-convergence on data whose
  maximum is finite and unique (the links are log-concave and the cells overlap).

The expected values here come from an independent maximisation done in this file: a log-space
binomial log-likelihood, maximised with :func:`scipy.optimize.minimize` (Nelder-Mead then BFGS)
from twelve starts, never from the library. The log-likelihood is compared with the full
binomial log-likelihood, ``lchoose`` term included, which is what ``Fit.log_likelihood`` reports.

The library's own subtractive log-likelihood is *not* a valid reference: at the released
``mu = 5.7434`` it reports ``-389.57`` for the first data set, which is higher than the true maximum
``-391.20``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import gammaln, log_expit
from scipy.stats import chi2, norm

from robustmargin import (
    Axis,
    Censoring,
    Definition,
    Observations,
    Status,
    fit_dose_response,
    threshold,
)
from robustmargin import models as _models

_LINEAR = Axis(name="severity", unit="unit", scale="linear")
_LOG = Axis(name="severity", unit="unit", scale="log")

_RICH_SEVERITY = [0.0, *np.logspace(-1.0, 1.0, 10).tolist()]
_RICH_SUCCESSES = [400, 400, 400, 395, 381, 264, 121, 16, 2, 0, 1]
_RICH_TRIALS = [400] * 11

Link = Literal["probit", "logit", "cloglog"]
_LINKS: tuple[Link, ...] = ("probit", "logit", "cloglog")


def _obs(
    axis: Axis, severity: list[float], successes: list[int], trials: list[int]
) -> Observations:
    return Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )


def _minimal() -> Observations:
    return _obs(_LINEAR, [1.0, 2.0, 4.0, 8.0], [395, 381, 16, 1], [400] * 4)


def _rich() -> Observations:
    return _obs(_LOG, _RICH_SEVERITY, _RICH_SUCCESSES, _RICH_TRIALS)


def _c1() -> Observations:
    return _obs(_LINEAR, [1.0, 2.0, 4.0, 8.0, 16.0], [999, 841, 398, 52, 3], [1000] * 5)


def _probit_case() -> Observations:
    return _obs(_LINEAR, [1.0, 2.0, 4.0, 8.0, 16.0], [995, 651, 201, 12, 1], [1000] * 5)


# --- Independent maximisation ------------------------------------------------------------------


def _log_survival_and_cdf(
    z: NDArray[np.float64], link: Link
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(log S(z), log F(z))`` in log space, with ``p = S(z)`` the success probability."""
    if link == "probit":
        return norm.logsf(z), norm.logcdf(z)
    if link == "logit":
        return log_expit(-z), log_expit(z)
    return -np.exp(z), np.log(-np.expm1(-np.exp(z)))


def _log_likelihood(
    mu: float,
    s: float,
    t: NDArray[np.float64],
    k: NDArray[np.float64],
    n: NDArray[np.float64],
    link: Link,
) -> float:
    """Binomial log-likelihood, ``upper=1``, ``lower=0``, without the ``lchoose`` term."""
    log_p, log_q = _log_survival_and_cdf((t - mu) / s, link)
    return float(np.sum(k * log_p + (n - k) * log_q))


def _lchoose(k: NDArray[np.float64], n: NDArray[np.float64]) -> float:
    return float(np.sum(gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)))


_MLE_CACHE: dict[object, tuple[float, float, float]] = {}


def _independent_mle(obs: Observations, link: Link) -> tuple[float, float, float]:
    key = (obs.axis.scale, link, tuple((c.severity, c.successes, c.trials) for c in obs.cells()))
    if key not in _MLE_CACHE:
        _MLE_CACHE[key] = _compute_independent_mle(obs, link)
    return _MLE_CACHE[key]


def _compute_independent_mle(obs: Observations, link: Link) -> tuple[float, float, float]:
    """``(mu, s, log-likelihood)`` at the maximum, with ``lchoose`` included.

    A log-axis zero-severity control has fitted probability exactly ``u = 1`` and so contributes
    ``0`` for every parameter; it is dropped from the maximisation and added back (as ``0``) by
    ``lchoose``, which is ``0`` for an all-success cell.
    """
    cells = obs.cells()
    x = np.array([c.severity for c in cells], dtype=np.float64)
    k = np.array([c.successes for c in cells], dtype=np.float64)
    n = np.array([c.trials for c in cells], dtype=np.float64)
    keep = x > 0.0 if obs.axis.scale == "log" else np.ones_like(x, dtype=bool)
    t = np.log(x[keep]) if obs.axis.scale == "log" else x[keep]

    def negative(theta: NDArray[np.float64]) -> float:
        return -_log_likelihood(
            float(theta[0]), math.exp(float(theta[1])), t, k[keep], n[keep], link
        )

    best = None
    for mu0 in np.linspace(t.min(), t.max(), 4):
        for s0 in (0.3, 1.0, 3.0):
            first = minimize(
                negative,
                [mu0, math.log(s0)],
                method="Nelder-Mead",
                options={"xatol": 1e-12, "fatol": 1e-13, "maxiter": 20000},
            )
            second = minimize(negative, first.x, method="BFGS", options={"gtol": 1e-10})
            if best is None or second.fun < best.fun - 1e-12:
                best = second
    assert best is not None
    return (
        float(best.x[0]),
        math.exp(float(best.x[1])),
        -float(best.fun) + _lchoose(k, n),
    )


_CASES: dict[str, tuple[Callable[[], Observations], Link]] = {
    "minimal-cloglog": (_minimal, "cloglog"),
    "rich-cloglog": (_rich, "cloglog"),
    "c1-cloglog": (_c1, "cloglog"),
    "probit": (_probit_case, "probit"),
}

# Values an independent reviewer derived by a separate maximisation. They are cross-checks on
# the in-test maximiser, never the thing the library is compared against.
_REVIEWER = {
    "minimal-cloglog": (3.6346, 1.0838, -179.23),
    "rich-cloglog": (0.2831, 0.5320, -102.24),
    "c1-cloglog": (5.7516, None, None),
}


@pytest.mark.parametrize("name", list(_REVIEWER))
def test_in_test_maximiser_agrees_with_reviewer_values(name: str) -> None:
    """Pins the reference itself: the in-test maximisation reproduces the reviewer's numbers to
    their printed precision, so the values the fits are compared with are not self-referential."""
    build, link = _CASES[name]
    mu, s, llf = _independent_mle(build(), link)
    r_mu, r_s, r_llf = _REVIEWER[name]

    assert mu == pytest.approx(r_mu, abs=6e-5)
    if r_s is not None:
        assert s == pytest.approx(r_s, abs=6e-5)
    if r_llf is not None:
        assert llf == pytest.approx(r_llf, abs=6e-3)


@pytest.fixture(scope="module")
def mle() -> dict[str, tuple[float, float, float]]:
    return {name: _independent_mle(build(), link) for name, (build, link) in _CASES.items()}


@pytest.mark.parametrize("name", list(_CASES))
def test_fit_returns_ok_at_the_independent_maximum(
    name: str, mle: dict[str, tuple[float, float, float]]
) -> None:
    """Pins: status ``OK``, parameters and log-likelihood equal to the independent maximisation
    (``mu``, ``s`` to 1e-6 relative, log-likelihood to 1e-6 absolute), log-likelihood finite."""
    build, link = _CASES[name]
    mu, s, llf = mle[name]

    fit = fit_dose_response(build(), model="binomial", link=link, upper=1.0, lower=0.0)

    assert fit.status is Status.OK
    assert fit.params is not None
    assert fit.params["mu"].value == pytest.approx(mu, rel=1e-6)
    assert fit.params["s"].value == pytest.approx(s, rel=1e-6)
    assert fit.log_likelihood is not None and math.isfinite(fit.log_likelihood)
    assert fit.log_likelihood == pytest.approx(llf, abs=1e-6)


def test_released_wrong_cloglog_estimate_is_not_returned(
    mle: dict[str, tuple[float, float, float]],
) -> None:
    """Pins the first data set: the clipped IRLS point ``mu = 5.7434`` (reported ``OK`` by 0.2.0)
    is 0.008 away from the maximum ``5.7516``; the fit must be at the maximum, not there."""
    mu_true = mle["c1-cloglog"][0]
    fit = fit_dose_response(_c1(), model="binomial", link="cloglog", upper=1.0, lower=0.0)

    assert fit.params is not None
    assert abs(fit.params["mu"].value - mu_true) < 1e-4
    assert abs(fit.params["mu"].value - 5.7434251103) > 5e-3


@pytest.mark.parametrize("name", list(_CASES))
def test_replaced_glm_estimate_carries_a_direct_likelihood_warning(name: str) -> None:
    """Pins: the estimate comes from direct likelihood maximisation of the same model and the
    result says so. Anchored on the words "direct" and "likelihood"; adjust to the final text."""
    build, link = _CASES[name]

    fit = fit_dose_response(build(), model="binomial", link=link, upper=1.0, lower=0.0)

    assert fit.status is Status.OK
    assert any("direct" in w.lower() and "likelihood" in w.lower() for w in fit.warnings)


@pytest.mark.parametrize("link", ["probit", "logit"])
def test_comparison_link_cloglog_is_ok_with_the_aic_of_the_maximum(
    link: Link, mle: dict[str, tuple[float, float, float]]
) -> None:
    """Pins: through ``diagnose`` on a probit or logit primary, the cloglog comparison link is
    ``OK`` and its AIC equals ``-2 * llf_mle + 2 * k`` with ``k = 2``."""
    fit = fit_dose_response(_minimal(), model="binomial", link=link, upper=1.0, lower=0.0)

    assert fit.diagnostics is not None
    assert fit.diagnostics.link_status["cloglog"] is Status.OK
    aic = fit.diagnostics.link_aic["cloglog"]
    assert aic is not None and math.isfinite(aic)
    assert aic == pytest.approx(-2.0 * mle["minimal-cloglog"][2] + 4.0, abs=1e-5)


# --- The defensive non-finite guard ------------------------------------------------------------


def test_nonfinite_log_likelihood_on_the_generic_path_is_a_flagged_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the defensive ``_fit_generic`` guard (an asymptote fixed at a value other than 1 or
    0; here ``lower=0.01``, which fits ``OK`` without the patch).

    On valid data the guard is unreachable, so it is pinned with a targeted monkeypatch rather
    than natural data: the full binomial log-likelihood returns ``-inf`` only for the final
    reporting call, the one made without the optimizer's ``center``/``spread`` keywords. The
    optimizer's own iterations are untouched. The result must be ``NOT_CONVERGED`` with no
    numbers and a warning that says the log-likelihood is not finite.
    """
    real = _models._full_binomial_log_likelihood

    def reporting_call_is_nonfinite(*args: object, **kwargs: object) -> float:
        if "center" not in kwargs:
            return -math.inf
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_models, "_full_binomial_log_likelihood", reporting_call_is_nonfinite)

    fit = fit_dose_response(_rich(), model="binomial", link="probit", upper=1.0, lower=0.01)

    assert fit.status is Status.NOT_CONVERGED
    assert fit.params is None
    assert fit.covariance is None
    assert fit.log_likelihood is None
    assert any("not finite" in w for w in fit.warnings)


def test_nonfinite_log_likelihood_on_the_glm_path_is_a_flagged_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the defensive ``_fit_glm`` guard the same way: only the final reporting call (no
    ``center`` keyword) is forced to ``-inf``; the result is ``NOT_CONVERGED`` with no numbers
    and a warning that says the log-likelihood is not finite."""
    real = _models._full_binomial_log_likelihood

    def reporting_call_is_nonfinite(*args: object, **kwargs: object) -> float:
        if "center" not in kwargs:
            return -math.inf
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_models, "_full_binomial_log_likelihood", reporting_call_is_nonfinite)

    fit = fit_dose_response(_c1(), model="binomial", link="probit", upper=1.0, lower=0.0)

    assert fit.status is Status.NOT_CONVERGED
    assert fit.params is None
    assert fit.log_likelihood is None
    assert any("not finite" in w for w in fit.warnings)


# --- The log-likelihood helper at extreme z ----------------------------------------------------

_Z = 40.0


def _probit_log_tail(z: float) -> float:
    """``log(1 - Phi(z))`` for large ``z``, from the asymptotic series of Mills' ratio."""
    series = 1.0 - 1.0 / z**2 + 3.0 / z**4 - 15.0 / z**6 + 105.0 / z**8
    return -(z**2) / 2.0 - math.log(z) - 0.5 * math.log(2.0 * math.pi) + math.log(series)


def _reference_log_p_and_q(link: Link, z: float) -> tuple[float, float]:
    """Closed-form ``(log p, log(1 - p))`` for ``p = S(z)``, ``|z| = 40``, ``u=1``, ``l=0``.
    The complement of a doubly-exponentially small tail is ``0`` to double precision."""
    if link == "probit":
        tail = _probit_log_tail(_Z)
        return (tail, 0.0) if z > 0 else (0.0, tail)
    if link == "logit":
        shift = math.log1p(math.exp(-_Z))
        return (-_Z - shift, -shift) if z > 0 else (-shift, -_Z - shift)
    if z > 0:
        return -math.exp(_Z), 0.0
    return -math.exp(-_Z), -_Z - math.exp(-_Z) / 2.0


@pytest.mark.parametrize("z", [_Z, -_Z])
@pytest.mark.parametrize("link", _LINKS)
@pytest.mark.parametrize("success", [True, False])
def test_log_likelihood_is_finite_and_accurate_at_extreme_z(
    link: Link, z: float, success: bool
) -> None:
    """Pins: one cell at ``z = +-40`` (``mu=0``, ``s=1``), one trial, gives a finite
    log-likelihood equal to ``log p`` (success) or ``log(1 - p)`` (failure) from closed forms.
    The subtractive ``u - (u - l) * F`` loses either one to an exact ``0`` or ``1``."""
    log_p, log_q = _reference_log_p_and_q(link, z)
    value = _models._full_binomial_log_likelihood(
        np.array([z]),
        np.array([1.0 if success else 0.0]),
        np.array([1.0]),
        mu=0.0,
        s=1.0,
        upper=1.0,
        lower=0.0,
        scale="linear",
        link=link,
    )

    assert math.isfinite(value)
    assert value == pytest.approx(log_p if success else log_q, rel=1e-9, abs=1e-12)


# --- Property: every OK fit is at a stationary point -------------------------------------------


def _numerical_score(
    mu: float,
    s: float,
    t: NDArray[np.float64],
    k: NDArray[np.float64],
    n: NDArray[np.float64],
    link: Link,
) -> tuple[float, float]:
    h_mu = 1e-5 * (1.0 + abs(mu))
    h_s = 1e-5 * (1.0 + abs(s))
    d_mu = (
        _log_likelihood(mu + h_mu, s, t, k, n, link) - _log_likelihood(mu - h_mu, s, t, k, n, link)
    ) / (2.0 * h_mu)
    d_s = (
        _log_likelihood(mu, s + h_s, t, k, n, link) - _log_likelihood(mu, s - h_s, t, k, n, link)
    ) / (2.0 * h_s)
    return d_mu, d_s


@settings(
    max_examples=60,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    link=st.sampled_from(_LINKS),
    mu=st.floats(min_value=2.0, max_value=7.0),
    s=st.floats(min_value=0.4, max_value=3.0),
    trials=st.sampled_from([100, 400, 1000]),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_every_ok_fit_has_a_vanishing_score_at_the_returned_parameters(
    link: Link, mu: float, s: float, trials: int, seed: int
) -> None:
    """Property: whenever the fit is ``OK`` the numerical score of the stable log-likelihood at
    the returned ``(mu, s)`` is within ``1e-4 * (1 + |llf|)``, the same relative bound the
    library's own check uses. Data are binomial draws from a random curve (seeded by the drawn
    ``seed``); a non-``OK`` fit is a legitimate outcome and is not asserted on here."""
    severity = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    p = np.asarray(
        np.exp(_log_survival_and_cdf((severity - mu) / s, link)[0]),
        dtype=np.float64,
    )
    counts = np.random.default_rng(seed).binomial(trials, p)
    fit = fit_dose_response(
        _obs(_LINEAR, severity.tolist(), counts.tolist(), [trials] * 5),
        model="binomial",
        link=link,
        upper=1.0,
        lower=0.0,
    )
    if fit.status is not Status.OK:
        return

    assert fit.params is not None and fit.log_likelihood is not None
    fitted_mu = fit.params["mu"].value
    fitted_s = fit.params["s"].value
    k = counts.astype(np.float64)
    n = np.full(5, float(trials))
    score = _numerical_score(fitted_mu, fitted_s, severity, k, n, link)
    llf = _log_likelihood(fitted_mu, fitted_s, severity, k, n, link) + _lchoose(k, n)

    assert fit.log_likelihood == pytest.approx(llf, abs=1e-6 * (1.0 + abs(llf)))
    assert max(abs(score[0]), abs(score[1])) <= 1e-4 * (1.0 + abs(llf))


# --- Trigger-specific warnings -----------------------------------------------------------------
# One stable, mutually exclusive lower-case substring per trigger.

_TRIGGER_NOT_CONVERGED = "glm did not converge"
_TRIGGER_OPTIMALITY = "failed the optimality check"
_TRIGGER_INFORMATION = "information could not be evaluated"
_TRIGGER_PROFILE = "profile inner fit was replaced"
_TRIGGERS = (
    _TRIGGER_NOT_CONVERGED,
    _TRIGGER_OPTIMALITY,
    _TRIGGER_INFORMATION,
    _TRIGGER_PROFILE,
)


def _triggers_in(warnings: tuple[str, ...]) -> set[str]:
    return {t for t in _TRIGGERS if any(t in w.lower() for w in warnings)}


def test_estimate_that_failed_the_optimality_check_names_only_that_trigger() -> None:
    """Pins: the clipped-IRLS point on the first data set converged but was not a maximum, and
    the warning says the optimality check failed -- not that the GLM did not converge."""
    fit = fit_dose_response(_c1(), model="binomial", link="cloglog", upper=1.0, lower=0.0)

    assert _triggers_in(fit.warnings) == {_TRIGGER_OPTIMALITY}


def test_probit_estimate_that_failed_the_optimality_check_names_only_that_trigger() -> None:
    """Pins: the probit case's GLM converges but fails the optimality check (Newton-decrement
    half about 0.004), and the warning names that trigger only."""
    fit = fit_dose_response(_probit_case(), model="binomial", link="probit", upper=1.0, lower=0.0)

    assert fit.status is Status.OK
    assert _triggers_in(fit.warnings) == {_TRIGGER_OPTIMALITY}


def test_glm_that_did_not_converge_names_only_that_trigger() -> None:
    """Pins: a cloglog design with a rising tail, on which the GLM itself reports
    non-convergence, returns ``OK`` at the independent maximum and names that trigger only."""
    obs = _obs(_LINEAR, [1.0, 2.0, 4.0, 8.0, 16.0], [914, 845, 693, 37, 3], [1000] * 5)
    mu, s, _ = _independent_mle(obs, "cloglog")

    fit = fit_dose_response(obs, model="binomial", link="cloglog", upper=1.0, lower=0.0)

    assert fit.status is Status.OK
    assert fit.params is not None
    assert fit.params["mu"].value == pytest.approx(mu, rel=1e-6)
    assert fit.params["s"].value == pytest.approx(s, rel=1e-6)
    assert _triggers_in(fit.warnings) == {_TRIGGER_NOT_CONVERGED}


# --- Newton-decrement criterion: a fit already at the maximum is not "corrected" ----------------

_SATURATED = [997, 838, 1, 0, 0]


def _linear_five(successes: list[int]) -> Observations:
    return _obs(_LINEAR, [1.0, 2.0, 4.0, 8.0, 16.0], successes, [1000] * 5)


@pytest.mark.parametrize("link", ["probit", "cloglog"])
def test_fit_already_at_the_maximum_is_ok_with_no_refit_warning(link: Link) -> None:
    """Pins: data with saturated cells whose GLM estimate already equals the maximum return
    ``OK`` at the independent maximum and carry no fallback warning of any trigger."""
    obs = _linear_five(_SATURATED)
    mu, s, llf = _independent_mle(obs, link)

    fit = fit_dose_response(obs, model="binomial", link=link, upper=1.0, lower=0.0)

    assert fit.status is Status.OK
    assert fit.params is not None and fit.log_likelihood is not None
    # A GLM estimate accepted without refit is within the acceptance criterion of the maximum
    # (half Newton decrement <= 1e-6, i.e. about 0.0014 SE), so agreement is stated in SE units.
    se = np.sqrt(np.diag(_numerical_covariance(obs, link, mu, s)))
    assert abs(fit.params["mu"].value - mu) <= 2e-3 * se[0]
    assert abs(fit.params["s"].value - s) <= 2e-3 * se[1]
    assert fit.log_likelihood == pytest.approx(llf, abs=1e-4)
    assert _triggers_in(fit.warnings) == set()
    assert not any("direct" in w.lower() and "likelihood" in w.lower() for w in fit.warnings)


# --- Non-finite observed information ------------------------------------------------------------


def _numerical_covariance(
    obs: Observations, link: Link, mu: float, s: float
) -> NDArray[np.float64]:
    """Inverse of the negative numerical Hessian of the log-likelihood in ``(mu, s)``."""
    cells = obs.cells()
    t = np.array([c.severity for c in cells], dtype=np.float64)
    k = np.array([c.successes for c in cells], dtype=np.float64)
    n = np.array([c.trials for c in cells], dtype=np.float64)

    def f(a: float, b: float) -> float:
        return _log_likelihood(a, b, t, k, n, link)

    h, g = 1e-4 * (1.0 + abs(mu)), 1e-4 * (1.0 + abs(s))
    f0 = f(mu, s)
    d_mm = (f(mu + h, s) - 2.0 * f0 + f(mu - h, s)) / h**2
    d_ss = (f(mu, s + g) - 2.0 * f0 + f(mu, s - g)) / g**2
    d_ms = (f(mu + h, s + g) - f(mu + h, s - g) - f(mu - h, s + g) + f(mu - h, s - g)) / (
        4.0 * h * g
    )
    return np.linalg.inv(-np.array([[d_mm, d_ms], [d_ms, d_ss]]))


@pytest.mark.parametrize("successes", [[995, 928, 9, 0, 0], [759, 194, 0, 0, 0], _SATURATED])
def test_logit_fit_with_saturated_cells_is_ok_with_finite_covariance(
    successes: list[int],
) -> None:
    """Pins: logit data with all-failure cells (where statsmodels' observed information divides
    by zero) returns ``OK`` at the independent maximum, a finite positive-definite covariance
    equal to the inverse numerical information of the stable log-likelihood (1e-3 relative on
    the entries), and a warning naming the non-evaluable-information trigger."""
    obs = _linear_five(successes)
    mu, s, llf = _independent_mle(obs, "logit")

    fit = fit_dose_response(obs, model="binomial", link="logit", upper=1.0, lower=0.0)

    assert fit.status is Status.OK
    assert fit.params is not None and fit.covariance is not None
    assert fit.params["mu"].value == pytest.approx(mu, rel=1e-6)
    assert fit.params["s"].value == pytest.approx(s, rel=1e-6)
    assert fit.log_likelihood == pytest.approx(llf, abs=1e-6)
    cov = np.array(fit.covariance.matrix, dtype=np.float64)
    assert np.all(np.isfinite(cov))
    assert np.all(np.linalg.eigvalsh((cov + cov.T) / 2.0) > 0.0)
    order = [fit.covariance.names.index("mu"), fit.covariance.names.index("s")]
    expected = _numerical_covariance(obs, "logit", mu, s)
    np.testing.assert_allclose(cov[np.ix_(order, order)], expected, rtol=1e-3)
    assert _TRIGGER_INFORMATION in _triggers_in(fit.warnings)


# --- cloglog far tail ---------------------------------------------------------------------------


@pytest.mark.parametrize("z", [-800.0, -750.0, -740.0, -100.0, -31.0])
def test_cloglog_log_cdf_is_finite_and_equals_z_in_the_far_tail(z: float) -> None:
    """Pins: ``log F(z) = log(1 - exp(-exp(z)))`` is ``z`` to double precision for very negative
    ``z`` (it must not underflow to ``-inf`` near ``z = -745``)."""
    value = _models._full_binomial_log_likelihood(
        np.array([z]),
        np.array([0.0]),
        np.array([1.0]),
        mu=0.0,
        s=1.0,
        upper=1.0,
        lower=0.0,
        scale="linear",
        link="cloglog",
    )

    assert math.isfinite(value)
    assert value == pytest.approx(z, rel=1e-12)


# --- Profile intervals against an independent constrained maximisation --------------------------

_LEVEL = 0.95
_TARGET = 0.5  # Definition.absolute(0.5): the severity at which the success probability is 0.5


def _link_quantile_for_target(link: Link) -> float:
    """``z*`` with ``S(z*) = 0.5``: ``p = S((t - mu) / s)``, so ``mu = t* - s z*``."""
    failure = 1.0 - _TARGET
    if link == "probit":
        return float(norm.ppf(failure))
    if link == "logit":
        return math.log(failure / (1.0 - failure))
    return math.log(-math.log1p(-failure))


def _independent_profile_interval(obs: Observations, link: Link) -> tuple[float, float]:
    """Profile-likelihood interval for the severity where ``p = 0.5``, in severity units.

    ``l_p(theta)`` maximises the log-space likelihood over ``s`` alone with
    ``mu = theta - s z*`` (``theta`` the transformed severity of the threshold); the endpoints
    are the roots of ``2 (l_hat - l_p(theta)) = chi2_{0.95, 1}`` either side of the maximum.
    """
    mu_hat, s_hat, llf_hat = _independent_mle(obs, link)
    cells = obs.cells()
    x = np.array([c.severity for c in cells], dtype=np.float64)
    k = np.array([c.successes for c in cells], dtype=np.float64)
    n = np.array([c.trials for c in cells], dtype=np.float64)
    is_log = obs.axis.scale == "log"
    keep = x > 0.0 if is_log else np.ones_like(x, dtype=bool)
    t = np.log(x[keep]) if is_log else x[keep]
    z_star = _link_quantile_for_target(link)
    constant = _lchoose(k, n)

    def profile(theta: float) -> float:
        def negative(log_s: float) -> float:
            s = math.exp(log_s)
            return -_log_likelihood(theta - s * z_star, s, t, k[keep], n[keep], link)

        result = minimize_scalar(
            negative, bracket=(math.log(s_hat) - 1.0, math.log(s_hat)), tol=1e-12
        )
        return -float(result.fun) + constant

    cutoff = float(chi2.ppf(_LEVEL, 1))
    theta_hat = mu_hat + s_hat * z_star

    def deviance_gap(theta: float) -> float:
        return 2.0 * (llf_hat - profile(theta)) - cutoff

    lo = brentq(deviance_gap, theta_hat - 5.0 * s_hat, theta_hat, xtol=1e-12)
    hi = brentq(deviance_gap, theta_hat, theta_hat + 5.0 * s_hat, xtol=1e-12)
    return (math.exp(lo), math.exp(hi)) if is_log else (lo, hi)


_PROFILE_CASES = ["minimal-cloglog", "rich-cloglog", "c1-cloglog", "probit"]

# Endpoints an independent reviewer derived for ``Definition.absolute(0.5)`` at level 0.95. They
# cross-check the in-test construction above, which is what the fits are compared with.
_REVIEWER_PROFILE = {
    "rich-cloglog": (1.0474, 1.1386),
    "c1-cloglog": (4.5541, 4.8612),
    "minimal-cloglog": (3.1191, 3.3555),
    "probit": (2.89532, 3.06161),
}


@pytest.mark.parametrize("name", _PROFILE_CASES)
def test_in_test_profile_agrees_with_reviewer_endpoints(name: str) -> None:
    """Pins the reference: the in-test constrained maximisation reproduces the reviewer's
    endpoints to their printed precision."""
    build, link = _CASES[name]

    lo, hi = _independent_profile_interval(build(), link)

    r_lo, r_hi = _REVIEWER_PROFILE[name]
    assert lo == pytest.approx(r_lo, rel=5e-5)
    assert hi == pytest.approx(r_hi, rel=5e-5)


@pytest.mark.parametrize("name", _PROFILE_CASES)
def test_profile_interval_matches_independent_constrained_maximisation(name: str) -> None:
    """Pins: ``threshold(..., absolute(0.5), interval_method="profile", level=0.95)`` has a
    finite interval whose endpoints equal the independent profile roots to 1e-3 relative, and
    whose point estimate is the severity where the maximum-likelihood curve crosses 0.5."""
    build, link = _CASES[name]
    obs = build()
    mu, s, _ = _independent_mle(obs, link)
    lo, hi = _independent_profile_interval(obs, link)
    fit = fit_dose_response(obs, model="binomial", link=link, upper=1.0, lower=0.0)

    result = threshold(
        fit, definition=Definition.absolute(_TARGET), interval_method="profile", level=_LEVEL
    )

    expected_value = mu + s * _link_quantile_for_target(link)
    if obs.axis.scale == "log":
        expected_value = math.exp(expected_value)
    assert result.status is Status.OK
    assert result.censoring is Censoring.NONE
    assert result.value == pytest.approx(expected_value, rel=1e-5)
    assert result.lo == pytest.approx(lo, rel=1e-3)
    assert result.hi == pytest.approx(hi, rel=1e-3)


@pytest.mark.parametrize("name", ["minimal-cloglog", "rich-cloglog", "c1-cloglog"])
def test_profile_names_the_inner_fit_replacement_when_it_happens(name: str) -> None:
    """Pins: where the offset GLM's inner fit was replaced by direct constrained maximisation
    the interval's result carries the profile trigger warning (these cloglog designs are ones
    where the clipped IRLS is not at the maximum)."""
    build, link = _CASES[name]
    fit = fit_dose_response(build(), model="binomial", link=link, upper=1.0, lower=0.0)

    result = threshold(
        fit, definition=Definition.absolute(_TARGET), interval_method="profile", level=_LEVEL
    )

    assert _TRIGGER_PROFILE in _triggers_in(result.warnings)


# --- Unit independence of the profile interval --------------------------------------------------


def _smooth_counts(link: Link) -> list[int]:
    """Counts at the expected values of a smooth curve (``mu = 3``, ``s = 1.2``), 1000 trials,
    on severities 1..6: every cell is interior, so the data are not near any boundary."""
    x = np.arange(1.0, 7.0)
    p = np.exp(_log_survival_and_cdf((x - 3.0) / 1.2, link)[0])
    return [int(round(1000.0 * v)) for v in p]


@pytest.mark.parametrize("link", ["probit", "logit"])
def test_profile_interval_and_warnings_do_not_depend_on_severity_units(link: Link) -> None:
    """Pins: rescaling a linear severity axis by ``c`` in {1e-3, 1, 1e3, 1e6} scales the profile
    threshold's ``value``, ``lo`` and ``hi`` by ``c`` (1e-4 relative) and leaves
    ``Threshold.warnings`` identical to the ``c = 1`` result."""
    severity = np.arange(1.0, 7.0)
    counts = _smooth_counts(link)
    results = {}
    for c in (1e-3, 1.0, 1e3, 1e6):
        obs = _obs(_LINEAR, (severity * c).tolist(), counts, [1000] * 6)
        fit = fit_dose_response(obs, model="binomial", link=link, upper=1.0, lower=0.0)
        results[c] = threshold(
            fit, definition=Definition.absolute(_TARGET), interval_method="profile", level=_LEVEL
        )

    base = results[1.0]
    assert base.value is not None and base.lo is not None and base.hi is not None
    for c, result in results.items():
        assert result.status is Status.OK
        assert result.value is not None and result.lo is not None and result.hi is not None
        assert result.value == pytest.approx(c * base.value, rel=1e-4)
        assert result.lo == pytest.approx(c * base.lo, rel=1e-4)
        assert result.hi == pytest.approx(c * base.hi, rel=1e-4)
        assert result.warnings == base.warnings
