"""The clustered ``"delta"`` interval uses Student t with ``G - 1`` degrees of freedom.

A CR1 sandwich with a normal quantile under-covers badly at small ``G``, so the clustered delta
interval and the paired Fieller/log-delta intervals use ``t(G - 1)``. The interval is computed
here independently from the stored covariance: ``value * exp(+-t * se)`` on a log axis, with
``se`` the delta standard error of ``theta = mu + s * z*``. The independent path stays on ``z``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _cluster_support import counts_obs, covariance_array, simulate_rows
from scipy.stats import norm
from scipy.stats import t as student_t

from robustmargin import Definition, Fit, Status, fit_dose_response, threshold

_LEVEL = 0.95


def _clustered_fit(clusters: int, seed: int = 20261301) -> Fit:
    obs = counts_obs(simulate_rows(seed, clusters=clusters))
    return fit_dose_response(
        obs, model="binomial", link="probit", upper=1.0, lower=0.0, dependence="clustered"
    )


def _expected(fit: Fit, performance: float, quantile: float) -> tuple[float, float, float]:
    assert fit.params is not None
    z_star = float(norm.ppf(1.0 - performance))
    mu = fit.params["mu"].value
    s = fit.params["s"].value
    cov = covariance_array(fit)
    gradient = np.array([1.0, z_star])
    se = math.sqrt(float(gradient @ cov @ gradient))
    value = math.exp(mu + s * z_star)
    return value, value * math.exp(-quantile * se), value * math.exp(quantile * se)


@pytest.mark.parametrize("clusters", [6, 12, 30])
@pytest.mark.parametrize("performance", [0.5, 0.8])
def test_clustered_delta_interval_uses_student_t_with_g_minus_one_degrees_of_freedom(
    clusters: int, performance: float
) -> None:
    fit = _clustered_fit(clusters)
    assert fit.status is Status.OK

    result = threshold(
        fit,
        definition=Definition.absolute(performance),
        interval_method="delta",
        level=_LEVEL,
        dependence="clustered",
    )

    quantile = float(student_t.ppf(0.5 + _LEVEL / 2.0, clusters - 1))
    value, lo, hi = _expected(fit, performance, quantile)
    assert result.value == pytest.approx(value, rel=1e-9)
    assert result.lo == pytest.approx(lo, rel=1e-6)
    assert result.hi == pytest.approx(hi, rel=1e-6)
    assert result.interval_method == "delta"
    assert result.dependence == "clustered"


def test_clustered_delta_interval_is_wider_than_the_normal_quantile_interval_at_few_clusters() -> (
    None
):
    fit = _clustered_fit(6)

    result = threshold(
        fit,
        definition=Definition.absolute(0.5),
        interval_method="delta",
        level=_LEVEL,
        dependence="clustered",
    )

    _, normal_lo, normal_hi = _expected(fit, 0.5, float(norm.ppf(0.5 + _LEVEL / 2.0)))
    assert result.lo is not None and result.hi is not None
    assert result.lo < normal_lo
    assert result.hi > normal_hi


def test_clustered_delta_interval_approaches_the_normal_interval_with_many_clusters() -> None:
    fit = _clustered_fit(400, seed=20261302)

    result = threshold(
        fit,
        definition=Definition.absolute(0.5),
        interval_method="delta",
        level=_LEVEL,
        dependence="clustered",
    )

    _, lo, hi = _expected(fit, 0.5, float(student_t.ppf(0.5 + _LEVEL / 2.0, 399)))
    assert result.lo == pytest.approx(lo, rel=1e-6)
    assert result.hi == pytest.approx(hi, rel=1e-6)


def test_independent_delta_interval_keeps_the_normal_quantile() -> None:
    obs = counts_obs(simulate_rows(20261301, clusters=6))
    fit = fit_dose_response(
        obs, model="binomial", link="probit", upper=1.0, lower=0.0, dependence="independent"
    )

    result = threshold(
        fit,
        definition=Definition.absolute(0.5),
        interval_method="delta",
        level=_LEVEL,
        dependence="independent",
    )

    _, lo, hi = _expected(fit, 0.5, float(norm.ppf(0.5 + _LEVEL / 2.0)))
    assert result.lo == pytest.approx(lo, rel=1e-6)
    assert result.hi == pytest.approx(hi, rel=1e-6)


def _expected_band(fit: Fit, severity: float, quantile: float) -> tuple[float, float, float]:
    """The curve ``P = 1 - F(z)`` and its band, from the delta method on ``z = (log x - mu) / s``
    with the stored covariance, mapped through the monotone link."""
    assert fit.params is not None
    mu = fit.params["mu"].value
    s = fit.params["s"].value
    z = (math.log(severity) - mu) / s
    gradient = np.array([-1.0 / s, -z / s])
    se = math.sqrt(float(gradient @ covariance_array(fit) @ gradient))
    return (
        float(norm.sf(z)),
        float(norm.sf(z + quantile * se)),
        float(norm.sf(z - quantile * se)),
    )


@pytest.mark.parametrize("clusters", [6, 12, 30])
def test_clustered_predict_band_uses_student_t_with_g_minus_one_degrees_of_freedom(
    clusters: int,
) -> None:
    # Fails until Fit.predict applies t(G - 1) on a clustered fit; it uses a normal quantile now.
    fit = _clustered_fit(clusters)
    assert fit.status is Status.OK
    severities = [1.5, 6.0, 20.0]

    band = fit.predict(severities, level=_LEVEL)

    quantile = float(student_t.ppf(0.5 + _LEVEL / 2.0, clusters - 1))
    for i, x in enumerate(severities):
        estimate, lo, hi = _expected_band(fit, x, quantile)
        assert band.estimate[i] == pytest.approx(estimate, rel=1e-9)
        assert band.lo[i] == pytest.approx(lo, rel=1e-6)
        assert band.hi[i] == pytest.approx(hi, rel=1e-6)


def test_independent_predict_band_equals_the_delta_band_on_z_with_a_normal_quantile() -> None:
    obs = counts_obs(simulate_rows(20261301, clusters=6))
    fit = fit_dose_response(
        obs, model="binomial", link="probit", upper=1.0, lower=0.0, dependence="independent"
    )

    band = fit.predict([6.0], level=_LEVEL)

    _, lo, hi = _expected_band(fit, 6.0, float(norm.ppf(0.5 + _LEVEL / 2.0)))
    assert band.lo[0] == pytest.approx(lo, rel=1e-6)
    assert band.hi[0] == pytest.approx(hi, rel=1e-6)
