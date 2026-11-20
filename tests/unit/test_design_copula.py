"""The clustered/paired data-generating process behind the width planner.

The process is a Gaussian copula with a latent threshold: ``Y = sqrt(icc) * b_g +
sqrt(1 - icc) * e`` with ``b_g, e ~ N(0, 1)``, and a trial succeeds when ``Phi(Y) < P(x)``.
Every level's marginal success rate is therefore the pilot curve's ``P(x)`` whatever ``icc`` is.

The planner returns only widths, so the process is checked through one private helper. This
file assumes it is::

    robustmargin.design._simulate_counts(
        curves: Sequence[Fit],          # one pilot fit per arm (1 or 2)
        severity: Sequence[float],
        trials_per_level: int,
        *,
        rng: numpy.random.Generator,
        n_clusters: int | None = None,
        icc: float = 0.0,
        shared_effects: bool = False,   # arms share the cluster effects b_g (paired)
    ) -> numpy.ndarray                  # int successes, shape (n_arms, n_levels, G)

where ``G = n_clusters`` (or 1 when ``n_clusters`` is ``None``) and every cell has
``trials_per_level // G`` trials. With ``shared_effects=False`` the arms draw independent
cluster effects.

Per-level counts of successes are for the outcome as counted (the pilot's own orientation), so
for a decreasing pilot the expected rate is ``P(x)``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy.stats import binomtest, norm

from robustmargin import Axis, Fit, Observations, fit_dose_response
from robustmargin.design import _simulate_counts

_LOG = Axis(name="severity", unit="unit", scale="log")
_MU = 0.0
_S = 0.5
_LEVELS = (0.25, 0.5, 1.0, 2.0, 4.0)


def _pilot() -> Fit:
    levels = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8])
    p = 1.0 - norm.cdf((np.log(levels) - _MU) / _S)
    obs = Observations.from_counts(
        _LOG,
        severity=list(levels),
        successes=[int(round(200 * v)) for v in p],
        trials=[200] * len(levels),
        outcome="outcome",
        direction="decreasing",
    )
    return fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)


def _curve_rate(fit: Fit, x: float) -> float:
    assert fit.params is not None
    mu, s = fit.params["mu"].value, fit.params["s"].value
    return float(1.0 - norm.cdf((math.log(x) - mu) / s))


def test_counts_have_the_documented_shape_and_bounds() -> None:
    out = _simulate_counts(
        [_pilot(), _pilot()],
        _LEVELS,
        60,
        rng=np.random.default_rng(1),
        n_clusters=6,
        icc=0.3,
        shared_effects=True,
    )
    assert out.shape == (2, len(_LEVELS), 6)
    assert out.min() >= 0
    assert out.max() <= 10


def test_unclustered_counts_have_one_cell_per_level() -> None:
    out = _simulate_counts([_pilot()], _LEVELS, 80, rng=np.random.default_rng(2))
    assert out.shape == (1, len(_LEVELS), 1)
    assert out.max() <= 80


def test_same_generator_state_gives_identical_counts() -> None:
    kwargs: dict[str, Any] = {"n_clusters": 5, "icc": 0.4}
    a = _simulate_counts([_pilot()], _LEVELS, 50, rng=np.random.default_rng(9), **kwargs)
    b = _simulate_counts([_pilot()], _LEVELS, 50, rng=np.random.default_rng(9), **kwargs)
    assert np.array_equal(a, b)


def test_independent_trials_match_the_curve_at_every_level() -> None:
    fit = _pilot()
    n = 400_000
    out = _simulate_counts([fit], _LEVELS, n, rng=np.random.default_rng(20261119))
    for j, x in enumerate(_LEVELS):
        k = int(out[0, j, :].sum())
        assert binomtest(k, n, _curve_rate(fit, x)).pvalue > 1e-4


@pytest.mark.parametrize("icc", [0.3, 0.8])
def test_clustered_marginal_rate_still_matches_the_curve(icc: float) -> None:
    # The estimand must not move with icc. Compare the mean over many independent draws to the
    # curve using the draws' own spread, so the cluster design effect needs no formula.
    fit = _pilot()
    rng = np.random.default_rng(20261120)
    draws = np.stack(
        [
            _simulate_counts([fit], _LEVELS, 2000, rng=rng, n_clusters=20, icc=icc)[0].sum(axis=1)
            / 2000.0
            for _ in range(300)
        ]
    )  # (draws, levels)
    mean = draws.mean(axis=0)
    se = draws.std(axis=0, ddof=1) / math.sqrt(draws.shape[0])
    for j, x in enumerate(_LEVELS):
        assert abs(mean[j] - _curve_rate(fit, x)) < 4.5 * se[j]


def _within_cluster_binary_correlation(icc: float, seed: int) -> float:
    """Moment estimate of the within-cluster correlation of the binary outcome at the level
    where the rate is one half (severity 1.0 is exp(mu))."""
    fit = _pilot()
    clusters, per = 2000, 20
    out = _simulate_counts(
        [fit],
        (0.5, 1.0, 2.0),
        clusters * per,
        rng=np.random.default_rng(seed),
        n_clusters=clusters,
        icc=icc,
    )
    k = out[0, 1, :].astype(float)
    p = k.sum() / (clusters * per)
    ratio = k.var(ddof=1) / (per * p * (1.0 - p))
    return float((ratio - 1.0) / (per - 1.0))


def test_zero_icc_gives_binomial_dispersion_across_clusters() -> None:
    assert abs(_within_cluster_binary_correlation(0.0, 31)) < 0.03


@pytest.mark.parametrize("icc", [0.3, 0.6, 0.9])
def test_latent_icc_maps_to_the_gaussian_copula_binary_correlation(icc: float) -> None:
    # At a rate of one half the binary correlation implied by a latent correlation r is
    # (2 / pi) * arcsin(r).
    expected = 2.0 / math.pi * math.asin(icc)
    assert _within_cluster_binary_correlation(icc, 32) == pytest.approx(expected, abs=0.04)


def test_within_cluster_correlation_rises_with_icc() -> None:
    estimates = [_within_cluster_binary_correlation(icc, 33) for icc in (0.0, 0.3, 0.6, 0.9)]
    assert estimates == sorted(estimates)
    assert estimates[-1] - estimates[0] > 0.4


def _arm_correlation(*, shared: bool, seed: int) -> float:
    fit = _pilot()
    clusters, per = 400, 50
    out = _simulate_counts(
        [fit, fit],
        (0.5, 1.0, 2.0),
        clusters * per,
        rng=np.random.default_rng(seed),
        n_clusters=clusters,
        icc=0.7,
        shared_effects=shared,
    )
    return float(np.corrcoef(out[0, 1, :], out[1, 1, :])[0, 1])


def test_paired_arms_share_cluster_effects() -> None:
    assert _arm_correlation(shared=True, seed=41) > 0.5


def test_independent_arms_do_not_share_cluster_effects() -> None:
    assert abs(_arm_correlation(shared=False, seed=42)) < 0.2
