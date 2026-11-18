"""Reduced coverage regression for clustered inference.

Marked ``slow`` and excluded from the default run. Run locally with::

    pytest -m slow --no-cov -s tests/coverage/test_cluster_coverage.py

The full calibration grid (which sets the few-clusters threshold ``K`` and the cross-check
constants ``c`` and ``d0``) is an offline, seeded script and is not part of the suite; this file
is the regression that keeps the qualitative facts true.

**Design.** Probit-normal clusters: ``P_c(x) = 1 - Phi((t - mu)/s + b_c)``, ``b_c ~ N(0, sigma^2)``
with ``t = log x``, so the marginal curve is exactly a probit with ``s_m = s * sqrt(1 + sigma^2)``
and the true threshold at performance ``p`` is ``exp(mu + s_m * Phi^-1(1 - p))``. Log axis, a
zero-severity control plus ``{1, 2, 4, 8, 16, 32}``, ``mu = log 6``, ``s = 0.6``, probit, fixed
``u = 1`` and ``l = 0``, 16 trials per cluster and level, intracluster correlation 0.3 on the
latent scale (``sigma^2 = icc / (1 - icc)``), ``p = 0.5``, 95% intervals.

**What is asserted.** Coverage is the share of datasets whose interval contains the truth, a side
left open counting as unbounded and a failed fit or a bootstrap that gives up as a miss. Each
assertion fails only when the exact (Clopper-Pearson) 99% interval of the observed coverage lies
wholly on the wrong side of the stated bound, i.e. only when the simulation shows with 99%
confidence that the claim is false:

* the sandwich ``"delta"`` interval with Student t on G - 1 degrees of freedom, the primary
  clustered interval, is not below ``0.92`` at 5, 10, 20 and 40 clusters;
* the independence profile interval under-covers (its upper 99% bound is below ``0.85``) at 5
  and 40 clusters, because it treats 16 correlated trials as independent;
* the percentile cluster bootstrap is not below ``0.88`` at 40 clusters.

The percentile cluster bootstrap at 5 clusters is printed, not asserted. It estimates the
cluster variance without the G/(G - 1) factor and reads normal-type quantiles, so its coverage
is about ``P(|t_(G-1)| <= 1.96 * sqrt((G - 1) / G))`` (0.85 at 5 clusters); poor coverage with
few clusters is a finding, not a bug, and ``FEW_CLUSTERS`` says so. Seeds are recorded below.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
import pytest
from _cluster_support import MU, SCALE, counts_obs, simulate_rows
from scipy.stats import binomtest, norm

from robustmargin import Definition, Status, fit_dose_response, threshold

_SEED = 20261801
_REPLICATES = 400
_BOOTSTRAP_REPLICATES = {5: 30, 40: 60}
_RESAMPLES = 199
_LEVEL = 0.95
_ICC = 0.3
_PERFORMANCE = 0.5
_TRIALS = 16


def _true_threshold() -> float:
    sigma2 = _ICC / (1.0 - _ICC)
    s_marginal = SCALE * math.sqrt(1.0 + sigma2)
    return math.exp(MU + s_marginal * float(norm.ppf(1.0 - _PERFORMANCE)))


@dataclass
class _Tally:
    covered: int = 0
    low_miss: int = 0
    high_miss: int = 0
    failed: int = 0
    open_sides: int = 0
    n: int = 0

    def add(self, lo: float | None, hi: float | None, truth: float, *, ok: bool) -> None:
        self.n += 1
        if not ok:
            self.failed += 1
            return
        if lo is not None and truth < lo:
            self.low_miss += 1
        elif hi is not None and truth > hi:
            self.high_miss += 1
        else:
            self.covered += 1
        if lo is None or hi is None:
            self.open_sides += 1

    def exact_interval(self) -> tuple[float, float]:
        ci = binomtest(self.covered, self.n).proportion_ci(confidence_level=0.99)
        return float(ci.low), float(ci.high)


@functools.cache
def _simulate(clusters: int) -> dict[str, _Tally]:
    sigma = math.sqrt(_ICC / (1.0 - _ICC))
    truth = _true_threshold()
    tallies = {"delta": _Tally(), "profile": _Tally(), "bootstrap": _Tally()}
    rng = np.random.default_rng([_SEED, clusters])
    definition = Definition.absolute(_PERFORMANCE)
    for rep in range(_REPLICATES):
        seed = int(rng.integers(0, 2**31 - 1))
        obs = counts_obs(simulate_rows(seed, clusters=clusters, trials=_TRIALS, sigma=sigma))

        fit_c = fit_dose_response(
            obs, model="binomial", link="probit", upper=1.0, lower=0.0, dependence="clustered"
        )
        t_d = threshold(
            fit_c,
            definition=definition,
            interval_method="delta",
            level=_LEVEL,
            dependence="clustered",
        )
        tallies["delta"].add(t_d.lo, t_d.hi, truth, ok=t_d.status is Status.OK)

        if clusters in (5, 40):
            fit_i = fit_dose_response(
                obs,
                model="binomial",
                link="probit",
                upper=1.0,
                lower=0.0,
                dependence="independent",
            )
            t_i = threshold(
                fit_i,
                definition=definition,
                interval_method="profile",
                level=_LEVEL,
                dependence="independent",
            )
            tallies["profile"].add(t_i.lo, t_i.hi, truth, ok=t_i.status is Status.OK)

            if rep < _BOOTSTRAP_REPLICATES[clusters]:
                t_b = threshold(
                    fit_c,
                    definition=definition,
                    interval_method="cluster_bootstrap",
                    level=_LEVEL,
                    dependence="clustered",
                    n_resamples=_RESAMPLES,
                    seed=seed,
                )
                tallies["bootstrap"].add(
                    t_b.lo, t_b.hi, truth, ok=t_b.status is Status.OK and t_b.value is not None
                )
    return tallies


def _report(label: str, clusters: int, tally: _Tally) -> tuple[float, float, float]:
    low, high = tally.exact_interval()
    p = tally.covered / tally.n
    se = math.sqrt(p * (1 - p) / tally.n)
    print(
        f"{label} G={clusters}: coverage {p:.3f} of {tally.n} (MC SE {se:.3f}, "
        f"exact 99% [{low:.3f}, {high:.3f}]); "
        f"low misses {tally.low_miss}, high misses {tally.high_miss}, failed {tally.failed}, "
        f"open {tally.open_sides}"
    )
    return p, low, high


@pytest.mark.slow
@pytest.mark.parametrize("clusters", [5, 10, 20, 40])
def test_clustered_delta_interval_with_t_critical_value_covers_near_nominal(clusters: int) -> None:
    tallies = _simulate(clusters)

    _, _, high = _report("clustered delta, t(G-1)", clusters, tallies["delta"])

    assert high >= 0.92


@pytest.mark.slow
@pytest.mark.parametrize("clusters", [5, 40])
def test_independence_profile_interval_under_covers_clustered_data(clusters: int) -> None:
    tallies = _simulate(clusters)

    _, _, high = _report("independence profile", clusters, tallies["profile"])

    assert high < 0.85


@pytest.mark.slow
def test_cluster_bootstrap_with_five_clusters_coverage_is_recorded() -> None:
    tallies = _simulate(5)

    _report("percentile cluster bootstrap", 5, tallies["bootstrap"])


@pytest.mark.slow
def test_cluster_bootstrap_with_forty_clusters_is_not_below_88_percent_coverage() -> None:
    tallies = _simulate(40)

    _, _, high = _report("percentile cluster bootstrap", 40, tallies["bootstrap"])

    assert high >= 0.88
