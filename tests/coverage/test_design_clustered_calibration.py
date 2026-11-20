"""The clustered draw of the width planner against an explicit per-trial latent simulation.

The planner draws, for each cluster, one shared normal effect and then binomial counts with the
conditional success probability (a shortcut). Here the same process is simulated trial by trial:
draw ``b_g`` per cluster and ``e`` per trial, ``Y = sqrt(icc) * b_g + sqrt(1 - icc) * e``, and
count a success when ``Phi(Y) < P(x)``. The joint distribution of the totals at two levels, and
of the totals of two paired arms, must agree.

The test is a chi-square test of homogeneity on a binned two-way table of the totals, with fixed
seeds. With 20000 draws per method and 5 x 5 bins the statistic has 16 degrees of freedom; the
cut ``p > 1e-3`` makes a false alarm on a correct implementation a one-in-a-thousand event for a
given seed, and the seeds are pinned, so the outcome is repeatable. A wrong shared effect (for
example independent effects per arm, or per level) moves the cross-table far outside that.

Marked ``slow`` and excluded from the default run. Run with ``pytest -m slow --no-cov``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import chi2_contingency, norm

from robustmargin import Axis, Fit, Observations, fit_dose_response
from robustmargin.design import _simulate_counts

pytestmark = pytest.mark.slow

_AXIS = Axis(name="severity", unit="unit", scale="log")
_LEVELS = (0.5, 2.0)
_DRAWS = 20_000
_CLUSTERS = 4
_TRIALS = 40  # per level, so 10 per cluster
_ICC = 0.5


def _pilot(mu: float) -> Fit:
    levels = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8])
    p = 1.0 - norm.cdf((np.log(levels) - mu) / 0.6)
    obs = Observations.from_counts(
        _AXIS,
        severity=list(levels),
        successes=[int(round(200 * v)) for v in p],
        trials=[200] * len(levels),
        outcome="outcome",
        direction="decreasing",
    )
    return fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)


def _planner_totals(curves: list[Fit], *, shared: bool, seed: int) -> NDArray[np.int64]:
    rng = np.random.default_rng(seed)
    out = np.empty((_DRAWS, len(curves), len(_LEVELS)), dtype=np.int64)
    for i in range(_DRAWS):
        counts = _simulate_counts(
            curves,
            _LEVELS,
            _TRIALS,
            rng=rng,
            n_clusters=_CLUSTERS,
            icc=_ICC,
            shared_effects=shared,
        )
        out[i] = counts.sum(axis=2)
    return out


def _latent_totals(curves: list[Fit], *, shared: bool, seed: int) -> NDArray[np.int64]:
    rng = np.random.default_rng(seed)
    arms, levels, per_cell = len(curves), len(_LEVELS), _TRIALS // _CLUSTERS
    p = np.stack([np.asarray(c.predict(np.array(_LEVELS)).estimate) for c in curves])
    b = rng.standard_normal((_DRAWS, 1 if shared else arms, 1, _CLUSTERS, 1))
    e = rng.standard_normal((_DRAWS, arms, levels, _CLUSTERS, per_cell))
    y = np.sqrt(_ICC) * b + np.sqrt(1.0 - _ICC) * e
    success = norm.cdf(y) < p[None, :, :, None, None]
    totals: NDArray[np.int64] = success.sum(axis=(3, 4)).astype(np.int64)
    return totals


def _binned_table(
    first: NDArray[np.int64],
    second: NDArray[np.int64],
    edges: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.int64]:
    table = np.zeros((5, 5), dtype=np.int64)
    a = np.digitize(first, edges[0])
    b = np.digitize(second, edges[1])
    np.add.at(table, (a, b), 1)
    return table


def _assert_same_joint(
    planner: NDArray[np.int64],
    latent: NDArray[np.int64],
    pair: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    (arm_a, level_a), (arm_b, level_b) = pair
    pooled_a = np.concatenate([planner[:, arm_a, level_a], latent[:, arm_a, level_a]])
    pooled_b = np.concatenate([planner[:, arm_b, level_b], latent[:, arm_b, level_b]])
    edges = (
        np.unique(np.quantile(pooled_a, [0.2, 0.4, 0.6, 0.8])),
        np.unique(np.quantile(pooled_b, [0.2, 0.4, 0.6, 0.8])),
    )
    tables = [
        _binned_table(x[:, arm_a, level_a], x[:, arm_b, level_b], edges) for x in (planner, latent)
    ]
    # Collapse cells that are empty in both tables so the test is well defined.
    both = tables[0] + tables[1]
    keep = both.ravel() > 0
    stacked = np.stack([tables[0].ravel()[keep], tables[1].ravel()[keep]])
    _, p_value, _, _ = chi2_contingency(stacked)
    print(f"pair {pair}: p = {p_value:.4f}")
    assert p_value > 1e-3


def test_unpaired_clustered_draw_matches_a_per_trial_latent_simulation_across_levels() -> None:
    curves = [_pilot(0.0)]
    planner = _planner_totals(curves, shared=False, seed=101)
    latent = _latent_totals(curves, shared=False, seed=202)
    _assert_same_joint(planner, latent, ((0, 0), (0, 1)))


def test_paired_clustered_draw_matches_a_per_trial_latent_simulation_across_arms() -> None:
    curves = [_pilot(0.0), _pilot(0.4)]
    planner = _planner_totals(curves, shared=True, seed=303)
    latent = _latent_totals(curves, shared=True, seed=404)
    _assert_same_joint(planner, latent, ((0, 1), (1, 1)))


def test_unshared_arms_are_independent_in_the_planner_and_in_the_latent_simulation() -> None:
    curves = [_pilot(0.0), _pilot(0.4)]
    planner = _planner_totals(curves, shared=False, seed=505)
    latent = _latent_totals(curves, shared=False, seed=606)
    _assert_same_joint(planner, latent, ((0, 1), (1, 1)))
