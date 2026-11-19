"""Properties of clustered inference (hypothesis, derandomised).

* **One cluster per row.** When every row is its own cluster, the sandwich is the row-level
  (HC0-type) sandwich ``B (sum_i s_i s_i') B`` times ``G / (G - 1)`` with ``G`` the number of
  rows, checked against the numerical per-trial oracle in ``tests/unit/_cluster_support.py``.
* **Relabelling and row order.** Neither the names of the clusters nor the order of the rows
  changes the covariance or the sign test.
* **Mirror.** An increasing fit on events ``k`` and a decreasing fit on ``n - k`` give the same
  ``cluster_bootstrap`` bounds under one seed.

Datasets are probit-normal clustered draws from a seed, so the examples are reproducible; an
example whose fit is not ``OK`` is discarded.
"""

from __future__ import annotations

import pytest
from _cluster_support import (
    Row,
    counts_obs,
    covariance_array,
    oracle_sandwich,
    relative_matrix_error,
    simulate_rows,
)
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from robustmargin import Definition, Fit, Status, check_monotonicity, fit_dose_response, threshold

_SETTINGS = settings(
    max_examples=12,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

seeds = st.integers(min_value=0, max_value=10**6)
cluster_counts = st.integers(min_value=6, max_value=10)
effects = st.floats(min_value=0.2, max_value=1.2)


def _clustered(rows: list[Row], *, direction: str = "decreasing") -> Fit:
    return fit_dose_response(
        counts_obs(rows, direction=direction),
        model="binomial",
        link="probit",
        upper=1.0,
        lower=0.0,
        dependence="clustered",
    )


@_SETTINGS
@given(seed=seeds, sigma=effects)
def test_one_cluster_per_row_gives_the_row_sandwich_scaled_by_g_over_g_minus_one(
    seed: int, sigma: float
) -> None:
    base = simulate_rows(seed, clusters=2, trials=60, sigma=sigma)
    rows: list[Row] = []
    for index, (sev, _name, k, n) in enumerate(base):
        half = k // 2
        rows += [(sev, f"r{index}a", half, n // 2), (sev, f"r{index}b", k - half, n - n // 2)]
    fit = _clustered(rows)
    assume(fit.status is Status.OK)

    oracle = oracle_sandwich(fit, rows)

    assert oracle.clusters == len(rows)
    assert relative_matrix_error(covariance_array(fit), oracle.covariance) < 1e-4


@_SETTINGS
@given(seed=seeds, clusters=cluster_counts, sigma=effects, shift=st.integers(1, 10**4))
def test_relabelling_the_clusters_does_not_change_the_clustered_covariance(
    seed: int, clusters: int, sigma: float, shift: int
) -> None:
    rows = simulate_rows(seed, clusters=clusters, sigma=sigma)
    relabelled = [
        (sev, f"x{(int(name[1:]) * 7919 + shift) % 10**6:06d}", k, n) for sev, name, k, n in rows
    ]
    a = _clustered(rows)
    assume(a.status is Status.OK)

    b = _clustered(relabelled)

    assert relative_matrix_error(covariance_array(a), covariance_array(b)) < 1e-9


@_SETTINGS
@given(seed=seeds, clusters=cluster_counts, sigma=effects)
def test_reversing_the_row_order_does_not_change_the_covariance_or_the_sign_test(
    seed: int, clusters: int, sigma: float
) -> None:
    rows = simulate_rows(seed, clusters=clusters, sigma=sigma)
    a = _clustered(rows)
    assume(a.status is Status.OK)

    b = _clustered(rows[::-1])

    assert relative_matrix_error(covariance_array(a), covariance_array(b)) < 1e-9
    check_a = check_monotonicity(counts_obs(rows), dependence="clustered")
    check_b = check_monotonicity(counts_obs(rows[::-1]), dependence="clustered")
    assert [p.p_value for p in check_a.pairs] == [p.p_value for p in check_b.pairs]


@_SETTINGS
@given(seed=seeds, clusters=cluster_counts, sigma=effects)
def test_cluster_bootstrap_of_an_increasing_fit_mirrors_the_decreasing_fit_under_one_seed(
    seed: int, clusters: int, sigma: float
) -> None:
    rows = simulate_rows(seed, clusters=clusters, sigma=sigma)
    decreasing = _clustered(rows)
    assume(decreasing.status is Status.OK)
    flipped = [(sev, name, n - k, n) for sev, name, k, n in rows]
    increasing = _clustered(flipped, direction="increasing")

    def solve(fit: Fit) -> tuple[float | None, float | None, str, int | None]:
        result = threshold(
            fit,
            definition=Definition.absolute(0.5),
            interval_method="cluster_bootstrap",
            level=0.95,
            dependence="clustered",
            n_resamples=29,
            seed=20261700,
        )
        failed = None if result.bootstrap is None else result.bootstrap.failed
        return result.lo, result.hi, result.status.name, failed

    lo_d, hi_d, status_d, failed_d = solve(decreasing)
    lo_i, hi_i, status_i, failed_i = solve(increasing)

    assert (status_i, failed_i) == (status_d, failed_d)
    for mirrored, original in ((lo_i, lo_d), (hi_i, hi_d)):
        if original is None:
            assert mirrored is None
        else:
            assert mirrored == pytest.approx(original, rel=1e-9)
