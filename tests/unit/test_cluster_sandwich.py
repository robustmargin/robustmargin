"""The cluster-robust sandwich covariance of ``fit_dose_response(dependence="clustered")``.

The covariance is ``V = B M B`` on the natural scale, ``B`` the observed-information inverse and
``M = G/(G-1) * sum_c s_c s_c'`` the cluster score outer product at the working-independence MLE
(no ``(N-1)/(N-k)`` term, because ``N`` depends on how rows are aggregated).

Oracles, none of which uses the implementation under test:

* **logit, two parameters.** statsmodels ``GLM(...).fit(cov_type="cluster")`` on one row per trial.
  statsmodels' own small-sample factor is switched off and replaced by ``G/(G-1)``, which is the
  same as rescaling its default correction. The logit is the canonical link, so observed equals
  expected information and the two bread definitions coincide.
* **probit, cloglog and estimated asymptotes.** A per-trial Bernoulli log-likelihood written in
  ``_cluster_support`` and differentiated numerically (scores per cluster, Hessian for the bread),
  on the natural scale. At the maximum the natural-scale sandwich is chart-invariant, so this is
  the same quantity the fit reports through its Jacobian.

R ``sandwich::vcovCL(type="HC0", cadjust=TRUE)`` fixtures are not present: R is not installed in
the authoring environment, so no reference JSON is invented.

Everything here fails until ``dependence="clustered"`` exists.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from _cluster_support import (
    LINEAR,
    RICH_LEVELS,
    Row,
    counts_obs,
    covariance_array,
    oracle_sandwich,
    relative_matrix_error,
    simulate_rows,
    trial_obs,
)
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.tools import add_constant

from robustmargin import Cell, Fit, Observations, Status, fit_dose_response

_SEED = 20261201
_Link = Literal["probit", "logit", "cloglog"]
_LINKS: tuple[_Link, ...] = ("probit", "logit", "cloglog")


def _fit(
    obs: Observations,
    *,
    link: _Link = "probit",
    upper: float | Literal["estimate"] = 1.0,
    lower: float | Literal["estimate"] = 0.0,
    dependence: str = "clustered",
) -> Fit:
    return fit_dose_response(
        obs,
        model="binomial",
        link=link,
        upper=upper,
        lower=lower,
        dependence=dependence,
    )


def _rows(clusters: int = 12, seed: int = _SEED, **kwargs: object) -> list[Row]:
    return simulate_rows(seed, clusters=clusters, **kwargs)  # type: ignore[arg-type]


# -- the fit is the independent fit, with another covariance ---------------------------------


@pytest.mark.parametrize("link", _LINKS)
def test_clustered_fit_keeps_the_independent_point_estimate_and_log_likelihood_exactly(
    link: _Link,
) -> None:
    obs = counts_obs(_rows())

    clustered = _fit(obs, link=link)
    independent = _fit(obs, link=link, dependence="independent")

    assert clustered.status is Status.OK
    assert clustered.dependence == "clustered"
    assert independent.dependence == "independent"
    assert clustered.params == independent.params
    assert clustered.log_likelihood == independent.log_likelihood


def test_clustered_fit_keeps_the_independent_estimate_with_estimated_asymptotes() -> None:
    rows = simulate_rows(_SEED, clusters=15, trials=24, levels=RICH_LEVELS, upper=0.93, lower=0.06)
    obs = counts_obs(rows)

    clustered = _fit(obs, upper="estimate", lower="estimate")
    independent = _fit(obs, upper="estimate", lower="estimate", dependence="independent")

    assert clustered.status is Status.OK
    assert clustered.params == independent.params
    assert clustered.log_likelihood == independent.log_likelihood


def test_clustered_covariance_differs_from_the_model_based_covariance() -> None:
    obs = counts_obs(_rows(sigma=1.0))

    clustered = covariance_array(_fit(obs))
    independent = covariance_array(_fit(obs, dependence="independent"))

    assert relative_matrix_error(clustered, independent) > 0.05


def test_clustered_fit_records_cluster_cells_summing_to_the_pooled_cells() -> None:
    rows = _rows()
    fit = _fit(counts_obs(rows))

    assert fit.cluster_cells is not None
    assert {c.cluster for c in fit.cluster_cells} == set(fit.cluster_ids or ())
    for cell in fit.cells:
        assert isinstance(cell, Cell)
        mine = [c for c in fit.cluster_cells if c.severity == cell.severity]
        assert sum(c.successes for c in mine) == cell.successes
        assert sum(c.trials for c in mine) == cell.trials


# -- logit: statsmodels GLM cluster covariance ------------------------------------------------


def _glm_cluster_natural(rows: list[Row]) -> np.ndarray:
    """statsmodels GLM Binomial/logit cluster covariance on one row per trial, with the
    small-sample correction replaced by G/(G-1), mapped from (a, b) to (mu, s) with
    ``logit P = a + b t``: ``mu = -a/b``, ``s = -1/b``. The log-axis control (all successes under
    ``u = 1``) contributes nothing to likelihood or score and is left out."""
    t: list[float] = []
    y: list[int] = []
    group: list[int] = []
    ids = list(dict.fromkeys(r[1] for r in rows))
    for sev, name, k, n in rows:
        if sev == 0.0:
            continue
        t += [float(np.log(sev))] * n
        y += [1] * k + [0] * (n - k)
        group += [ids.index(name)] * n
    exog = add_constant(np.array(t))
    result = GLM(np.array(y), exog, family=families.Binomial()).fit(
        cov_type="cluster",
        cov_kwds={"groups": np.array(group), "use_correction": False},
    )
    g = len(ids)
    v_ab = np.asarray(result.cov_params()) * g / (g - 1.0)
    a, b = (float(v) for v in result.params)
    jac = np.array([[-1.0 / b, a / b**2], [0.0, 1.0 / b**2]])
    return np.asarray(jac @ v_ab @ jac.T, dtype=np.float64)


def test_clustered_logit_covariance_matches_statsmodels_glm_cluster_covariance() -> None:
    rows = _rows(clusters=14, seed=_SEED + 1)

    fit = _fit(counts_obs(rows), link="logit")

    assert fit.status is Status.OK
    expected = _glm_cluster_natural(rows)
    assert relative_matrix_error(covariance_array(fit), expected) < 1e-5


def test_clustered_logit_covariance_matches_statsmodels_with_unbalanced_cluster_sizes() -> None:
    sizes = [4, 6, 8, 12, 16, 20, 24, 30, 5, 9, 14, 18]
    rows = _rows(clusters=len(sizes), seed=_SEED + 2, trials=sizes)

    fit = _fit(counts_obs(rows), link="logit")

    assert relative_matrix_error(covariance_array(fit), _glm_cluster_natural(rows)) < 1e-5


# -- independent numerical oracle: probit, cloglog, estimated asymptotes ------------------------


@pytest.mark.parametrize("link", ["probit", "cloglog", "logit"])
def test_clustered_covariance_matches_a_numerical_per_trial_oracle_for_each_link(
    link: _Link,
) -> None:
    rows = _rows(clusters=12, seed=_SEED + 3)
    fit = _fit(counts_obs(rows), link=link)

    assert fit.status is Status.OK
    oracle = oracle_sandwich(fit, rows)
    assert relative_matrix_error(covariance_array(fit), oracle.covariance) < 1e-4


def test_clustered_covariance_matches_the_oracle_on_a_linear_axis() -> None:
    rows = simulate_rows(
        _SEED + 4,
        clusters=12,
        levels=(1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.0),
        mu=7.0,
        s=3.0,
        scale="linear",
    )
    fit = _fit(counts_obs(rows, axis=LINEAR), link="probit")

    assert fit.status is Status.OK
    oracle = oracle_sandwich(fit, rows)
    assert relative_matrix_error(covariance_array(fit), oracle.covariance) < 1e-4


def test_clustered_covariance_matches_the_oracle_with_both_asymptotes_estimated() -> None:
    rows = simulate_rows(
        _SEED + 5, clusters=16, trials=24, levels=RICH_LEVELS, upper=0.93, lower=0.06
    )
    fit = _fit(counts_obs(rows), upper="estimate", lower="estimate")

    assert fit.status is Status.OK
    assert fit.covariance is not None and len(fit.covariance.names) == 4
    oracle = oracle_sandwich(fit, rows)
    # The estimated-asymptote path is differentiated in the optimiser chart, then mapped: allow
    # for the optimiser's own stopping tolerance.
    assert relative_matrix_error(covariance_array(fit), oracle.covariance) < 2e-3


def test_clustered_covariance_matches_the_oracle_with_only_the_upper_estimated() -> None:
    rows = simulate_rows(_SEED + 6, clusters=14, trials=24, levels=RICH_LEVELS, upper=0.9)
    fit = _fit(counts_obs(rows), upper="estimate", lower=0.0)

    assert fit.status is Status.OK
    oracle = oracle_sandwich(fit, rows)
    assert relative_matrix_error(covariance_array(fit), oracle.covariance) < 2e-3


# -- invariances -----------------------------------------------------------------------------


def test_clustered_covariance_is_the_same_for_aggregated_rows_and_one_row_per_trial() -> None:
    rows = _rows(clusters=10, seed=_SEED + 7, trials=8)

    aggregated = _fit(counts_obs(rows))
    per_trial = _fit(trial_obs(rows))

    assert relative_matrix_error(covariance_array(aggregated), covariance_array(per_trial)) < 1e-6


def test_clustered_covariance_does_not_depend_on_how_a_cluster_is_split_into_rows() -> None:
    rows = _rows(clusters=10, seed=_SEED + 8, trials=8)
    split: list[Row] = []
    for sev, name, k, n in rows:
        half_k = k // 2
        split += [(sev, name, half_k, n // 2), (sev, name, k - half_k, n - n // 2)]

    whole = _fit(counts_obs(rows))
    halves = _fit(counts_obs(split))

    assert relative_matrix_error(covariance_array(whole), covariance_array(halves)) < 1e-6


def test_clustered_covariance_is_invariant_to_relabelling_the_clusters() -> None:
    rows = _rows(clusters=11, seed=_SEED + 9)
    relabelled = [(sev, f"z{(97 * int(name[1:])) % 1000:04d}", k, n) for sev, name, k, n in rows]

    a = _fit(counts_obs(rows))
    b = _fit(counts_obs(relabelled))

    assert relative_matrix_error(covariance_array(a), covariance_array(b)) < 1e-9


def test_clustered_covariance_is_invariant_to_row_order() -> None:
    rows = _rows(clusters=11, seed=_SEED + 10)
    shuffled = [rows[i] for i in np.random.default_rng(3).permutation(len(rows))]

    a = _fit(counts_obs(rows))
    b = _fit(counts_obs(shuffled))

    assert relative_matrix_error(covariance_array(a), covariance_array(b)) < 1e-9


def test_clustered_covariance_of_an_increasing_fit_is_the_mirror_of_the_decreasing_fit() -> None:
    rows = _rows(clusters=12, seed=_SEED + 11)
    decreasing = _fit(counts_obs(rows))
    flipped = [(sev, name, n - k, n) for sev, name, k, n in rows]
    increasing = _fit(counts_obs(flipped, direction="increasing"))

    assert increasing.status is Status.OK
    assert relative_matrix_error(covariance_array(increasing), covariance_array(decreasing)) < 1e-8
    # cluster_cells are stored as counted, not mirrored.
    assert increasing.cluster_cells is not None
    assert sum(c.successes for c in increasing.cluster_cells) == sum(r[2] for r in flipped)


def test_clustered_covariance_of_an_increasing_fit_mirrors_estimated_asymptotes() -> None:
    """For an increasing outcome ``upper`` and ``lower`` are the complements of the decreasing
    fit's ``lower`` and ``upper``; the sandwich of the mirrored parameters is the decreasing
    sandwich with each asymptote's row and column negated, so absolute entries agree."""
    rows = simulate_rows(
        _SEED + 12, clusters=16, trials=24, levels=RICH_LEVELS, upper=0.93, lower=0.06
    )
    decreasing = _fit(counts_obs(rows), upper="estimate", lower="estimate")
    flipped = [(sev, name, n - k, n) for sev, name, k, n in rows]
    increasing = _fit(
        counts_obs(flipped, direction="increasing"), upper="estimate", lower="estimate"
    )

    assert increasing.status is Status.OK
    assert decreasing.covariance is not None and increasing.covariance is not None
    partner = {"mu": "mu", "s": "s", "upper": "lower", "lower": "upper"}
    dec_names = decreasing.covariance.names
    inc_names = increasing.covariance.names
    dec = covariance_array(decreasing)
    inc = covariance_array(increasing)
    for i, name_i in enumerate(inc_names):
        for j, name_j in enumerate(inc_names):
            expected = dec[dec_names.index(partner[name_i]), dec_names.index(partner[name_j])]
            assert abs(inc[i, j]) == pytest.approx(abs(expected), rel=1e-3, abs=1e-10)


# -- centering reuse ---------------------------------------------------------------------------


def test_clustered_covariance_is_right_when_clusters_observe_different_levels() -> None:
    """The conditioning centre and spread are those of the pooled design. Recomputing them on
    the per-(severity, cluster) rows would differ here, because several clusters skip the outer
    levels and so the row mean of ``log x`` is not the level mean; a score and a bread taken in
    different charts would then disagree with the numerical oracle."""
    drop = {f"c{i:02d}": {32.0, 16.0} for i in range(0, 8)} | {
        f"c{i:02d}": {1.0} for i in range(8, 12)
    }
    rows = _rows(clusters=14, seed=_SEED + 13, drop=drop)
    fit = _fit(counts_obs(rows))

    assert fit.status is Status.OK
    oracle = oracle_sandwich(fit, rows)
    assert relative_matrix_error(covariance_array(fit), oracle.covariance) < 1e-4


# -- guards ----------------------------------------------------------------------------------


def test_clustered_dependence_without_cluster_ids_raises() -> None:
    obs = counts_obs(_rows(), clustered=False)

    with pytest.raises(ValueError):
        _fit(obs)


def test_clustered_fit_with_no_more_clusters_than_parameters_raises_before_fitting() -> None:
    rows = _rows(clusters=2)

    with pytest.raises(ValueError, match="too few clusters"):
        _fit(counts_obs(rows))


def test_clustered_fit_with_estimated_asymptotes_needs_more_clusters_than_four() -> None:
    rows = simulate_rows(_SEED, clusters=4, trials=24, levels=RICH_LEVELS, upper=0.93, lower=0.06)

    with pytest.raises(ValueError, match="too few clusters"):
        _fit(counts_obs(rows), upper="estimate", lower="estimate")
