"""Paired ratios of two thresholds measured on the same clusters.

**Analytic intervals (``"log_delta"``, ``"fieller"`` with ``dependence="paired"``).** The
cross-covariance of the two thresholds comes from one stacked sandwich,
``(B_a + B_b) (G/(G-1) sum_c s_c s_c') (B_a + B_b)``, with ``s_c = (s_a,c, s_b,c)``. The oracle
here rebuilds all of it from a per-trial Bernoulli log-likelihood (``_cluster_support``), forms
``Var(theta_a)``, ``Var(theta_b)`` and ``Cov(theta_a, theta_b)`` by the delta method with a
numerical gradient of ``theta(phi)``, and applies Student ``t`` with ``G - 1`` degrees of
freedom. Scores are aligned by cluster id, never by position.

The private stacked-covariance helper is not named in the specification, so the check that the
recomputed ``V_a`` equals the stored ``Fit.covariance`` is made through the public results, with
the stored covariance on one side (``Threshold``'s own delta interval) and the oracle on the other.

**Cluster bootstrap.** Scripted replicate thresholds, as in ``test_cluster_bootstrap.py`` (the refit
entry is wrapped and returns scripted ``mu``/``upper``/``lower``). Outcomes ``"outcome_a"`` and
``"outcome_b"`` are scripted independently, by replicate order.

``LOW`` is a replicate with no crossing on the high side (``theta = +inf``), ``HIGH`` one with no
crossing on the low side (``theta = -inf``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pytest
from _cluster_support import (
    HIGH,
    LINEAR,
    LOG,
    LOW,
    Row,
    counts_obs,
    covariance_array,
    finite,
    install_scripts,
    oracle_sandwich,
    simulate_paired_rows,
    simulate_rows,
)
from scipy.stats import norm
from scipy.stats import t as student_t

from robustmargin import (
    Axis,
    Censoring,
    Definition,
    Fit,
    IntervalShape,
    Observations,
    Ratio,
    Status,
    Threshold,
    fit_dose_response,
    ratio_interval,
    threshold,
)

_SEED = 20261601
_LEVEL = 0.95


# -- builders --------------------------------------------------------------------------------


def _obs(rows: list[Row], outcome: str, axis: Axis = LOG) -> Observations:
    base = counts_obs(rows, axis=axis)
    return Observations(
        axis=base.axis,
        outcome=outcome,
        direction=base.direction,
        severity=base.severity,
        successes=base.successes,
        trials=base.trials,
        cluster=base.cluster,
    )


def _clustered_fit(
    rows: list[Row],
    outcome: str,
    *,
    axis: Axis = LOG,
    upper: float | Literal["estimate"] = 1.0,
    lower: float | Literal["estimate"] = 0.0,
) -> Fit:
    return fit_dose_response(
        _obs(rows, outcome, axis),
        model="binomial",
        link="probit",
        upper=upper,
        lower=lower,
        dependence="clustered",
    )


def _delta_threshold(fit: Fit, performance: float, level: float = _LEVEL) -> Threshold:
    return threshold(
        fit,
        definition=Definition.absolute(performance),
        interval_method="delta",
        level=level,
        dependence="clustered",
    )


def _pair(
    seed: int = _SEED,
    *,
    clusters: int = 12,
    rho: float = 0.6,
    performance: float = 0.5,
    permute_b: bool = False,
    **kwargs: Any,
) -> tuple[Threshold, Threshold, list[Row], list[Row]]:
    rows_a, rows_b = simulate_paired_rows(
        seed, clusters=clusters, rho=rho, mu_b=math.log(4.0), **kwargs
    )
    if permute_b:
        order = np.random.default_rng(seed).permutation(len(rows_b))
        shuffled = [rows_b[i] for i in order]
        # make the first-seen cluster order differ from the other fit's
        shuffled.sort(key=lambda r: r[1], reverse=True)
        fit_b = _clustered_fit(shuffled, "outcome_b")
    else:
        fit_b = _clustered_fit(rows_b, "outcome_b")
    fit_a = _clustered_fit(rows_a, "outcome_a")
    return (
        _delta_threshold(fit_a, performance),
        _delta_threshold(fit_b, performance),
        rows_a,
        rows_b,
    )


# -- the oracle ------------------------------------------------------------------------------


def _theta(fit: Fit, phi: dict[str, float], performance: float) -> float:
    z = float(norm.ppf((phi["upper"] - performance) / (phi["upper"] - phi["lower"])))
    return phi["mu"] + phi["s"] * z


def _theta_gradient(fit: Fit, names: tuple[str, ...], performance: float) -> np.ndarray:
    assert fit.params is not None
    phi = {name: p.value for name, p in fit.params.items()}
    gradient = np.zeros(len(names))
    for i, name in enumerate(names):
        h = 1e-6 * max(1.0, abs(phi[name]))
        up = dict(phi, **{name: phi[name] + h})
        down = dict(phi, **{name: phi[name] - h})
        gradient[i] = (_theta(fit, up, performance) - _theta(fit, down, performance)) / (2 * h)
    return gradient


def _oracle_moments(
    t_a: Threshold, t_b: Threshold, rows_a: list[Row], rows_b: list[Row], performance: float
) -> tuple[float, float, float, float, float]:
    """``(theta_a, theta_b, Var(theta_a), Var(theta_b), Cov(theta_a, theta_b))`` from the
    stacked per-trial sandwich, scores aligned by cluster id."""
    fit_a, fit_b = t_a.fit, t_b.fit
    assert fit_a.covariance is not None and fit_b.covariance is not None
    assert fit_a.params is not None and fit_b.params is not None
    oa = oracle_sandwich(fit_a, rows_a)
    ob = oracle_sandwich(fit_b, rows_b)
    index_b = {name: i for i, name in enumerate(ob.cluster_ids)}
    scores_b = ob.scores[[index_b[name] for name in oa.cluster_ids]]
    g = oa.clusters
    cross = oa.bread @ (g / (g - 1.0) * oa.scores.T @ scores_b) @ ob.bread
    ga = _theta_gradient(fit_a, fit_a.covariance.names, performance)
    gb = _theta_gradient(fit_b, fit_b.covariance.names, performance)
    theta_a = _theta(fit_a, {k: p.value for k, p in fit_a.params.items()}, performance)
    theta_b = _theta(fit_b, {k: p.value for k, p in fit_b.params.items()}, performance)
    return (
        theta_a,
        theta_b,
        float(ga @ oa.covariance @ ga),
        float(gb @ ob.covariance @ gb),
        float(ga @ cross @ gb),
    )


def _fieller_roots(
    a: float, b: float, va: float, vb: float, c: float, q: float
) -> tuple[float, float, float, float, float]:
    coef_a = b * b - q * q * vb
    coef_b = a * b - q * q * c
    coef_k = a * a - q * q * va
    disc = coef_b * coef_b - coef_a * coef_k
    root = math.sqrt(max(disc, 0.0))
    r1 = (coef_b - root) / coef_a
    r2 = (coef_b + root) / coef_a
    return coef_a, coef_b, coef_k, min(r1, r2), max(r1, r2)


# -- analytic paired intervals ---------------------------------------------------------------


@pytest.mark.parametrize("performance", [0.5, 0.7])
@pytest.mark.parametrize("rho", [0.0, 0.8])
def test_paired_log_delta_interval_matches_the_stacked_sandwich_oracle_with_t_quantile(
    performance: float, rho: float
) -> None:
    t_a, t_b, rows_a, rows_b = _pair(rho=rho, performance=performance)
    th_a, th_b, va, vb, cov = _oracle_moments(t_a, t_b, rows_a, rows_b, performance)

    result = ratio_interval(t_a, t_b, method="log_delta", dependence="paired", level=_LEVEL)

    q = float(student_t.ppf(0.5 + _LEVEL / 2.0, len(t_a.fit.cluster_ids or ()) - 1))
    se = math.sqrt(va + vb - 2.0 * cov)
    estimate = math.exp(th_a - th_b)
    assert result.dependence == "paired"
    assert result.method == "log_delta"
    assert result.shape is IntervalShape.BOUNDED
    assert result.estimate == pytest.approx(estimate, rel=1e-8)
    assert result.lo == pytest.approx(estimate * math.exp(-q * se), rel=1e-4)
    assert result.hi == pytest.approx(estimate * math.exp(q * se), rel=1e-4)
    assert result.bootstrap is None


def test_paired_log_delta_standard_error_lies_within_the_cauchy_schwarz_band() -> None:
    t_a, t_b, *_ = _pair(rho=0.9, clusters=14)

    paired = ratio_interval(t_a, t_b, method="log_delta", dependence="paired", level=_LEVEL)

    # The same two thresholds treated as independent differ only in C and in the quantile; the
    # paired standard error, recovered from the interval, is within the Cauchy-Schwarz band
    # |sd_a - sd_b| <= se <= sd_a + sd_b, with sd from each threshold's own clustered delta.
    assert paired.lo is not None and paired.hi is not None and paired.estimate is not None
    q = float(student_t.ppf(0.5 + _LEVEL / 2.0, 13))
    se = math.log(paired.hi / paired.lo) / (2.0 * q)
    sd_a = float(np.sqrt(covariance_array(t_a.fit)[0, 0]))
    sd_b = float(np.sqrt(covariance_array(t_b.fit)[0, 0]))
    assert abs(sd_a - sd_b) - 1e-9 <= se <= sd_a + sd_b + 1e-9


@pytest.mark.parametrize("performance", [0.5, 0.7])
def test_paired_fieller_interval_matches_the_stacked_sandwich_oracle_with_t_quantile(
    performance: float,
) -> None:
    t_a, t_b, rows_a, rows_b = _pair(rho=0.7, clusters=16, performance=performance)
    th_a, th_b, va, vb, cov = _oracle_moments(t_a, t_b, rows_a, rows_b, performance)
    a, b = math.exp(th_a), math.exp(th_b)
    q = float(student_t.ppf(0.5 + _LEVEL / 2.0, 15))
    coef_a, _, _, lo, hi = _fieller_roots(a, b, a * a * va, b * b * vb, a * b * cov, q)
    assert coef_a > 0.0, "fixture must have a bounded Fieller set"

    result = ratio_interval(t_a, t_b, method="fieller", dependence="paired", level=_LEVEL)

    assert result.shape is IntervalShape.BOUNDED
    assert result.lo == pytest.approx(max(lo, 0.0), rel=1e-4)
    assert result.hi == pytest.approx(hi, rel=1e-4)
    assert result.dependence == "paired"


def test_paired_intervals_use_the_cross_covariance_not_zero() -> None:
    t_a, t_b, *_ = _pair(rho=0.95, clusters=14, sigma=1.0)

    paired = ratio_interval(t_a, t_b, method="log_delta", dependence="paired", level=_LEVEL)
    assert paired.lo is not None and paired.hi is not None
    q = float(student_t.ppf(0.5 + _LEVEL / 2.0, 13))
    se_paired = math.log(paired.hi / paired.lo) / (2.0 * q)
    var_a = covariance_array(t_a.fit)[0, 0]
    var_b = covariance_array(t_b.fit)[0, 0]

    assert se_paired**2 != pytest.approx(var_a + var_b, rel=1e-3)


def test_paired_ratio_aligns_scores_by_cluster_id_not_by_position() -> None:
    t_a, t_b, *_ = _pair(rho=0.8)
    t_a2, t_b2, *_ = _pair(rho=0.8, permute_b=True)
    assert t_b2.fit.cluster_ids != t_b.fit.cluster_ids
    assert set(t_b2.fit.cluster_ids or ()) == set(t_b.fit.cluster_ids or ())

    reference = ratio_interval(t_a, t_b, method="log_delta", dependence="paired", level=_LEVEL)
    shuffled = ratio_interval(t_a2, t_b2, method="log_delta", dependence="paired", level=_LEVEL)

    assert shuffled.estimate == pytest.approx(reference.estimate, rel=1e-9)
    assert shuffled.lo == pytest.approx(reference.lo, rel=1e-7)
    assert shuffled.hi == pytest.approx(reference.hi, rel=1e-7)


def test_paired_log_delta_matches_the_oracle_when_the_fits_estimate_different_parameters() -> None:
    rows_a, rows_b = simulate_paired_rows(
        _SEED + 3,
        clusters=18,
        rho=0.7,
        trials=24,
        levels_a=(0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
        levels_b=(0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0, 64.0),
        mu_b=math.log(5.0),
    )
    rows_b = [
        (sev, name, int(round(0.93 * k)), n) for sev, name, k, n in rows_b
    ]  # a plateau below one: the second fit estimates its upper asymptote
    fit_a = _clustered_fit(rows_a, "outcome_a")
    fit_b = _clustered_fit(rows_b, "outcome_b", upper="estimate")
    assert fit_a.status is Status.OK and fit_b.status is Status.OK
    assert fit_a.covariance is not None and fit_b.covariance is not None
    assert len(fit_a.covariance.names) == 2 and len(fit_b.covariance.names) == 3
    t_a, t_b = _delta_threshold(fit_a, 0.5), _delta_threshold(fit_b, 0.5)
    th_a, th_b, va, vb, cov = _oracle_moments(t_a, t_b, rows_a, rows_b, 0.5)

    result = ratio_interval(t_a, t_b, method="log_delta", dependence="paired", level=_LEVEL)

    q = float(student_t.ppf(0.5 + _LEVEL / 2.0, 17))
    se = math.sqrt(va + vb - 2.0 * cov)
    estimate = math.exp(th_a - th_b)
    assert result.lo == pytest.approx(estimate * math.exp(-q * se), rel=5e-3)
    assert result.hi == pytest.approx(estimate * math.exp(q * se), rel=5e-3)


def test_paired_fieller_reaches_the_exclusive_shape_on_a_constructed_case() -> None:
    """Both thresholds poorly determined and strongly related: ``A < 0``, ``K < 0`` and
    ``B < 0`` in the Fieller quadratic ``A r^2 - 2 B r + K <= 0``, which puts both roots on one
    positive side. Found by scanning seeds with the oracle above (seed 92: five clusters of
    twelve trials, a flat curve against a steep one, cluster effects correlated at 0.9); the
    fixture asserts the oracle's own shape first, so a drifting generator fails loudly here and
    not in the implementation."""
    rows_a, rows_b = simulate_paired_rows(
        92,
        clusters=5,
        rho=0.9,
        sigma=1.2,
        trials=12,
        levels_a=(0.0, 2.0, 4.0, 8.0, 16.0, 32.0),
        levels_b=(0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
        mu_a=math.log(6.0),
        mu_b=math.log(5.0),
        s_a=1.2,
        s_b=0.4,
    )
    t_a = _delta_threshold(_clustered_fit(rows_a, "outcome_a"), 0.5)
    t_b = _delta_threshold(_clustered_fit(rows_b, "outcome_b"), 0.5)
    th_a, th_b, va, vb, cov = _oracle_moments(t_a, t_b, rows_a, rows_b, 0.5)
    a, b = math.exp(th_a), math.exp(th_b)
    q = float(student_t.ppf(0.5 + _LEVEL / 2.0, 4))
    coef_a, coef_b, coef_k, lo, hi = _fieller_roots(a, b, a * a * va, b * b * vb, a * b * cov, q)
    assert coef_a < 0.0 and coef_k < 0.0 and coef_b < 0.0, "oracle no longer sees EXCLUSIVE"

    result = ratio_interval(t_a, t_b, method="fieller", dependence="paired", level=_LEVEL)

    assert result.shape is IntervalShape.EXCLUSIVE
    assert result.lo == pytest.approx(lo, rel=1e-3)
    assert result.hi == pytest.approx(hi, rel=1e-3)


# -- few clusters on a paired ratio ----------------------------------------------------------


def test_paired_ratio_carries_the_few_clusters_token_and_adds_none_of_its_own() -> None:
    t_a, t_b, *_ = _pair(clusters=4, rho=0.5, trials=40)
    carried = [w for t in (t_a, t_b) for w in t.warnings if w.startswith("FEW_CLUSTERS:")]
    assert carried, "the inputs must carry the token for this check to mean anything"

    result = ratio_interval(t_a, t_b, method="log_delta", dependence="paired", level=_LEVEL)

    ratio_tokens = [w for w in result.warnings if w.startswith("FEW_CLUSTERS:")]
    assert 1 <= len(ratio_tokens) <= len(carried)


# -- paired cluster bootstrap on scripted replicates -------------------------------------------

B = 200  # replicates
LEVEL90 = 0.90  # tail 5% = 10 replicates; give-up above 5% failed


def _ratio_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    script_a: Callable[[int], tuple[float, float, float]],
    script_b: Callable[[int], tuple[float, float, float]],
    *,
    axis: Axis = LOG,
    n_resamples: int = B,
    level: float = LEVEL90,
) -> Ratio:
    if axis.scale == "log":
        rows_a, rows_b = simulate_paired_rows(_SEED + 5, clusters=12, rho=0.5, mu_b=math.log(4.0))
    else:
        rows_a = simulate_rows(
            _SEED + 6,
            clusters=12,
            levels=(1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.0),
            mu=7.0,
            s=3.0,
            scale="linear",
        )
        rows_b = simulate_rows(
            _SEED + 7,
            clusters=12,
            levels=(1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.0),
            mu=5.0,
            s=3.0,
            scale="linear",
        )
    t_a = _delta_threshold(_clustered_fit(rows_a, "outcome_a", axis=axis), 0.5, level)
    t_b = _delta_threshold(_clustered_fit(rows_b, "outcome_b", axis=axis), 0.5, level)
    install_scripts(monkeypatch, {"outcome_a": script_a, "outcome_b": script_b})
    return ratio_interval(
        t_a,
        t_b,
        method="cluster_bootstrap",
        dependence="paired",
        level=level,
        n_resamples=n_resamples,
        seed=_SEED,
    )


def _weibull(values: list[float], p: float) -> float:
    return float(np.quantile(np.array(sorted(values)), p, method="weibull"))


def test_paired_bootstrap_on_a_log_axis_takes_percentiles_of_the_theta_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n = 199
    theta_a = [1.0 + 0.01 * (i % 37) for i in range(n)]
    theta_b = [0.2 + 0.004 * i for i in range(n)]
    diffs = [a - b for a, b in zip(theta_a, theta_b, strict=True)]

    result = _ratio_bootstrap(
        monkeypatch,
        lambda i: finite(theta_a[i]),
        lambda i: finite(theta_b[i]),
        n_resamples=n,
        level=_LEVEL,
    )

    assert result.method == "cluster_bootstrap" and result.dependence == "paired"
    assert result.shape is IntervalShape.BOUNDED
    assert result.lo == pytest.approx(math.exp(_weibull(diffs, 0.025)), rel=1e-12)
    assert result.hi == pytest.approx(math.exp(_weibull(diffs, 0.975)), rel=1e-12)
    assert result.bootstrap is not None
    assert (result.bootstrap.n_resamples, result.bootstrap.seed) == (n, _SEED)
    assert result.bootstrap.failed == 0


def test_paired_bootstrap_on_a_linear_axis_takes_percentiles_of_the_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n = 199
    theta_a = [6.0 + 0.02 * (i % 31) for i in range(n)]
    theta_b = [3.0 + 0.01 * i for i in range(n)]
    ratios = [a / b for a, b in zip(theta_a, theta_b, strict=True)]

    result = _ratio_bootstrap(
        monkeypatch,
        lambda i: finite(theta_a[i]),
        lambda i: finite(theta_b[i]),
        axis=LINEAR,
        n_resamples=n,
        level=_LEVEL,
    )

    assert result.shape is IntervalShape.BOUNDED
    assert result.lo == pytest.approx(_weibull(ratios, 0.025), rel=1e-12)
    assert result.hi == pytest.approx(_weibull(ratios, 0.975), rel=1e-12)


def _mixed(first: tuple[float, float, float], count: int) -> Callable[[int], Any]:
    return lambda i: first if i < count else finite(1.0 + 0.001 * (i % 50))


def test_paired_bootstrap_upper_open_shape_is_half_open_with_a_lower_bound_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _ratio_bootstrap(monkeypatch, _mixed(LOW, 15), lambda i: finite(0.5))

    assert result.shape is IntervalShape.HALF_OPEN
    assert result.lo is not None and result.hi is None
    assert result.censoring is Censoring.NONE
    assert result.bootstrap is not None
    assert result.bootstrap.unreachable_high == 15 and result.bootstrap.unreachable_low == 0


def test_paired_bootstrap_lower_open_shape_is_bounded_from_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _ratio_bootstrap(monkeypatch, _mixed(HIGH, 15), lambda i: finite(0.5))

    assert result.shape is IntervalShape.BOUNDED
    assert result.lo == 0.0
    assert result.hi is not None and result.hi > 0.0
    assert result.censoring is Censoring.NONE
    assert result.bootstrap is not None
    assert result.bootstrap.unreachable_low == 15 and result.bootstrap.unreachable_high == 0


def test_paired_bootstrap_with_both_sides_open_is_unbounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def script_a(i: int) -> tuple[float, float, float]:
        if i < 15:
            return LOW
        if i < 30:
            return HIGH
        return finite(1.0 + 0.001 * i)

    result = _ratio_bootstrap(monkeypatch, script_a, lambda i: finite(0.5))

    assert result.shape is IntervalShape.UNBOUNDED
    assert result.lo is None and result.hi is None
    assert result.estimate is not None


@pytest.mark.parametrize(
    ("a_replicate", "b_replicate", "side"),
    [
        (LOW, finite(0.5), "high"),  # inf / finite
        (HIGH, finite(0.5), "low"),  # 0 / finite
        (finite(1.0), LOW, "low"),  # finite / inf
        (finite(1.0), HIGH, "high"),  # finite / 0
        (LOW, HIGH, "high"),  # inf / 0: opposite sides are defined
        (HIGH, LOW, "low"),  # 0 / inf
    ],
)
def test_paired_bootstrap_counts_an_unreachable_threshold_on_the_ratios_own_side(
    monkeypatch: pytest.MonkeyPatch,
    a_replicate: tuple[float, float, float],
    b_replicate: tuple[float, float, float],
    side: str,
) -> None:
    count = 6  # 3% of 200: below the 5% give-up cut-off at level 0.90
    result = _ratio_bootstrap(
        monkeypatch,
        lambda i: a_replicate if i < count else finite(1.0 + 0.001 * i),
        lambda i: b_replicate if i < count else finite(0.5),
    )

    assert result.bootstrap is not None
    assert result.bootstrap.failed == 0
    assert result.bootstrap.unreachable_high == (count if side == "high" else 0)
    assert result.bootstrap.unreachable_low == (count if side == "low" else 0)
    assert result.status is Status.OK


@pytest.mark.parametrize("both", [LOW, HIGH], ids=["inf_over_inf", "zero_over_zero"])
def test_paired_bootstrap_counts_a_same_side_pair_as_failed_not_unreachable(
    monkeypatch: pytest.MonkeyPatch, both: tuple[float, float, float]
) -> None:
    count = 6
    result = _ratio_bootstrap(
        monkeypatch,
        lambda i: both if i < count else finite(1.0 + 0.001 * i),
        lambda i: both if i < count else finite(0.5),
    )

    assert result.bootstrap is not None
    assert result.bootstrap.failed == count
    assert result.bootstrap.unreachable_high == 0 and result.bootstrap.unreachable_low == 0
    assert any(w.startswith("BOOTSTRAP_FAILURES:") for w in result.warnings)


def test_paired_bootstrap_counts_a_failed_refit_of_either_fit_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from robustmargin import models as models_module

    real = models_module._fit_counts
    calls = {"outcome_b": 0}

    def flaky(obs: Observations, **kwargs: Any) -> Fit:
        fit = real(obs, **kwargs)
        if obs.outcome == "outcome_b":
            calls["outcome_b"] += 1
            if calls["outcome_b"] <= 6:
                return replace(
                    fit,
                    status=Status.NOT_CONVERGED,
                    params=None,
                    covariance=None,
                    log_likelihood=None,
                    diagnostics=None,
                    warnings=("scripted",),
                )
        return fit

    rows_a, rows_b = simulate_paired_rows(_SEED + 5, clusters=12, rho=0.5, mu_b=math.log(4.0))
    t_a = _delta_threshold(_clustered_fit(rows_a, "outcome_a"), 0.5, LEVEL90)
    t_b = _delta_threshold(_clustered_fit(rows_b, "outcome_b"), 0.5, LEVEL90)
    from robustmargin import intervals as intervals_module
    from robustmargin import ratio as ratio_module

    for module in (models_module, intervals_module, ratio_module):
        monkeypatch.setattr(module, "_fit_counts", flaky, raising=False)

    result = ratio_interval(
        t_a,
        t_b,
        method="cluster_bootstrap",
        dependence="paired",
        level=LEVEL90,
        n_resamples=B,
        seed=_SEED,
    )

    assert result.bootstrap is not None
    assert result.bootstrap.failed >= 6


def test_paired_bootstrap_gives_up_above_the_failed_fraction_with_no_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _ratio_bootstrap(
        monkeypatch,
        lambda i: LOW if i < 40 else finite(1.0 + 0.001 * i),
        lambda i: LOW if i < 40 else finite(0.5),
    )

    assert result.status is Status.NOT_CONVERGED
    assert (result.estimate, result.lo, result.hi) == (None, None, None)
    assert result.bootstrap is not None and result.bootstrap.failed == 40
    assert result.warnings


def test_paired_bootstrap_flags_a_censored_share_above_five_percent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _ratio_bootstrap(monkeypatch, _mixed(LOW, 16), lambda i: finite(0.5))  # 8%

    tokens = [w for w in result.warnings if w.startswith("RATIO_BOOTSTRAP_CENSORED:")]
    assert len(tokens) == 1


def test_paired_bootstrap_does_not_flag_a_censored_share_of_five_percent_or_less(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _ratio_bootstrap(monkeypatch, _mixed(LOW, 8), lambda i: finite(0.5))  # 4%

    assert [w for w in result.warnings if w.startswith("RATIO_BOOTSTRAP_CENSORED:")] == []


def test_paired_bootstrap_on_a_linear_axis_maps_a_negative_crossing_to_a_zero_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _ratio_bootstrap(
        monkeypatch,
        lambda i: finite(-1.0) if i < 15 else finite(6.0 + 0.01 * i),
        lambda i: finite(3.0 + 0.005 * i),
        axis=LINEAR,
    )

    assert result.shape is IntervalShape.BOUNDED
    assert result.lo == 0.0
    assert result.bootstrap is not None and result.bootstrap.unreachable_low == 15


def test_paired_bootstrap_aligns_cluster_draws_by_id_when_the_fits_list_clusters_differently() -> (
    None
):
    """One draw of G cluster ids applies to both fits. With the second fit's clusters listed in
    another order, a positional pairing would draw different clusters for the two fits and the
    interval would change; an id-aligned pairing is unchanged under one seed."""
    t_a, t_b, *_ = _pair(rho=0.9, clusters=10)
    t_a2, t_b2, *_ = _pair(rho=0.9, clusters=10, permute_b=True)
    assert t_b2.fit.cluster_ids != t_b.fit.cluster_ids

    def run(x: Threshold, y: Threshold) -> Ratio:
        return ratio_interval(
            x,
            y,
            method="cluster_bootstrap",
            dependence="paired",
            level=_LEVEL,
            n_resamples=99,
            seed=_SEED,
        )

    reference = run(t_a, t_b)
    shuffled = run(t_a2, t_b2)

    assert reference.lo is not None
    assert shuffled.estimate == pytest.approx(reference.estimate, rel=1e-9)
    assert shuffled.lo == pytest.approx(reference.lo, rel=1e-9)
    assert shuffled.hi == pytest.approx(reference.hi, rel=1e-9)
    assert shuffled.bootstrap == reference.bootstrap


def test_paired_bootstrap_is_deterministic_under_a_seed() -> None:
    t_a, t_b, *_ = _pair(rho=0.5, clusters=10)

    def run(seed: int) -> Ratio:
        return ratio_interval(
            t_a,
            t_b,
            method="cluster_bootstrap",
            dependence="paired",
            level=_LEVEL,
            n_resamples=99,
            seed=seed,
        )

    assert run(3) == run(3)
    assert (run(3).lo, run(3).hi) != (run(4).lo, run(4).hi)


# -- the paired cross-check (log axis: log_delta) ---------------------------------------------


def _paired_cross_check(monkeypatch: pytest.MonkeyPatch, *, scale: float, shift: float) -> Ratio:
    rows_a, rows_b = simulate_paired_rows(_SEED + 5, clusters=12, rho=0.5, mu_b=math.log(4.0))
    t_a = _delta_threshold(_clustered_fit(rows_a, "outcome_a"), 0.5)
    t_b = _delta_threshold(_clustered_fit(rows_b, "outcome_b"), 0.5)
    th_a, th_b, va, vb, cov = _oracle_moments(t_a, t_b, rows_a, rows_b, 0.5)
    se = math.sqrt(va + vb - 2.0 * cov)
    n = 399
    z = norm.ppf((np.arange(n) + 0.5) / n)
    diffs = (th_a - th_b) + shift * se + scale * se * z
    install_scripts(
        monkeypatch,
        {"outcome_a": lambda i: finite(th_b + diffs[i]), "outcome_b": lambda i: finite(th_b)},
    )
    return ratio_interval(
        t_a,
        t_b,
        method="cluster_bootstrap",
        dependence="paired",
        level=_LEVEL,
        n_resamples=n,
        seed=_SEED,
    )


def _cross_check(result: Ratio) -> list[str]:
    return [w for w in result.warnings if w.startswith("CLUSTER_CROSS_CHECK:")]


def test_paired_cross_check_is_silent_when_the_bootstrap_matches_the_paired_log_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _cross_check(_paired_cross_check(monkeypatch, scale=1.0, shift=0.0)) == []


@pytest.mark.parametrize(("scale", "shift"), [(0.01, 0.0), (100.0, 0.0), (1.0, 5.0)])
def test_paired_cross_check_flags_a_bootstrap_far_from_the_paired_log_delta(
    monkeypatch: pytest.MonkeyPatch, scale: float, shift: float
) -> None:
    result = _paired_cross_check(monkeypatch, scale=scale, shift=shift)

    assert len(_cross_check(result)) == 1
    assert result.lo is not None
