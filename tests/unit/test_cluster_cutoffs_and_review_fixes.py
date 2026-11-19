"""Regression tests for the clustered-inference cutoffs and review fixes: few-clusters cutoffs,
the scaled cross-check, Student-t independent ratios of clustered thresholds, per-replicate
failure-cause counts, the Fieller ``A == 0``, ``B < 0`` branch, and the stored-covariance
recomputation tolerance.

The cutoffs and the cross-check bounds are read from the modules, so a change to a constant moves
the tests with it; the boundary cases are written as ``cutoff - 1`` and ``cutoff``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from _cluster_fixtures import clustered_fit, clustered_threshold, paired_fits
from _cluster_support import (
    LOG,
    Row,
    counts_obs,
    simulate_paired_rows,
)
from scipy.stats import norm
from scipy.stats import t as student_t

from robustmargin import (
    Definition,
    Fit,
    IntervalShape,
    Status,
    Threshold,
    fit_dose_response,
    ratio_interval,
    threshold,
)
from robustmargin import intervals as intervals_module
from robustmargin import models as models_module
from robustmargin.intervals import _cross_check_warning
from robustmargin.models import Covariance
from robustmargin.ratio import _fieller_shape

PREFIX = "FEW_CLUSTERS:"
SANDWICH = models_module._FEW_CLUSTERS_SANDWICH
BOOTSTRAP = models_module._FEW_CLUSTERS_BOOTSTRAP
LEVEL = 0.95


def _few(warnings: tuple[str, ...]) -> list[str]:
    return [w for w in warnings if w.startswith(PREFIX)]


# -- 1. few-clusters cutoffs -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("clusters", "warns"), [(SANDWICH - 1, True), (SANDWICH, False)], ids=["below", "at_cutoff"]
)
def test_clustered_fit_warns_few_clusters_only_below_the_sandwich_cutoff(
    clusters: int, warns: bool
) -> None:
    fit = clustered_fit(clusters)

    assert bool(_few(fit.warnings)) is warns


@pytest.mark.parametrize(
    ("clusters", "warns"), [(SANDWICH - 1, True), (SANDWICH, False)], ids=["below", "at_cutoff"]
)
def test_delta_threshold_warns_few_clusters_only_below_the_sandwich_cutoff(
    clusters: int, warns: bool
) -> None:
    t = clustered_threshold(clustered_fit(clusters))

    assert bool(_few(t.warnings)) is warns


@pytest.mark.parametrize(
    ("clusters", "warns"),
    [(BOOTSTRAP - 1, True), (BOOTSTRAP, False)],
    ids=["below", "at_cutoff"],
)
def test_cluster_bootstrap_threshold_warns_only_below_the_bootstrap_cutoff(
    clusters: int, warns: bool
) -> None:
    t = clustered_threshold(
        clustered_fit(clusters), interval_method="cluster_bootstrap", n_resamples=30, seed=7
    )

    few = _few(t.warnings)
    assert bool(few) is warns
    if warns:
        assert len(few) == 1
        assert "under-covers" in few[0]
        assert "delta" in few[0]


def test_cluster_bootstrap_threshold_between_cutoffs_gets_the_bootstrap_text() -> None:
    # G in [SANDWICH, BOOTSTRAP): the sandwich is fine, the percentile bootstrap is not.
    t = clustered_threshold(
        clustered_fit(SANDWICH), interval_method="cluster_bootstrap", n_resamples=30, seed=7
    )

    few = _few(t.warnings)
    assert len(few) == 1
    assert "under-covers" in few[0]


def _paired(g: int) -> tuple[Threshold, Threshold]:
    fit_a, fit_b = paired_fits(g)
    return clustered_threshold(fit_a), clustered_threshold(fit_b)


def test_paired_cluster_bootstrap_below_the_bootstrap_cutoff_carries_one_bootstrap_text() -> None:
    # G below both cutoffs: the inputs carry the sandwich-style text, the ratio replaces it.
    g = SANDWICH - 1
    t_a, t_b = _paired(g)
    assert _few(t_a.warnings), "the inputs must carry the sandwich text for this check to bite"

    result = ratio_interval(
        t_a,
        t_b,
        method="cluster_bootstrap",
        dependence="paired",
        level=LEVEL,
        n_resamples=20,
        seed=3,
    )

    few = _few(result.warnings)
    assert len(few) == 1
    assert "under-covers" in few[0]
    assert "unreliable below" not in few[0]


def test_paired_cluster_bootstrap_at_the_bootstrap_cutoff_has_no_few_clusters_text() -> None:
    t_a, t_b = _paired(BOOTSTRAP)

    result = ratio_interval(
        t_a,
        t_b,
        method="cluster_bootstrap",
        dependence="paired",
        level=LEVEL,
        n_resamples=20,
        seed=3,
    )

    assert _few(result.warnings) == []


@pytest.mark.parametrize("method", ["log_delta", "fieller"])
def test_paired_analytic_ratio_below_the_sandwich_cutoff_carries_the_text_once(
    method: str,
) -> None:
    t_a, t_b = _paired(SANDWICH - 1)

    result = ratio_interval(t_a, t_b, method=method, dependence="paired", level=LEVEL)

    few = _few(result.warnings)
    assert len(few) == 1
    assert "unreliable below" in few[0]


@pytest.mark.parametrize("method", ["log_delta", "fieller"])
def test_paired_analytic_ratio_at_the_sandwich_cutoff_has_no_few_clusters_text(
    method: str,
) -> None:
    t_a, t_b = _paired(SANDWICH)

    result = ratio_interval(t_a, t_b, method=method, dependence="paired", level=LEVEL)

    assert _few(result.warnings) == []


# -- 2. cross-check scaling ----------------------------------------------------------------------

_THETAS = np.linspace(-1.0, 1.0, 101)  # symmetric about 0: median 0, so no shift
_SPREAD = float(np.subtract(*np.percentile(_THETAS, [75.0, 25.0]))) / 1.349
_CUTOFF = intervals_module._CROSS_CHECK_SPREAD_RATIO


def _scaled_ratio_check(scaled_ratio: float, clusters: int) -> str | None:
    """Standard error chosen so ``spread / se * sqrt(G / (G - 1))`` equals ``scaled_ratio``."""
    standard_error = _SPREAD * math.sqrt(clusters / (clusters - 1)) / scaled_ratio
    return _cross_check_warning(
        _THETAS, theta_hat=0.0, standard_error=standard_error, clusters=clusters
    )


@pytest.mark.parametrize("clusters", [2, 5, 30])
def test_cross_check_flags_a_scaled_spread_ratio_just_above_the_upper_bound(
    clusters: int,
) -> None:
    note = _scaled_ratio_check(_CUTOFF * 1.02, clusters)

    assert note is not None and note.startswith("CLUSTER_CROSS_CHECK: ")


@pytest.mark.parametrize("clusters", [2, 5, 30])
def test_cross_check_accepts_a_scaled_spread_ratio_just_below_the_upper_bound(
    clusters: int,
) -> None:
    assert _scaled_ratio_check(_CUTOFF * 0.98, clusters) is None


@pytest.mark.parametrize("clusters", [2, 5, 30])
def test_cross_check_flags_a_scaled_spread_ratio_just_below_the_lower_bound(
    clusters: int,
) -> None:
    assert _scaled_ratio_check(0.98 / _CUTOFF, clusters) is not None


@pytest.mark.parametrize("clusters", [2, 5, 30])
def test_cross_check_accepts_a_scaled_spread_ratio_just_above_the_lower_bound(
    clusters: int,
) -> None:
    assert _scaled_ratio_check(1.02 / _CUTOFF, clusters) is None


def test_cross_check_scales_the_spread_by_the_cluster_count_not_by_the_raw_ratio() -> None:
    # G = 2 scales by sqrt(2) ~ 1.41. A raw ratio of 1.02 * cutoff / 1.41 is below the bound, but
    # its scaled value is above it, so only a scaled comparison flags it.
    clusters = 2
    standard_error = _SPREAD / (_CUTOFF * 1.02 / math.sqrt(2.0))  # raw ratio < cutoff
    assert _SPREAD / standard_error < _CUTOFF

    note = _cross_check_warning(
        _THETAS, theta_hat=0.0, standard_error=standard_error, clusters=clusters
    )

    assert note is not None
    assert "sqrt(G / (G - 1))" in note


# -- 3. independent ratios of clustered thresholds ----------------------------------------------


def _log_se(t: Threshold, df: float) -> float:
    """Standard error of ``log(value)`` recovered from the threshold's own delta interval, whose
    half-width on the log axis is ``q * se`` with ``q`` Student ``t`` on ``df`` (normal if
    infinite)."""
    assert t.lo is not None and t.hi is not None
    q = norm.ppf(0.5 + LEVEL / 2.0) if math.isinf(df) else student_t.ppf(0.5 + LEVEL / 2.0, df)
    return float((math.log(t.hi) - math.log(t.lo)) / (2.0 * q))


def _welch(va: float, dfa: float, vb: float, dfb: float) -> float:
    return (va + vb) ** 2 / (va**2 / dfa + vb**2 / dfb)


def _independent_pair(clustered_b: bool) -> tuple[Threshold, Threshold, float, float]:
    g_a, g_b = 8, 11
    fit_a = clustered_fit(g_a, prefix="a", base="a")
    t_a = clustered_threshold(fit_a)
    if clustered_b:
        fit_b = clustered_fit(g_b, prefix="b", base="b")
        t_b = clustered_threshold(fit_b)
        df_b = float(g_b - 1)
    else:
        fit_b = clustered_fit(g_b, prefix="b", base="b", dependence="independent")
        t_b = threshold(
            fit_b,
            definition=Definition.absolute(0.5),
            interval_method="delta",
            level=LEVEL,
            dependence="independent",
        )
        df_b = math.inf
    return t_a, t_b, float(g_a - 1), df_b


@pytest.mark.parametrize("clustered_b", [True, False], ids=["both_clustered", "one_unclustered"])
def test_independent_log_delta_ratio_of_clustered_thresholds_uses_welch_satterthwaite_t(
    clustered_b: bool,
) -> None:
    t_a, t_b, df_a, df_b = _independent_pair(clustered_b)
    assert t_a.value is not None and t_b.value is not None
    se_a, se_b = _log_se(t_a, df_a), _log_se(t_b, df_b)
    variance = se_a**2 + se_b**2
    df = _welch(se_a**2, df_a, se_b**2, df_b)
    q = student_t.ppf(0.5 + LEVEL / 2.0, df)
    centre = math.log(t_a.value / t_b.value)

    result = ratio_interval(t_a, t_b, method="log_delta", dependence="independent", level=LEVEL)

    assert result.lo == pytest.approx(math.exp(centre - q * math.sqrt(variance)), rel=1e-3)
    assert result.hi == pytest.approx(math.exp(centre + q * math.sqrt(variance)), rel=1e-3)


@pytest.mark.parametrize("clustered_b", [True, False], ids=["both_clustered", "one_unclustered"])
def test_independent_fieller_ratio_of_clustered_thresholds_uses_welch_satterthwaite_t(
    clustered_b: bool,
) -> None:
    t_a, t_b, df_a, df_b = _independent_pair(clustered_b)
    assert t_a.value is not None and t_b.value is not None
    a, b = t_a.value, t_b.value
    va = (a * _log_se(t_a, df_a)) ** 2
    vb = (b * _log_se(t_b, df_b)) ** 2
    rho = a / b
    df = _welch(va, df_a, rho**2 * vb, df_b)
    q = student_t.ppf(0.5 + LEVEL / 2.0, df)
    coef_a, coef_b, coef_k = b * b - q * q * vb, a * b, a * a - q * q * va
    assert coef_a > 0.0
    root = math.sqrt(coef_b**2 - coef_a * coef_k)
    expected_lo = max((coef_b - root) / coef_a, 0.0)
    expected_hi = (coef_b + root) / coef_a

    result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=LEVEL)

    assert result.shape is IntervalShape.BOUNDED
    assert result.lo == pytest.approx(expected_lo, rel=1e-3)
    assert result.hi == pytest.approx(expected_hi, rel=1e-3)


def test_independent_ratio_of_clustered_thresholds_is_wider_than_the_normal_interval() -> None:
    t_a, t_b, df_a, df_b = _independent_pair(True)
    assert t_a.value is not None and t_b.value is not None
    se_a, se_b = _log_se(t_a, df_a), _log_se(t_b, df_b)
    z_half = norm.ppf(0.5 + LEVEL / 2.0) * math.sqrt(se_a**2 + se_b**2)
    centre = math.log(t_a.value / t_b.value)

    result = ratio_interval(t_a, t_b, method="log_delta", dependence="independent", level=LEVEL)

    assert result.hi is not None and result.lo is not None
    assert math.log(result.hi) - centre > z_half
    assert centre - math.log(result.lo) > z_half


# -- 4. paired bootstrap failure causes are counted per replicate --------------------------------


def _paired_clustered_thresholds() -> tuple[Threshold, Threshold]:
    rows_a, rows_b = simulate_paired_rows(20261118, clusters=12, rho=0.5, mu_b=math.log(4.0))

    def fit(rows: list[Row], outcome: str) -> Fit:
        base = counts_obs(rows, axis=LOG)
        obs = replace(base, outcome=outcome)
        return fit_dose_response(
            obs, model="binomial", link="probit", upper=1.0, lower=0.0, dependence="clustered"
        )

    return (
        clustered_threshold(fit(rows_a, "outcome_a")),
        clustered_threshold(fit(rows_b, "outcome_b")),
    )


def test_paired_bootstrap_counts_a_cause_shared_by_both_fits_once_per_replicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_resamples, bad = 200, 3
    design = intervals_module._BootstrapDesign
    real = design._baseline_is_undefined
    calls = {"n": 0}

    def flaky(self: Any, *args: Any, **kwargs: Any) -> bool:
        # Each replicate reads the first fit's baseline and then the second's, so calls
        # 2i and 2i + 1 belong to replicate i: the first `bad` replicates fail in both fits.
        index = calls["n"] // 2
        calls["n"] += 1
        return True if index < bad else bool(real(self, *args, **kwargs))

    t_a, t_b = _paired_clustered_thresholds()
    # Every refit returns the original fit of its outcome, so only the failure counting is under
    # test and the 200 replicates cost nothing.
    originals = {"outcome_a": t_a.fit, "outcome_b": t_b.fit}

    def cheap_refit(obs: Any, **kwargs: Any) -> Fit:
        return originals[obs.outcome]

    monkeypatch.setattr(design, "_baseline_is_undefined", flaky)
    for module in (models_module, intervals_module):
        monkeypatch.setattr(module, "_fit_counts", cheap_refit, raising=False)
    calls["n"] = 0

    result = ratio_interval(
        t_a,
        t_b,
        method="cluster_bootstrap",
        dependence="paired",
        level=LEVEL,
        n_resamples=n_resamples,
        seed=11,
    )

    assert result.bootstrap is not None
    assert result.bootstrap.failed == bad
    text = [w for w in result.warnings if w.startswith("BOOTSTRAP_FAILURES:")]
    assert len(text) == 1
    assert f"{bad} of them because a replicate's baseline was undefined" in text[0]
    assert f"{2 * bad} of them" not in text[0]


# -- 5. Fieller with A == 0 ----------------------------------------------------------------------
# z = 2, theta_b = 2, Vb = 1 give A = theta_b^2 - z^2 Vb = 0 exactly.


def test_fieller_with_a_zero_and_b_negative_is_bounded_from_zero_to_k_over_two_b() -> None:
    # B = 1 * 2 - 4 * 1 = -2; K = 1 - 4 * 1 = -3; K / (2B) = 0.75.
    shape, lo, hi, _ = _fieller_shape(theta_a=1.0, va=1.0, theta_b=2.0, vb=1.0, z=2.0, cov=1.0)

    assert shape is IntervalShape.BOUNDED
    assert lo == 0.0
    assert hi == pytest.approx(0.75)
    assert hi is not None
    assert lo <= 1.0 / 2.0 <= hi


def test_fieller_with_a_zero_b_negative_and_k_positive_raises() -> None:
    with pytest.raises(ValueError):
        _fieller_shape(theta_a=1.0, va=0.1, theta_b=2.0, vb=1.0, z=2.0, cov=1.0)


def test_fieller_with_a_and_b_zero_and_k_positive_raises() -> None:
    # B = 1 * 2 - 4 * 0.5 = 0 and K = 1 - 4 * 0.1 > 0: an empty set.
    with pytest.raises(ValueError):
        _fieller_shape(theta_a=1.0, va=0.1, theta_b=2.0, vb=1.0, z=2.0, cov=0.5)


# -- 6. stored-covariance recomputation tolerance ------------------------------------------------


def _scaled_covariance(t: Threshold, factor: float) -> Threshold:
    """``t`` with its fit's stored covariance multiplied by ``factor``; every entry then differs
    from the true one by ``factor - 1`` of its own scale."""
    assert t.fit.covariance is not None
    matrix = tuple(tuple(v * factor for v in row) for row in t.fit.covariance.matrix)
    covariance = Covariance(names=t.fit.covariance.names, matrix=matrix)
    return replace(t, fit=replace(t.fit, covariance=covariance))


def test_paired_ratio_accepts_a_stored_covariance_perturbed_by_1e_minus_5() -> None:
    t_a, t_b = _paired(12)
    baseline = ratio_interval(t_a, t_b, method="log_delta", dependence="paired", level=LEVEL)

    result = ratio_interval(
        _scaled_covariance(t_a, 1.0 + 1e-5),
        t_b,
        method="log_delta",
        dependence="paired",
        level=LEVEL,
    )

    assert result.status is Status.OK
    assert result.lo == pytest.approx(baseline.lo, rel=1e-3)
    assert result.hi == pytest.approx(baseline.hi, rel=1e-3)


@pytest.mark.parametrize("which", ["first", "second"])
def test_paired_ratio_rejects_a_stored_covariance_perturbed_by_1e_minus_3(which: str) -> None:
    t_a, t_b = _paired(12)
    if which == "first":
        t_a = _scaled_covariance(t_a, 1.0 + 1e-3)
    else:
        t_b = _scaled_covariance(t_b, 1.0 + 1e-3)

    with pytest.raises(ValueError, match="stored covariance"):
        ratio_interval(t_a, t_b, method="log_delta", dependence="paired", level=LEVEL)
