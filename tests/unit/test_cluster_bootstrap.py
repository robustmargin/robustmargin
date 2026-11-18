"""Mechanics of ``threshold(..., interval_method="cluster_bootstrap", dependence="clustered")``.

Resampling clusters with replacement, summing their per-(severity, cluster) counts into cells,
refitting, and reading each replicate's threshold.

How the tests reach the mechanism without private hooks: the refit entry ``_fit_counts`` (the
function the bootstrap refits through) is wrapped, after the fits under test have been built, so
that it returns the real refit with ``mu``/``upper``/``lower`` replaced by a scripted value, in
replicate order. Replicate ``i``'s threshold for ``absolute(0.5)`` with fixed ``u = 1, l = 0`` is
then exactly the scripted ``mu``, whatever the seed drew, and the percentiles, shapes, counts and
warnings can be compared with a hand computation. The wrapper is applied to the modules that
import the function (``models``, ``intervals``, ``ratio``) with ``raising=False``; it needs the
refits to be sequential and to read ``params`` of the returned fit.

*Provisional constants.* The cross-check's ``c`` and ``d0`` and the few-clusters ``K`` are not yet
calibrated. The cross-check tests use scripted replicate distributions that are far inside or far
outside any plausible ``(c, d0)`` (a scale ratio of 1, or of 0.01 and 100; a location shift of 0 or
5 standard errors), and the few-clusters tests use ``G = 4`` and ``G = 60``, below the smallest and
above the largest ``K`` the pre-registered rule can return (5 to 40).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from _cluster_support import (
    LOW,
    Row,
    counts_obs,
    covariance_array,
    finite,
    install_scripts,
    simulate_rows,
)
from scipy.stats import norm

from robustmargin import (
    Definition,
    Fit,
    Observations,
    Status,
    Threshold,
    fit_dose_response,
    threshold,
)
from robustmargin import intervals as intervals_module
from robustmargin import models as models_module
from robustmargin import ratio as ratio_module

_SEED = 20261501
_ABS_HALF = Definition.absolute(0.5)


def clustered_fit(rows: list[Row], *, direction: str = "decreasing") -> Fit:
    return fit_dose_response(
        counts_obs(rows, direction=direction),
        model="binomial",
        link="probit",
        upper=1.0,
        lower=0.0,
        dependence="clustered",
    )


def bootstrap_threshold(
    fit: Fit,
    *,
    n_resamples: int = 99,
    seed: int = _SEED,
    level: float = 0.95,
    definition: Definition = _ABS_HALF,
) -> Threshold:
    return threshold(
        fit,
        definition=definition,
        interval_method="cluster_bootstrap",
        level=level,
        dependence="clustered",
        n_resamples=n_resamples,
        seed=seed,
    )


def _rows(clusters: int = 12, seed: int = _SEED, **kwargs: Any) -> list[Row]:
    return simulate_rows(seed, clusters=clusters, **kwargs)


# -- determinism ----------------------------------------------------------------------------


def test_cluster_bootstrap_is_deterministic_under_a_seed_and_varies_with_it() -> None:
    fit = clustered_fit(_rows())

    first = bootstrap_threshold(fit, seed=11)
    again = bootstrap_threshold(fit, seed=11)
    other = bootstrap_threshold(fit, seed=12)

    assert first.status is Status.OK
    assert (first.lo, first.hi, first.bootstrap) == (again.lo, again.hi, again.bootstrap)
    assert (first.lo, first.hi) != (other.lo, other.hi)


def test_cluster_bootstrap_records_method_dependence_and_seed() -> None:
    fit = clustered_fit(_rows())

    result = bootstrap_threshold(fit, n_resamples=60, seed=77)

    assert result.interval_method == "cluster_bootstrap"
    assert result.dependence == "clustered"
    assert result.bootstrap is not None
    assert (result.bootstrap.n_resamples, result.bootstrap.seed) == (60, 77)
    assert result.censoring.name in {"NONE", "OPEN_UPPER", "OPEN_LOWER"}


def test_cluster_bootstrap_keeps_the_point_estimate_of_the_fit() -> None:
    fit = clustered_fit(_rows())
    assert fit.params is not None

    result = bootstrap_threshold(fit)

    assert result.value == pytest.approx(math.exp(fit.params["mu"].value), rel=1e-9)


# -- what a replicate is ---------------------------------------------------------------------


def test_cluster_bootstrap_replicates_sum_whole_clusters_drawn_with_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cluster ``c`` has ``7**c`` trials at every level, so a level's pooled trials spell out the
    multiplicity of each cluster in base 7 (G = 6 draws, so no digit reaches 7). Every level of a
    replicate must show the same multiplicities, the multiplicities must add to G, the successes
    must equal the multiplicity-weighted cluster successes exactly (never more than the trials),
    and the pooled trials must vary between replicates (a fixed-N design would not vary)."""
    g = 6
    sizes = [7**c for c in range(g)]
    rows = _rows(clusters=g, trials=sizes, sigma=0.5)
    fit = clustered_fit(rows)
    assert fit.status is Status.OK
    per_cluster: dict[tuple[float, str], int] = {(r[0], r[1]): r[2] for r in rows}
    ids = [f"c{c:02d}" for c in range(g)]
    levels = sorted({r[0] for r in rows})

    seen: list[Observations] = []
    real = models_module._fit_counts

    def spy(obs: Observations, **kwargs: Any) -> Fit:
        seen.append(obs)
        return real(obs, **kwargs)

    for module in (models_module, intervals_module, ratio_module):
        monkeypatch.setattr(module, "_fit_counts", spy, raising=False)
    bootstrap_threshold(fit, n_resamples=40)

    assert len(seen) >= 40
    totals = set()
    for obs in seen[-40:]:
        cells = {c.severity: c for c in obs.cells()}
        assert set(cells) <= set(levels)
        multiplicities = None
        for level, cell in cells.items():
            assert cell.successes <= cell.trials
            digits = [(cell.trials // 7**c) % 7 for c in range(g)]
            assert sum(d * 7**c for c, d in enumerate(digits)) == cell.trials
            if multiplicities is None:
                multiplicities = digits
            assert digits == multiplicities
            assert cell.successes == sum(
                m * per_cluster[(level, name)] for m, name in zip(digits, ids, strict=True)
            )
        assert multiplicities is not None and sum(multiplicities) == g
        totals.add(sum(c.trials for c in cells.values()))
    assert len(totals) > 1


def test_cluster_bootstrap_counts_a_replicate_that_loses_every_level_as_failed_not_raised() -> None:
    """Only three of six clusters observe any level besides the zero-severity control, so a
    replicate that draws none of them has no level to fit. ``_fit_counts`` raises ``ValueError``
    for that design; the bootstrap counts it as a failed replicate and carries on."""
    rows: list[Row] = []
    for name in ("c00", "c01", "c02", "c03", "c04", "c05"):
        rows.append((0.0, name, 10, 10))
    informative = _rows(clusters=3, seed=_SEED + 1, levels=(2.0, 4.0, 8.0, 16.0))
    rows += [r for r in informative]
    fit = clustered_fit(rows)
    assert fit.status is Status.OK

    result = bootstrap_threshold(fit, n_resamples=300, level=0.90)

    assert result.bootstrap is not None
    assert result.bootstrap.failed >= 1
    assert any(w.startswith("BOOTSTRAP_FAILURES:") for w in result.warnings)


def test_cluster_bootstrap_does_not_swallow_an_unrelated_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = clustered_fit(_rows())

    def broken(obs: Observations, **kwargs: Any) -> Fit:
        raise ValueError("something unrelated to the design")

    for module in (models_module, intervals_module, ratio_module):
        monkeypatch.setattr(module, "_fit_counts", broken, raising=False)

    with pytest.raises(ValueError, match="unrelated"):
        bootstrap_threshold(fit, n_resamples=20)


# -- percentiles, open sides and the failure rules, on scripted replicates --------------------


def _weibull(values: list[float], p: float) -> float:
    return float(np.quantile(np.array(sorted(values)), p, method="weibull"))


def test_cluster_bootstrap_interval_is_the_weibull_percentile_of_the_replicate_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = clustered_fit(_rows())
    thetas = [1.0 + 0.01 * i for i in range(199)]
    install_scripts(monkeypatch, {"success": lambda i: finite(thetas[i])})

    result = bootstrap_threshold(fit, n_resamples=199, seed=3)

    assert result.lo == pytest.approx(math.exp(_weibull(thetas, 0.025)), rel=1e-12)
    assert result.hi == pytest.approx(math.exp(_weibull(thetas, 0.975)), rel=1e-12)
    assert result.bootstrap is not None and result.bootstrap.failed == 0


def test_cluster_bootstrap_unreachable_replicates_open_the_side_they_fall_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = clustered_fit(_rows())
    # 10% of replicates have no crossing on the high side: that side cannot be bounded.
    install_scripts(
        monkeypatch, {"success": lambda i: LOW if i % 10 == 0 else finite(1.0 + 0.001 * i)}
    )

    result = bootstrap_threshold(fit, n_resamples=200, seed=3)

    assert result.censoring.name == "OPEN_UPPER"
    assert result.hi is None and result.lo is not None
    assert result.bootstrap is not None and result.bootstrap.unreachable_high == 20


# -- the sandwich-delta cross-check, on scripted replicate distributions -----------------------


def _cross_check_warnings(result: Threshold) -> list[str]:
    return [w for w in result.warnings if w.startswith("CLUSTER_CROSS_CHECK:")]


def _scripted_normal(
    monkeypatch: pytest.MonkeyPatch, fit: Fit, *, scale: float, shift: float, n: int
) -> Threshold:
    assert fit.params is not None
    theta_hat = fit.params["mu"].value
    se = math.sqrt(covariance_array(fit)[0, 0])
    z = norm.ppf((np.arange(n) + 0.5) / n)
    thetas = list(theta_hat + shift * se + scale * se * z)
    install_scripts(monkeypatch, {"success": lambda i: finite(thetas[i])})
    return bootstrap_threshold(fit, n_resamples=n, seed=5)


def test_cluster_cross_check_is_silent_when_scale_and_location_agree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = clustered_fit(_rows())

    result = _scripted_normal(monkeypatch, fit, scale=1.0, shift=0.0, n=399)

    assert _cross_check_warnings(result) == []


@pytest.mark.parametrize("scale", [0.01, 100.0])
def test_cluster_cross_check_flags_a_bootstrap_spread_far_from_the_sandwich_standard_error(
    monkeypatch: pytest.MonkeyPatch, scale: float
) -> None:
    fit = clustered_fit(_rows())

    result = _scripted_normal(monkeypatch, fit, scale=scale, shift=0.0, n=399)

    assert len(_cross_check_warnings(result)) == 1
    assert result.lo is not None and result.hi is not None  # the numbers are never changed


def test_cluster_cross_check_flags_a_bootstrap_centre_far_from_the_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = clustered_fit(_rows())

    result = _scripted_normal(monkeypatch, fit, scale=1.0, shift=5.0, n=399)

    assert len(_cross_check_warnings(result)) == 1


def test_cluster_cross_check_is_skipped_when_the_bootstrap_interval_is_not_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = clustered_fit(_rows())
    install_scripts(
        monkeypatch,
        {"success": lambda i: LOW if i % 10 == 0 else finite(1.0 + 100.0 * ((i % 7) - 3) / 7.0)},
    )

    result = bootstrap_threshold(fit, n_resamples=200, seed=5)

    assert result.censoring.name == "OPEN_UPPER"
    assert _cross_check_warnings(result) == []


# -- few clusters ---------------------------------------------------------------------------


def _few(texts: tuple[str, ...]) -> list[str]:
    return [t for t in texts if t.startswith("FEW_CLUSTERS:")]


def test_few_clusters_is_disclosed_on_a_clustered_fit_and_threshold_below_every_candidate_k() -> (
    None
):
    fit = clustered_fit(_rows(clusters=4, trials=40))
    assert fit.status is Status.OK

    result = threshold(
        fit, definition=_ABS_HALF, interval_method="delta", level=0.95, dependence="clustered"
    )

    assert len(_few(fit.warnings)) == 1
    assert len(_few(result.warnings)) == 1
    assert "4" in _few(result.warnings)[0]


def test_few_clusters_is_not_disclosed_with_more_clusters_than_every_candidate_k() -> None:
    fit = clustered_fit(_rows(clusters=60, trials=8))

    result = threshold(
        fit, definition=_ABS_HALF, interval_method="delta", level=0.95, dependence="clustered"
    )

    assert _few(fit.warnings) == []
    assert _few(result.warnings) == []


def test_few_clusters_is_not_disclosed_on_an_independent_fit_of_the_same_data() -> None:
    fit = fit_dose_response(
        counts_obs(_rows(clusters=4, trials=40)),
        model="binomial",
        link="probit",
        upper=1.0,
        lower=0.0,
        dependence="independent",
    )

    assert _few(fit.warnings) == []


# -- failed fits and the censoring rules (no exact claims) ------------------------------------


def test_clustered_threshold_on_a_separated_fit_keeps_the_fit_status_with_no_numbers() -> None:
    rows = [
        (sev, f"c{c}", 10 if sev <= 2.0 else 0, 10)
        for c in range(6)
        for sev in (0.0, 1.0, 2.0, 4.0, 8.0)
    ]
    fit = clustered_fit(rows)
    assert fit.status is Status.SEPARATION

    result = threshold(
        fit, definition=_ABS_HALF, interval_method="delta", level=0.95, dependence="clustered"
    )

    assert result.status is Status.SEPARATION
    assert (result.value, result.lo, result.hi) == (None, None, None)
    assert result.interval_method == "exact_bound"
    assert result.grid is None
    assert result.censoring.name == "NONE"


def test_clustered_threshold_never_reports_an_exact_censoring_label() -> None:
    """A target above every tested level would be a cell-level exact ``RIGHT`` under independence;
    under clustering it is not an exact claim."""
    rows = _rows(clusters=12)
    fit = clustered_fit(rows)

    result = threshold(
        fit,
        definition=Definition.absolute(0.9995),
        interval_method="delta",
        level=0.95,
        dependence="clustered",
    )

    assert result.censoring.name not in {"RIGHT", "LEFT"}
    assert result.status.name != "FAILS_AT_BASELINE"
