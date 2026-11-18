"""Contract test: the per-unit, clustered call shape.

Mirrors a consumer that records per-scenario, per-seed binary outcomes over a log-spaced severity
grid plus a zero-severity control, with the same scenario ids recurring at every level --
clustered, not independent. Only the public API surface is used here -- no internal robustmargin
module is imported.

A failing **live** test in this file means a breaking change to the public API such a consumer
depends on. **Do not edit a live test to make it pass.** Stop and ask; that failure is the signal
that a breaking change needs an explicit decision.

Clustered input needs an explicit ``dependence`` at every stage. ``dependence="independent"``
acknowledges that independence is assumed (``fit_dose_response``, which passes it to its
monotonicity check, then ``threshold()`` and ``ratio_interval()``); without it each raises
``ValueError``. ``dependence="clustered"`` on ``fit_dose_response`` and ``threshold()`` and
``dependence="paired"`` on ``ratio_interval()`` are the cluster-aware paths: a cluster-robust
covariance, a cluster bootstrap, and a ratio of two outcomes measured on the same clusters. The
tests that exercise the independence stages pass ``dependence="independent"`` to the earlier ones.
``Fit.cluster_ids`` is set from the input data regardless of which ``dependence`` the fit itself
was built with.

The per-level fixture has four clusters, which exceeds the two parameters a fixed-asymptote fit
estimates; a cluster-robust covariance needs more clusters than parameters. The bootstrap fixture
(``_scenario_rows``) has ten clusters, so a cluster resample rarely collapses to a single-outcome
refit and the cluster bootstrap stays under its failed-refit limit.
"""

from __future__ import annotations

import pytest

from robustmargin import Axis, Cell, Observations

_AXIS = Axis(name="noise", unit="ratio", scale="log")

# Nine log-spaced severities (0.01 doubling up to 1.0) plus the zero control, four scenario ids
# repeated at every level so the design is clustered rather than independent (clustered by
# scenario, paired outcomes). The pass counts per level are a fixed, hand-written schedule, not
# drawn from any random source, so there is no hidden randomness here to pin a seed for.
_SEVERITIES: tuple[float, ...] = (0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0)
_SCENARIOS: tuple[str, ...] = ("scenario_a", "scenario_b", "scenario_c", "scenario_d")
_PASSING_SCENARIOS_PER_LEVEL: tuple[int, ...] = (4, 4, 3, 3, 2, 2, 1, 1, 0)


def _clustered_rows() -> tuple[tuple[float, ...], tuple[int, ...], tuple[str, ...]]:
    """Build per-unit ``(severity, success, cluster)`` rows for the fixture above."""
    severity: list[float] = []
    success: list[int] = []
    cluster: list[str] = []
    for level, n_passing in zip(_SEVERITIES, _PASSING_SCENARIOS_PER_LEVEL, strict=True):
        for index, scenario in enumerate(_SCENARIOS):
            severity.append(level)
            success.append(1 if index < n_passing else 0)
            cluster.append(scenario)
    return tuple(severity), tuple(success), tuple(cluster)


def test_from_observations_with_cluster_ids_marks_observations_as_clustered() -> None:
    severity, success, cluster = _clustered_rows()

    obs = Observations.from_observations(
        _AXIS,
        severity=severity,
        success=success,
        outcome="consistent",
        direction="decreasing",
        cluster=cluster,
    )

    assert obs.is_clustered is True
    assert obs.cluster == cluster


def test_clustered_observations_cells_totals_match_the_per_level_schedule() -> None:
    severity, success, cluster = _clustered_rows()

    obs = Observations.from_observations(
        _AXIS,
        severity=severity,
        success=success,
        outcome="consistent",
        direction="decreasing",
        cluster=cluster,
    )

    expected = tuple(
        Cell(severity=level, successes=n_passing, trials=len(_SCENARIOS))
        for level, n_passing in zip(_SEVERITIES, _PASSING_SCENARIOS_PER_LEVEL, strict=True)
    )
    assert obs.cells() == expected


def test_fit_dose_response_on_clustered_input_without_dependence_raises() -> None:
    """``fit_dose_response`` threads its own ``dependence`` argument through to the monotonicity
    check it computes before choosing a fitting path, so a clustered ``Observations`` with no
    ``dependence=`` raises at *fit* time, not only at the later ``threshold()``/
    ``ratio_interval()`` calls this file also covers."""
    from robustmargin import fit_dose_response

    severity, success, cluster = _clustered_rows()
    obs = Observations.from_observations(
        _AXIS,
        severity=severity,
        success=success,
        outcome="consistent",
        direction="decreasing",
        cluster=cluster,
    )

    with pytest.raises(ValueError):
        fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)


def test_threshold_on_clustered_input_without_independent_dependence_raises() -> None:
    """A method that assumes independence must refuse clustered input outright."""
    from robustmargin import Definition, fit_dose_response, threshold

    severity, success, cluster = _clustered_rows()
    obs = Observations.from_observations(
        _AXIS,
        severity=severity,
        success=success,
        outcome="consistent",
        direction="decreasing",
        cluster=cluster,
    )
    # fit_dose_response threads dependence through to its own clustered-input guard (the
    # monotonicity check), so clustered data needs it here too -- threshold() below still gets no
    # dependence= of its own, which is what this test checks.
    fit = fit_dose_response(
        obs, model="binomial", link="probit", upper=1.0, lower=0.0, dependence="independent"
    )

    with pytest.raises(ValueError):
        threshold(fit, definition=Definition.absolute(0.5), interval_method="profile", level=0.95)


def test_paired_ratio_on_independent_thresholds_from_one_fit_raises() -> None:
    """Two thresholds solved from one fit are the same data twice, so ``dependence="paired"``
    cannot describe them: a ratio of one fit with itself is not a comparison of two outcomes, and
    these thresholds are built ``"independent"``, not clustered. ``ratio_interval`` must raise
    rather than silently treating the pair as anything else.

    ``dependence="independent"`` is passed to both ``threshold()`` calls (module docstring): the
    clustered-input guard would otherwise raise here, before ever reaching the
    ``dependence="paired"`` call this test exists to check.
    """
    from robustmargin import Definition, fit_dose_response, ratio_interval, threshold

    severity, success, cluster = _clustered_rows()
    obs = Observations.from_observations(
        _AXIS,
        severity=severity,
        success=success,
        outcome="consistent",
        direction="decreasing",
        cluster=cluster,
    )
    # fit_dose_response threads dependence through to its own clustered-input guard (the
    # monotonicity check).
    fit = fit_dose_response(
        obs, model="binomial", link="probit", upper=1.0, lower=0.0, dependence="independent"
    )
    t_a = threshold(
        fit,
        definition=Definition.absolute(0.5),
        interval_method="profile",
        level=0.95,
        dependence="independent",
    )
    t_b = threshold(
        fit,
        definition=Definition.baseline_fraction(0.5),
        interval_method="profile",
        level=0.95,
        dependence="independent",
    )

    with pytest.raises(ValueError):
        ratio_interval(t_a, t_b, method="fieller", dependence="paired")


# The bootstrap fixture: ten scenario ids over the same levels. With only four clusters a cluster
# resample often drops every passing (or failing) scenario at a level and the refit fails.
_BOOTSTRAP_SCENARIOS: tuple[str, ...] = tuple(f"scenario_{index:02d}" for index in range(10))
_BOOTSTRAP_PASSING_SCENARIOS_PER_LEVEL: tuple[int, ...] = (10, 10, 8, 7, 6, 5, 3, 2, 0)


def _scenario_rows(
    passing: tuple[int, ...],
) -> tuple[tuple[float, ...], tuple[int, ...], tuple[str, ...]]:
    """Per-unit rows over the same scenarios and levels, with a different pass schedule."""
    severity: list[float] = []
    success: list[int] = []
    cluster: list[str] = []
    for level, n_passing in zip(_SEVERITIES, passing, strict=True):
        for index, scenario in enumerate(_BOOTSTRAP_SCENARIOS):
            severity.append(level)
            success.append(1 if index < n_passing else 0)
            cluster.append(scenario)
    return tuple(severity), tuple(success), tuple(cluster)


# A second outcome on the same ten scenarios: it fails earlier along the severity grid.
_SECOND_OUTCOME_PASSING: tuple[int, ...] = (10, 9, 8, 7, 5, 4, 2, 1, 0)


def _observations(passing: tuple[int, ...], outcome: str) -> Observations:
    severity, success, cluster = _scenario_rows(passing)
    return Observations.from_observations(
        _AXIS,
        severity=severity,
        success=success,
        outcome=outcome,
        direction="decreasing",
        cluster=cluster,
    )


def test_clustered_fit_and_cluster_bootstrap_threshold_return_clustered_results() -> None:
    from robustmargin import Definition, fit_dose_response, threshold

    fit = fit_dose_response(
        _observations(_BOOTSTRAP_PASSING_SCENARIOS_PER_LEVEL, "consistent"),
        model="binomial",
        link="probit",
        upper=1.0,
        lower=0.0,
        dependence="clustered",
    )

    assert fit.dependence == "clustered"
    assert fit.cluster_ids == _BOOTSTRAP_SCENARIOS

    result = threshold(
        fit,
        definition=Definition.absolute(0.5),
        interval_method="cluster_bootstrap",
        level=0.95,
        dependence="clustered",
        n_resamples=49,
        seed=20261117,
    )

    assert result.dependence == "clustered"
    assert result.interval_method == "cluster_bootstrap"
    assert result.level == 0.95
    assert result.bootstrap is not None
    assert result.bootstrap.seed == 20261117


def test_paired_ratio_between_two_clustered_fits_on_shared_clusters_returns_a_paired_ratio() -> (
    None
):
    """Two outcomes measured on the same scenarios, one definition: ``dependence="paired"`` with
    the cluster bootstrap returns a ``Ratio`` carrying the bootstrap summary. The assertion is on
    shape only (the type, ``dependence``, ``method`` and ``bootstrap``), not on a status or a
    value; the seed is pinned so the call is reproducible."""
    from robustmargin import Definition, Ratio, fit_dose_response, ratio_interval, threshold

    thresholds = []
    for passing, outcome in (
        (_BOOTSTRAP_PASSING_SCENARIOS_PER_LEVEL, "consistent"),
        (_SECOND_OUTCOME_PASSING, "complete"),
    ):
        fit = fit_dose_response(
            _observations(passing, outcome),
            model="binomial",
            link="probit",
            upper=1.0,
            lower=0.0,
            dependence="clustered",
        )
        thresholds.append(
            threshold(
                fit,
                definition=Definition.absolute(0.5),
                interval_method="cluster_bootstrap",
                level=0.95,
                dependence="clustered",
                n_resamples=49,
                seed=20261117,
            )
        )
    t_a, t_b = thresholds

    result = ratio_interval(
        t_a,
        t_b,
        method="cluster_bootstrap",
        dependence="paired",
        n_resamples=49,
        seed=20261117,
    )

    assert isinstance(result, Ratio)
    assert result.dependence == "paired"
    assert result.method == "cluster_bootstrap"
    assert result.bootstrap is not None
    assert result.bootstrap.seed == 20261117
