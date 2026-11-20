"""``plan_threshold_width`` / ``plan_ratio_width`` and ``WidthPlan`` semantics.

Simulated runs are kept small (a handful of replicates, fixed asymptotes ``upper=1``,
``lower=0``, ``interval_method="delta"``) so the file stays fast. Semantics tests build
``WidthPlan`` objects directly and need no simulation.

Assumed field types, beyond the names the contract file pins: ``axis`` is the ``Axis`` the pilot
fit was made on, ``definition`` the ``Definition`` passed in, ``severity`` a tuple of floats.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Literal

import numpy as np
import pytest
from scipy.stats import norm

from robustmargin import (
    Axis,
    Censoring,
    ContinuousObservations,
    Definition,
    Fit,
    IntervalShape,
    Observations,
    Status,
    fit_dose_response,
)
from robustmargin.design import WidthPlan, plan_ratio_width, plan_threshold_width

_LOG = Axis(name="severity", unit="unit", scale="log")
_LINEAR = Axis(name="severity", unit="unit", scale="linear")
_SEED = 20261119
_LEVELS = (0.25, 0.5, 1.0, 2.0, 4.0)
_ALLOWED_REASONS = {
    "SEPARATION",
    "NOT_CONVERGED",
    "UNREACHABLE",
    "FAILS_AT_BASELINE",
    "CONTROL_INCOMPATIBLE",
    "RIGHT",
    "LEFT",
    "OPEN_UPPER",
    "OPEN_LOWER",
    "UNBOUNDED",
    "EXCLUSIVE",
    "HALF_OPEN",
    "BOTH_SIDES_OPEN",
    "LOWER_AT_ZERO",
}


def _pilot(
    mu: float = 0.0,
    s: float = 0.5,
    *,
    axis: Axis = _LOG,
    link: Literal["probit", "logit", "cloglog"] = "probit",
    upper: float | Literal["estimate"] = 1.0,
    lower: float | Literal["estimate"] = 0.0,
    direction: str = "decreasing",
) -> Fit:
    """A pilot fit on counts rounded from the exact curve (no randomness)."""
    levels = (
        np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8])
        if axis.scale == "log"
        else (np.linspace(1.0, 8.0, 8))
    )
    z = (np.log(levels) if axis.scale == "log" else levels) - mu
    p = 1.0 - norm.cdf(z / s)
    if direction == "increasing":
        p = 1.0 - p
    trials = 200
    obs = Observations.from_counts(
        axis,
        severity=list(levels),
        successes=[int(round(trials * v)) for v in p],
        trials=[trials] * len(levels),
        outcome="outcome",
        direction=direction,
    )
    fit = fit_dose_response(obs, model="binomial", link=link, upper=upper, lower=lower)
    return fit


def _plan(curve: Fit | None = None, **overrides: Any) -> WidthPlan:
    kwargs: dict[str, Any] = {
        "severity": _LEVELS,
        "trials_per_level": 50,
        "definition": Definition.absolute(0.5),
        "interval_method": "delta",
        "level": 0.95,
        "n_replicates": 6,
        "seed": _SEED,
    }
    kwargs.update(overrides)
    return plan_threshold_width(curve if curve is not None else _pilot(), **kwargs)


def _ratio_plan(**overrides: Any) -> WidthPlan:
    kwargs: dict[str, Any] = {
        "severity": _LEVELS,
        "trials_per_level": 50,
        "definition": Definition.absolute(0.5),
        "interval_method": "delta",
        "ratio_method": "log_delta",
        "dependence": "independent",
        "level": 0.95,
        "n_replicates": 6,
        "seed": _SEED,
    }
    kwargs.update(overrides)
    return plan_ratio_width(_pilot(0.0), _pilot(0.4), **kwargs)


def _direct(**overrides: Any) -> WidthPlan:
    """A WidthPlan built without simulation."""
    kwargs: dict[str, Any] = {
        "kind": "threshold",
        "axis": _LOG,
        "severity": _LEVELS,
        "trials_per_level": 50,
        "n_clusters": None,
        "icc": 0.0,
        "definition": Definition.absolute(0.5),
        "interval_method": "delta",
        "ratio_method": None,
        "dependence": "independent",
        "level": 0.95,
        "n_replicates": 4,
        "seed": 1,
        "n_resamples": None,
        "width_scale": "log_ratio",
        "widths": (1.0, 2.0, 3.0, None),
        "outcomes": ("OK", "OK", "OK", "UNBOUNDED"),
    }
    kwargs.update(overrides)
    return WidthPlan(**kwargs)


# -- argument validation -------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_replicates": 0},
        {"workers": 0},
        {"trials_per_level": 0},
        {"severity": (1.0,)},
        {"severity": (0.0, 1.0)},
        {"severity": (1.0, 1.0, 2.0)},
        {"severity": (2.0, 1.0, 0.5)},
        {"severity": (-1.0, 1.0, 2.0)},
        {"n_clusters": 1},
        {"n_clusters": 7},
        {"n_clusters": 5, "icc": 1.0},
        {"n_clusters": 5, "icc": -0.1},
        {"icc": 0.3},
    ],
    ids=[
        "no_replicates",
        "no_workers",
        "no_trials",
        "one_level",
        "one_nonzero_level",
        "repeated_level",
        "decreasing_levels",
        "negative_level",
        "one_cluster",
        "clusters_do_not_divide_trials",
        "icc_one",
        "icc_negative",
        "icc_without_clusters",
    ],
)
def test_invalid_arguments_raise_value_error(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _plan(**overrides)


@pytest.mark.parametrize("name", ["n_replicates", "trials_per_level", "workers"])
def test_bool_where_an_int_is_needed_raises_type_error(name: str) -> None:
    overrides: dict[str, Any] = {name: True}
    with pytest.raises(TypeError):
        _plan(**overrides)


def test_a_fit_that_is_not_ok_is_rejected() -> None:
    obs = Observations.from_counts(
        _LOG,
        severity=[0.5, 1.0, 2.0, 4.0],
        successes=[50, 50, 0, 0],
        trials=[50] * 4,
        outcome="outcome",
        direction="decreasing",
    )
    separated = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
    assert separated.status is not Status.OK
    with pytest.raises(ValueError):
        _plan(separated)


def test_a_gaussian_fit_is_rejected() -> None:
    severity = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0] * 3
    rng = np.random.default_rng(3)
    value = [10.0 / (1.0 + x * x) + rng.normal(0.0, 0.2) for x in severity]
    obs = ContinuousObservations.from_observations(
        _LOG, severity=severity, value=value, outcome="response", direction="decreasing"
    )
    fit = fit_dose_response(obs, model="gaussian", link="logit", upper="estimate", lower="estimate")
    with pytest.raises(ValueError):
        _plan(fit)


def test_an_isotonic_fit_is_rejected() -> None:
    obs = Observations.from_counts(
        _LOG,
        severity=[0.5, 1.0, 2.0, 4.0],
        successes=[48, 40, 20, 4],
        trials=[50] * 4,
        outcome="outcome",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="isotonic")
    with pytest.raises(ValueError):
        _plan(fit)


def test_ratio_curves_on_different_axis_scales_are_rejected() -> None:
    with pytest.raises(ValueError):
        plan_ratio_width(
            _pilot(0.0),
            _pilot(5.0, 2.0, axis=_LINEAR),
            severity=_LEVELS,
            trials_per_level=50,
            definition=Definition.absolute(0.5),
            interval_method="delta",
            ratio_method="fieller",
            dependence="independent",
            level=0.95,
            n_replicates=2,
            seed=1,
        )


def test_ratio_curves_with_different_links_are_rejected() -> None:
    with pytest.raises(ValueError):
        plan_ratio_width(
            _pilot(0.0),
            _pilot(0.0, link="logit"),
            severity=_LEVELS,
            trials_per_level=50,
            definition=Definition.absolute(0.5),
            interval_method="delta",
            ratio_method="log_delta",
            dependence="independent",
            level=0.95,
            n_replicates=2,
            seed=1,
        )


def test_ratio_curves_with_different_asymptote_specs_are_rejected() -> None:
    with pytest.raises(ValueError):
        plan_ratio_width(
            _pilot(0.0),
            _pilot(0.0, upper=0.95),
            severity=_LEVELS,
            trials_per_level=50,
            definition=Definition.absolute(0.5),
            interval_method="delta",
            ratio_method="log_delta",
            dependence="independent",
            level=0.95,
            n_replicates=2,
            seed=1,
        )


def test_paired_dependence_without_clusters_is_rejected() -> None:
    with pytest.raises(ValueError):
        _ratio_plan(dependence="paired")


def test_ratio_dependence_other_than_independent_or_paired_is_rejected() -> None:
    with pytest.raises(ValueError):
        _ratio_plan(dependence="clustered")


# -- what the planner records --------------------------------------------------------------


def test_threshold_plan_echoes_the_design_and_marks_it_independent() -> None:
    provenance = {"note": "pilot sweep"}
    result = _plan(provenance=provenance)
    assert result.kind == "threshold"
    assert result.axis == _LOG
    assert tuple(result.severity) == _LEVELS
    assert result.trials_per_level == 50
    assert result.n_clusters is None
    assert result.icc == 0.0
    assert result.definition == Definition.absolute(0.5)
    assert result.interval_method == "delta"
    assert result.ratio_method is None
    assert result.dependence == "independent"
    assert result.level == 0.95
    assert result.n_replicates == 6
    assert result.seed == _SEED
    assert result.width_scale == "log_ratio"
    assert result.plan_schema_version == "1"
    assert result.provenance == provenance


def test_threshold_plan_with_clusters_is_marked_clustered() -> None:
    result = _plan(n_clusters=5, icc=0.2, n_replicates=3)
    assert result.dependence == "clustered"
    assert result.n_clusters == 5
    assert result.icc == 0.2


def test_linear_axis_threshold_plan_measures_width_as_a_difference() -> None:
    curve = _pilot(4.0, 1.5, axis=_LINEAR)
    result = _plan(curve, severity=(1.0, 2.5, 4.0, 5.5, 7.0), n_replicates=4)
    assert result.width_scale == "difference"


def test_ratio_plan_records_methods_and_a_log_ratio_scale() -> None:
    result = _ratio_plan(n_replicates=4)
    assert result.kind == "ratio"
    assert result.ratio_method == "log_delta"
    assert result.interval_method == "delta"
    assert result.dependence == "independent"
    assert result.width_scale == "log_ratio"


def test_paired_ratio_plan_runs_with_clusters() -> None:
    result = _ratio_plan(dependence="paired", n_clusters=10, icc=0.3, n_replicates=3)
    assert result.dependence == "paired"
    assert result.n_clusters == 10
    assert len(result.widths) == 3


def test_increasing_pilot_is_respected_and_gives_finite_widths() -> None:
    curve = _pilot(direction="increasing")
    result = _plan(curve, definition=Definition.absolute(0.5), n_replicates=6)
    assert result.n_replicates == 6
    assert any(w is not None for w in result.widths)


def test_provenance_defaults_to_an_empty_dict() -> None:
    assert _plan(n_replicates=2).provenance == {}


# -- WidthPlan invariants on simulated output ----------------------------------------------


def test_simulated_plan_satisfies_the_width_outcome_pairing() -> None:
    result = _plan(trials_per_level=200, severity=(0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0))
    assert len(result.widths) == result.n_replicates == len(result.outcomes)
    for width, outcome in zip(result.widths, result.outcomes, strict=True):
        assert (outcome == "OK") == (width is not None)
        if width is not None:
            assert math.isfinite(width)
            assert width >= 0.0
        else:
            assert outcome in _ALLOWED_REASONS
    assert any(w is not None for w in result.widths)


def test_outcome_counts_total_the_replicates() -> None:
    result = _plan(trials_per_level=5)
    counts = result.outcome_counts()
    assert sum(counts.values()) == result.n_replicates
    assert counts.get("OK", 0) == sum(w is not None for w in result.widths)


# -- determinism ---------------------------------------------------------------------------


def test_workers_one_and_two_give_identical_plans() -> None:
    serial = _plan(workers=1)
    parallel = _plan(workers=2)
    assert serial == parallel


def test_same_seed_gives_the_same_plan_and_a_different_seed_a_different_one() -> None:
    first = _plan()
    assert first == _plan()
    assert first.widths != _plan(seed=_SEED + 1).widths


def test_clustered_plan_is_identical_across_worker_counts() -> None:
    assert _plan(n_clusters=5, icc=0.4, n_replicates=4, workers=1) == _plan(
        n_clusters=5, icc=0.4, n_replicates=4, workers=2
    )


def test_first_replicates_do_not_depend_on_how_many_replicates_are_requested() -> None:
    # Per-replicate streams are spawned from the seed, so replicate i is the same run whether
    # 4 or 6 replicates were asked for.
    short = _plan(n_replicates=4)
    long = _plan(n_replicates=6)
    assert long.widths[:4] == short.widths
    assert long.outcomes[:4] == short.outcomes


# -- forced censoring and failure: nothing is dropped ----------------------------------------


def test_target_outside_the_tested_range_gives_no_finite_width_and_a_reason_each() -> None:
    # Every tested level lies far above the crossing at severity 1, so every replicate is
    # (almost) all failure: no bounded interval can exist.
    result = _plan(severity=(8.0, 12.0, 16.0, 24.0), trials_per_level=50, n_replicates=8)
    assert all(w is None for w in result.widths)
    assert len(result.outcomes) == 8
    assert all(o != "OK" for o in result.outcomes)
    assert set(result.outcomes) <= _ALLOWED_REASONS
    assert sum(result.outcome_counts().values()) == 8
    assert "OK" not in result.outcome_counts()


def test_target_below_the_tested_range_gives_no_finite_width() -> None:
    result = _plan(severity=(0.02, 0.03, 0.04, 0.05), trials_per_level=50, n_replicates=8)
    assert all(w is None for w in result.widths)
    assert set(result.outcomes) <= _ALLOWED_REASONS


def test_a_forced_failure_design_keeps_every_replicate_in_the_tally() -> None:
    result = _plan(severity=(8.0, 12.0, 16.0), trials_per_level=1, n_replicates=10)
    assert len(result.widths) == 10
    assert sum(result.outcome_counts().values()) == 10
    assert result.quantile(0.5) == math.inf
    assert result.share_at_most(1e9) == (0.0, 0.0)


# -- quantile and share_at_most semantics --------------------------------------------------


def test_quantile_of_all_finite_widths() -> None:
    result = _direct(
        widths=(1.0, 2.0, 3.0, 4.0, 5.0),
        outcomes=("OK",) * 5,
        n_replicates=5,
    )
    assert result.quantile(0.0) == 1.0
    assert result.quantile(0.5) == 3.0
    assert result.quantile(1.0) == 5.0


def test_quantile_treats_none_as_infinite() -> None:
    result = _direct(
        widths=(1.0, 2.0, 3.0, None, None),
        outcomes=("OK", "OK", "OK", "SEPARATION", "UNBOUNDED"),
        n_replicates=5,
    )
    assert result.quantile(0.0) == 1.0
    assert result.quantile(0.5) == 3.0
    assert result.quantile(0.9) == math.inf
    assert result.quantile(1.0) == math.inf


@pytest.mark.parametrize("q", [-0.1, 1.5])
def test_quantile_outside_zero_one_is_rejected(q: float) -> None:
    with pytest.raises(ValueError):
        _direct().quantile(q)


def test_share_at_most_counts_only_finite_widths_and_reports_a_binomial_se() -> None:
    result = _direct()
    share, se = result.share_at_most(2.0)
    assert share == 0.5
    assert se == pytest.approx(math.sqrt(0.5 * 0.5 / 4))


def test_share_at_most_is_inclusive_of_the_threshold_width() -> None:
    assert _direct().share_at_most(3.0)[0] == 0.75


def test_share_at_most_never_counts_a_none_width_however_loose_the_bound() -> None:
    share, se = _direct().share_at_most(1e12)
    assert share == 0.75
    assert se == pytest.approx(math.sqrt(0.75 * 0.25 / 4))


def test_share_at_most_below_every_width_is_zero_with_zero_se() -> None:
    assert _direct().share_at_most(0.5) == (0.0, 0.0)


def test_share_at_most_is_non_decreasing_in_the_width() -> None:
    result = _direct()
    shares = [result.share_at_most(w)[0] for w in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 9.0)]
    assert shares == sorted(shares)


def test_outcome_counts_tallies_each_label() -> None:
    result = _direct(
        widths=(1.0, None, 2.0, None, None),
        outcomes=("OK", "SEPARATION", "OK", "UNBOUNDED", "SEPARATION"),
        n_replicates=5,
    )
    assert result.outcome_counts() == {"OK": 2, "SEPARATION": 2, "UNBOUNDED": 1}


# -- WidthPlan validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"outcomes": ("OK", "OK", "OK")},
        {"n_replicates": 5},
        {"outcomes": ("OK", "OK", "OK", "OK")},
        {"widths": (1.0, 2.0, 3.0, 4.0), "outcomes": ("OK", "OK", "OK", "SEPARATION")},
        {"widths": (1.0, 2.0, 3.0, -0.5), "outcomes": ("OK",) * 4},
        {"widths": (1.0, 2.0, 3.0, math.inf), "outcomes": ("OK",) * 4},
        {"widths": (1.0, 2.0, 3.0, math.nan), "outcomes": ("OK",) * 4},
        {"width_scale": "difference"},
        {"width_scale": "nonsense"},
        {"kind": "ratio", "width_scale": "difference", "ratio_method": "log_delta"},
    ],
    ids=[
        "outcomes_too_short",
        "n_replicates_mismatch",
        "ok_outcome_with_none_width",
        "finite_width_with_reason",
        "negative_width",
        "infinite_width",
        "nan_width",
        "log_axis_threshold_with_difference_scale",
        "unknown_scale",
        "ratio_with_difference_scale",
    ],
)
def test_inconsistent_width_plan_is_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _direct(**overrides)


def test_linear_axis_threshold_plan_with_log_ratio_scale_is_rejected() -> None:
    with pytest.raises(ValueError):
        _direct(axis=_LINEAR, width_scale="log_ratio")


def test_zero_width_is_allowed() -> None:
    result = _direct(widths=(0.0, 1.0, 2.0, 3.0), outcomes=("OK",) * 4)
    assert result.widths[0] == 0.0


def test_width_plan_is_frozen() -> None:
    result = _direct()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.seed = 2  # type: ignore[misc]


# -- the closed outcome label set ----------------------------------------------------------


def test_outcome_label_set_is_ok_plus_status_censoring_shape_and_two_planner_labels() -> None:
    expected = (
        {"OK", "BOTH_SIDES_OPEN", "LOWER_AT_ZERO"}
        | {s.value for s in Status if s is not Status.OK}
        | {c.value for c in Censoring if c is not Censoring.NONE}
        | {i.value for i in IntervalShape if i is not IntervalShape.BOUNDED}
    )
    for label in sorted(expected - {"OK"}):
        built = _direct(widths=(1.0, 2.0, 3.0, None), outcomes=("OK", "OK", "OK", label))
        assert built.outcome_counts()[label] == 1
    assert _ALLOWED_REASONS | {"OK"} >= expected - {"OK"}


@pytest.mark.parametrize("label", ["ok", "separation", "open_lower", "NONE", "BOUNDED", "nonsense"])
def test_an_outcome_label_outside_the_closed_set_is_rejected(label: str) -> None:
    with pytest.raises(ValueError):
        _direct(widths=(1.0, 2.0, 3.0, None), outcomes=("OK", "OK", "OK", label))


def test_lower_at_zero_and_both_sides_open_are_distinct_from_open_lower() -> None:
    built = _direct(
        widths=(None, None, None, 1.0),
        outcomes=("LOWER_AT_ZERO", "BOTH_SIDES_OPEN", "OPEN_LOWER", "OK"),
    )
    assert built.outcome_counts() == {
        "LOWER_AT_ZERO": 1,
        "BOTH_SIDES_OPEN": 1,
        "OPEN_LOWER": 1,
        "OK": 1,
    }


# -- regression: a flat pilot used to crash the tally -----------------------------------------


def test_flat_pilot_with_a_profile_interval_tallies_every_replicate_and_never_raises() -> None:
    # Regression: a profile interval that crosses on neither side gave an OK threshold with
    # lo and hi both None, which tripped an assertion in the planner. It is now tallied as
    # BOTH_SIDES_OPEN. A pilot with s near 4 makes the curve nearly flat across the design.
    pilot = _pilot(0.0, 4.0)
    assert pilot.status is Status.OK
    assert pilot.params is not None
    assert pilot.params["s"].value > 2.0
    result = _plan(
        pilot,
        severity=(0.8, 1.0, 1.25),
        trials_per_level=20,
        interval_method="profile",
        n_replicates=40,
    )
    counts = result.outcome_counts()
    assert sum(counts.values()) == 40
    assert counts.get("BOTH_SIDES_OPEN", 0) >= 1
    assert set(counts) <= _ALLOWED_REASONS | {"OK"}


# -- WidthPlan field validation ------------------------------------------------------------


def test_width_plan_construction_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        constructor: Any = WidthPlan
        constructor("threshold")


def test_width_plan_defaults_schema_library_version_and_provenance() -> None:
    import robustmargin

    built = _direct()
    assert built.plan_schema_version == "1"
    assert built.library_version == robustmargin.__version__
    assert built.provenance == {}
    assert built.n_resamples is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": "other"},
        {"severity": (0.25, 0.25, 1.0)},
        {"severity": (1.0, 0.5, 0.25)},
        {"severity": (-1.0, 0.5, 1.0)},
        {"severity": (0.0, 0.0, 1.0)},
        {"severity": (0.0, 1.0)},
        {"severity": (0.5, math.nan, 2.0)},
        {"severity": (0.5, math.inf)},
        {"trials_per_level": 0},
        {"n_clusters": 1},
        {"n_clusters": 3},
        {"n_clusters": 5, "icc": 1.0},
        {"n_clusters": 5, "icc": -0.1},
        {"n_clusters": 5, "icc": math.nan},
        {"icc": 0.3},
        {"interval_method": "bogus"},
        {"ratio_method": "log_delta"},
        {"kind": "ratio", "ratio_method": None},
        {"kind": "ratio", "ratio_method": "bogus"},
        {"dependence": "paired"},
        {"dependence": "clustered"},
        {"kind": "ratio", "ratio_method": "log_delta", "dependence": "clustered"},
        {"kind": "ratio", "ratio_method": "log_delta", "dependence": "paired"},
        {"level": 0.0},
        {"level": 1.0},
        {"level": math.nan},
        {"n_replicates": 0},
        {"seed": -1},
        {"n_resamples": 0},
        {"plan_schema_version": "2"},
        {"widths": (1.0, 2.0, 3.0, math.nan), "outcomes": ("OK",) * 4},
        {"widths": (-1.0, 2.0, 3.0, 4.0), "outcomes": ("OK",) * 4},
    ],
)
def test_width_plan_field_rules_raise_value_error(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _direct(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"trials_per_level": True},
        {"n_replicates": True},
        {"seed": True},
        {"n_resamples": True},
        {"n_clusters": True},
        {"icc": True},
        {"level": True},
        {"trials_per_level": 50.0},
        {"widths": (True, 2.0, 3.0, None)},
        {"axis": "log"},
        {"definition": 0.5},
        {"library_version": 1},
    ],
)
def test_width_plan_wrong_types_raise_type_error(overrides: dict[str, Any]) -> None:
    with pytest.raises(TypeError):
        _direct(**overrides)


def test_a_valid_clustered_and_bootstrap_width_plan_is_accepted() -> None:
    built = _direct(n_clusters=5, icc=0.2, dependence="clustered")
    assert built.dependence == "clustered"
    boot = _direct(interval_method="parametric_bootstrap", n_resamples=99)
    assert boot.n_resamples == 99


def test_weak_ratio_design_with_a_fieller_interval_tallies_lower_at_zero() -> None:
    # A flat pair of pilots and 20 trials at three levels: Fieller's interval often clips a
    # negative lower root to zero, a bounded ratio with no finite log width. That is labelled
    # LOWER_AT_ZERO, not as a censoring outcome. Seed pinned; 11 of 100 replicates showed it
    # when the design was first explored.
    result = plan_ratio_width(
        _pilot(0.0, 1.5),
        _pilot(0.2, 1.5),
        severity=(0.5, 1.0, 2.0),
        trials_per_level=20,
        definition=Definition.absolute(0.5),
        interval_method="delta",
        ratio_method="fieller",
        dependence="independent",
        level=0.95,
        n_replicates=40,
        seed=1,
    )
    counts = result.outcome_counts()
    assert counts.get("LOWER_AT_ZERO", 0) >= 1
    assert sum(counts.values()) == 40
    assert set(counts) <= _ALLOWED_REASONS | {"OK"}
    for width, outcome in zip(result.widths, result.outcomes, strict=True):
        assert (width is None) == (outcome != "OK")
