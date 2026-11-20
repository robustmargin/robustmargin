"""The simulated experiment of the width planner: truth, stream, refit and early validation.

* The success probability the planner simulates from is the pilot fit's own curve, for every
  link, axis scale, asymptote specification and direction (the zero-severity control included).
* The draw order is fixed: replicate ``i`` of an independent fixed-asymptote design is rebuilt
  here, with its own loop, from ``SeedSequence(seed).spawn(n)[i]`` and must give the planner's
  width bit for bit.
* The refit uses the pilot's link and asymptote specification, and ``dependence="clustered"``
  for a clustered design.
* Invalid method combinations raise before any replicate runs.
"""

from __future__ import annotations

import math
import time
from typing import Any, Literal

import numpy as np
import pytest
from scipy.stats import norm

import robustmargin.design as design_module
from robustmargin import Axis, Definition, Fit, Observations, Status, fit_dose_response, threshold
from robustmargin.design import plan_ratio_width, plan_threshold_width

_LOG = Axis(name="severity", unit="unit", scale="log")
_LINEAR = Axis(name="severity", unit="unit", scale="linear")
_SEED = 20261123


def _pilot(
    axis: Axis,
    *,
    link: Literal["probit", "logit", "cloglog"] = "probit",
    direction: str = "decreasing",
    upper: float | Literal["estimate"] = 1.0,
    lower: float | Literal["estimate"] = 0.0,
    control: bool = False,
    mu: float | None = None,
    s: float | None = None,
) -> Fit:
    """A converged binomial pilot on exact (rounded) counts of a curve with known asymptotes."""
    log = axis.scale == "log"
    levels = (
        np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]) if log else np.linspace(1.0, 8.0, 8)
    )
    mu_value = (0.0 if log else 4.5) if mu is None else mu
    s_value = (0.6 if log else 1.5) if s is None else s
    u_true = 0.95 if upper == "estimate" else float(upper)
    l_true = 0.05 if lower == "estimate" else float(lower)
    z = ((np.log(levels) if log else levels) - mu_value) / s_value
    decreasing = u_true - (u_true - l_true) * norm.cdf(z)
    p = (
        decreasing
        if direction == "decreasing"
        else 1.0 - (1.0 - l_true) + (u_true - l_true) * (norm.cdf(z))
    )
    severity = list(levels)
    successes = [int(round(400 * v)) for v in p]
    if control:
        severity.insert(0, 0.0)
        successes.insert(0, int(round(400 * (u_true if direction == "decreasing" else l_true))))
    obs = Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=[400] * len(severity),
        outcome="outcome",
        direction=direction,
    )
    fit = fit_dose_response(obs, model="binomial", link=link, upper=upper, lower=lower)
    assert fit.status is Status.OK
    return fit


# -- the simulated truth is the pilot's own curve --------------------------------------------


@pytest.mark.parametrize("direction", ["increasing", "decreasing"])
@pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
@pytest.mark.parametrize("scale", ["log", "linear"])
# Estimated-asymptote pilots take about 15 s each to fit, so those cases run only in the slow tier.
@pytest.mark.parametrize(
    "asymptotes",
    [
        (1.0, 0.0),
        (0.95, 0.05),
        pytest.param(("estimate", "estimate"), marks=pytest.mark.slow),
    ],
    ids=["01", "fixed", "est"],
)
def test_simulated_success_probability_equals_the_pilot_curve(
    direction: str, link: Any, scale: str, asymptotes: tuple[Any, Any]
) -> None:
    axis = _LOG if scale == "log" else _LINEAR
    upper, lower = asymptotes
    curve = _pilot(axis, link=link, direction=direction, upper=upper, lower=lower, control=True)
    x = np.array([0.0, 0.15, 0.5, 1.3, 3.0, 7.0] if scale == "log" else [0.0, 2.0, 4.0, 6.0, 8.0])
    trials = 10**12
    counts = design_module._simulate_counts(
        [curve], list(x), trials, rng=np.random.default_rng(_SEED)
    )
    assert counts.shape == (1, x.size, 1)
    simulated = counts[0, :, 0] / trials
    expected = np.asarray(curve.predict(x).estimate)
    # binomial sd at 1e12 trials is at most 5e-7; 1e-5 is twenty of them.
    np.testing.assert_allclose(simulated, expected, atol=1e-5, rtol=0.0)


# -- the refit is the pilot's analysis -------------------------------------------------------


@pytest.mark.parametrize(
    ("link", "upper", "lower"),
    [
        ("probit", 1.0, 0.0),
        ("logit", 0.95, 0.05),
        pytest.param("cloglog", "estimate", "estimate", marks=pytest.mark.slow),
    ],
)
def test_refit_uses_the_pilot_link_and_asymptote_specification(
    monkeypatch: pytest.MonkeyPatch, link: Any, upper: Any, lower: Any
) -> None:
    pilot = _pilot(_LOG, link=link, upper=upper, lower=lower)
    calls: list[dict[str, Any]] = []
    real = fit_dose_response

    def spy(*args: Any, **kwargs: Any) -> Fit:
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr("robustmargin.design.fit_dose_response", spy)
    plan_threshold_width(
        pilot,
        severity=(0.25, 0.5, 1.0, 2.0, 4.0),
        trials_per_level=50,
        definition=Definition.absolute(0.5),
        interval_method="delta",
        level=0.95,
        n_replicates=2,
        seed=_SEED,
    )
    assert len(calls) == 2
    for kwargs in calls:
        assert kwargs["link"] == link
        assert kwargs["upper"] == upper
        assert kwargs["lower"] == lower
        assert kwargs["model"] == "binomial"
        assert kwargs.get("dependence") is None


def test_clustered_design_refits_with_cluster_robust_dependence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    real = fit_dose_response

    def spy(*args: Any, **kwargs: Any) -> Fit:
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr("robustmargin.design.fit_dose_response", spy)
    plan_threshold_width(
        _pilot(_LOG),
        severity=(0.25, 0.5, 1.0, 2.0, 4.0),
        trials_per_level=50,
        definition=Definition.absolute(0.5),
        interval_method="delta",
        level=0.95,
        n_replicates=2,
        seed=_SEED,
        n_clusters=5,
        icc=0.2,
    )
    assert [c["dependence"] for c in calls] == ["clustered", "clustered"]


# -- the draw order is fixed: replay a replicate -----------------------------------------------


def test_independent_fixed_asymptote_replicates_replay_exactly_from_their_spawned_streams() -> None:
    pilot = _pilot(_LOG)
    assert pilot.params is not None
    mu, s = pilot.params["mu"].value, pilot.params["s"].value
    levels = (0.25, 0.5, 1.0, 2.0, 4.0)
    n_replicates, trials = 5, 40
    definition = Definition.absolute(0.5)
    result = plan_threshold_width(
        pilot,
        severity=levels,
        trials_per_level=trials,
        definition=definition,
        interval_method="delta",
        level=0.95,
        n_replicates=n_replicates,
        seed=_SEED,
    )
    x = np.array(levels)
    p = 1.0 - norm.cdf((np.log(x) - mu) / s)
    streams = np.random.SeedSequence(_SEED).spawn(n_replicates)
    for index, child in enumerate(streams):
        counts = np.random.default_rng(child).binomial(trials, p)
        obs = Observations.from_counts(
            _LOG,
            severity=list(levels),
            successes=[int(k) for k in counts],
            trials=[trials] * len(levels),
            outcome="outcome",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        expected: float | None = None
        if fit.status is Status.OK:
            t = threshold(fit, definition=definition, interval_method="delta", level=0.95)
            if t.status is Status.OK and t.lo is not None and t.hi is not None and t.lo > 0.0:
                expected = math.log(t.hi / t.lo)
        assert result.widths[index] == expected
    assert any(w is not None for w in result.widths)


# -- reproducibility of the bootstrap methods --------------------------------------------------


@pytest.mark.parametrize("method", ["parametric_bootstrap", "cell_bootstrap"])
def test_bootstrap_interval_plan_is_reproducible_for_a_seed_and_any_worker_count(
    method: str,
) -> None:
    # Regression: bootstrap calls inside a replicate were unseeded, so two runs of one seed
    # gave different widths. Each replicate now derives its bootstrap seed from its own stream.
    kwargs: dict[str, Any] = {
        "severity": (0.25, 0.5, 1.0, 2.0, 4.0),
        "trials_per_level": 50,
        "definition": Definition.absolute(0.5),
        "interval_method": method,
        "level": 0.95,
        "n_replicates": 2,
        "seed": _SEED,
        "n_resamples": 49,
    }
    pilot = _pilot(_LOG)
    first = plan_threshold_width(pilot, **kwargs)
    again = plan_threshold_width(pilot, **kwargs)
    parallel = plan_threshold_width(pilot, workers=2, **kwargs)
    assert first == again
    assert first == parallel
    assert first.n_resamples == 49


def test_n_resamples_with_a_non_bootstrap_interval_method_raises() -> None:
    with pytest.raises(ValueError):
        plan_threshold_width(
            _pilot(_LOG),
            severity=(0.25, 0.5, 1.0, 2.0, 4.0),
            trials_per_level=50,
            definition=Definition.absolute(0.5),
            interval_method="delta",
            level=0.95,
            n_replicates=2,
            seed=_SEED,
            n_resamples=49,
        )


# -- validation happens before simulation ------------------------------------------------------


def _fast(call: Any) -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError):
        call()
    assert time.perf_counter() - started < 5.0, "validation ran after simulation had started"


def test_profile_interval_for_a_clustered_design_raises_before_any_replicate() -> None:
    _fast(
        lambda: plan_threshold_width(
            _pilot(_LOG),
            severity=(0.25, 0.5, 1.0, 2.0, 4.0),
            trials_per_level=50,
            definition=Definition.absolute(0.5),
            interval_method="profile",
            level=0.95,
            n_replicates=10_000,
            seed=_SEED,
            n_clusters=5,
            icc=0.2,
        )
    )


def test_unknown_threshold_interval_method_raises_before_any_replicate() -> None:
    _fast(
        lambda: plan_threshold_width(
            _pilot(_LOG),
            severity=(0.25, 0.5, 1.0, 2.0, 4.0),
            trials_per_level=50,
            definition=Definition.absolute(0.5),
            interval_method="bogus",
            level=0.95,
            n_replicates=10_000,
            seed=_SEED,
        )
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"ratio_method": "bogus"},
        {"ratio_method": "cluster_bootstrap", "dependence": "independent"},
        {"interval_method": "bogus"},
        {"interval_method": "profile", "n_clusters": 5, "icc": 0.2},
    ],
    ids=[
        "unknown_ratio_method",
        "cluster_bootstrap_independent",
        "unknown_interval",
        "profile_clusters",
    ],
)
def test_invalid_ratio_method_combinations_raise_before_any_replicate(
    overrides: dict[str, Any],
) -> None:
    kwargs: dict[str, Any] = {
        "severity": (0.25, 0.5, 1.0, 2.0, 4.0),
        "trials_per_level": 50,
        "definition": Definition.absolute(0.5),
        "interval_method": "delta",
        "ratio_method": "log_delta",
        "dependence": "independent",
        "level": 0.95,
        "n_replicates": 10_000,
        "seed": _SEED,
    }
    kwargs.update(overrides)
    _fast(lambda: plan_ratio_width(_pilot(_LOG, mu=0.0), _pilot(_LOG, mu=0.4), **kwargs))
