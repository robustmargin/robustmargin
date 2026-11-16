"""Coverage simulations and a refit cross-check for the bootstrap interval methods.

Marked ``slow`` and excluded from the default run. Run locally with::

    pytest -m slow --no-cov tests/coverage/test_bootstrap_coverage.py

Three parts.

**(a) Coverage of the parametric bootstrap, asserted.** The marginbench-like design (ten
log-spaced levels and a zero-severity control, 400 trials each, probit, asymptotes fixed at 1 and
0, so the fitted model is correctly specified) is simulated ``_REPLICATES`` times. For each
dataset the 0.95 percentile interval is computed with ``_RESAMPLES`` resamples for
``absolute(0.5)`` and ``baseline_fraction(0.8)``, and the share containing the true threshold is
counted.

*Monte Carlo error.* A GLM refit takes about 8 ms, so one dataset costs about
``2 * _RESAMPLES`` refits, and 250 datasets with 199 resamples fit in about twenty minutes. The
standard error of an observed coverage near 0.95 is then ``sqrt(0.95 * 0.05 / 250) = 0.014``, so
``[0.93, 0.97]`` is not a three-standard-error-safe band for a bare comparison. The assertion
follows the other coverage files: it builds the exact (Clopper-Pearson) 99% interval of the
observed coverage and fails only if that interval lies wholly outside ``[0.93, 0.97]``, that is
only when the simulation shows with 99% confidence that true coverage is outside the band. A
percentile interval from 199 resamples is also slightly liberal (the quantile ranks are
``alpha / 2 * (B + 1)``), which is part of what is being measured.

**(b) The parent-starts refit against a full refit, recorded.** Bootstrap replicates on the
estimated-asymptote path refit from the parent fit's starts, with no per-replicate scan. For each
design here, the simulated datasets are refitted a second time by ``fit_dose_response`` (its full
multi-start), and each refit's threshold is read off its own parameters with the arithmetic the
bootstrap uses, so the two values are the same quantity: ``mu + s * z`` for the target, ``-inf`` or
``+inf`` when it is unreachable, ``nan`` when the refit failed. (``threshold()`` is not used on the
refit: its exact tests classify some replicate datasets as censored, which a bootstrap replicate
never does, and counting those as disagreements would compare unlike things.) The agreement rate is
recorded, not asserted: there is no justified floor until it has been measured. What is asserted
is that every replicate was compared and every value is a float.

**(c) Isotonic coverage, recorded.** The ``cell_bootstrap`` interval on an isotonic fit, on a
rich and on a coarse design. Interpolation between tested severities is biased on a coarse grid by
construction, so poor coverage there is a finding, not a bug, and nothing is asserted about it.

The printed results are the numbers to quote.
"""

from __future__ import annotations

import functools
import importlib
import math
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.stats import binomtest, norm

from robustmargin import (
    Axis,
    Censoring,
    ContinuousObservations,
    Definition,
    Fit,
    Observations,
    Status,
    fit_dose_response,
    threshold,
)

# One seed per part, recorded so a reported number can be reproduced exactly. Changing a seed
# changes the reported numbers and is a deliberate act, not a tidy-up.
_SEED_COVERAGE = 20261120
_SEED_REFIT = 20261121
_SEED_ISOTONIC = 20261122

_LEVEL = 0.95
_AXIS = Axis(name="severity", unit="unit", scale="log")
_RICH_SEVERITY = np.concatenate([[0.0], np.logspace(-1.0, 1.0, 10)])

# ---------------------------------------------------------------------------------------------
# (a) coverage of the parametric bootstrap
# ---------------------------------------------------------------------------------------------

_REPLICATES = 250
_RESAMPLES = 199
_TRUE_MU, _TRUE_S = 0.0, 0.5


def _p_decreasing(severity: np.ndarray, *, mu: float, s: float, upper: float, lower: float):  # type: ignore[no-untyped-def]
    with np.errstate(divide="ignore"):
        z = (np.log(severity) - mu) / s
    return np.where(severity == 0.0, upper, upper - (upper - lower) * norm.cdf(z))


_DEFINITIONS_A = {
    "absolute(0.5)": (Definition.absolute(0.5), math.exp(_TRUE_MU + _TRUE_S * norm.ppf(0.5))),
    # u = 1, l = 0: baseline_fraction(0.8) is the level 0.8, where F(z) = 0.2.
    "baseline_fraction(0.8)": (
        Definition.baseline_fraction(0.8),
        math.exp(_TRUE_MU + _TRUE_S * norm.ppf(0.2)),
    ),
}


@functools.cache
def _simulate_coverage() -> dict[str, dict[str, int]]:
    rng = np.random.default_rng(_SEED_COVERAGE)
    probability = _p_decreasing(_RICH_SEVERITY, mu=_TRUE_MU, s=_TRUE_S, upper=1.0, lower=0.0)
    tallies = {name: {"covered": 0, "usable": 0, "open": 0, "failed": 0} for name in _DEFINITIONS_A}
    for i in range(_REPLICATES):
        successes = rng.binomial(400, probability)
        obs = Observations.from_counts(
            _AXIS,
            severity=_RICH_SEVERITY.tolist(),
            successes=successes.tolist(),
            trials=[400] * _RICH_SEVERITY.size,
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        for name, (definition, truth) in _DEFINITIONS_A.items():
            tally = tallies[name]
            if fit.status is not Status.OK:
                tally["failed"] += 1
                continue
            t = threshold(
                fit,
                definition=definition,
                interval_method="parametric_bootstrap",
                level=_LEVEL,
                n_resamples=_RESAMPLES,
                seed=_SEED_COVERAGE + i,
            )
            if t.status is not Status.OK:
                tally["failed"] += 1
            elif t.censoring is not Censoring.NONE or t.lo is None or t.hi is None:
                tally["open"] += 1
            else:
                tally["usable"] += 1
                tally["covered"] += t.lo <= truth <= t.hi
    return tallies


@pytest.mark.slow
@pytest.mark.parametrize("definition_name", sorted(_DEFINITIONS_A))
def test_parametric_bootstrap_coverage_on_the_rich_design(definition_name: str) -> None:
    tally = _simulate_coverage()[definition_name]
    usable = tally["usable"]
    ci = binomtest(tally["covered"], usable).proportion_ci(confidence_level=0.99, method="exact")
    coverage = tally["covered"] / usable if usable else float("nan")
    print(
        f"\nrich (10 levels + control, n=400) | {definition_name} | parametric_bootstrap\n"
        f"  true threshold : {_DEFINITIONS_A[definition_name][1]:.6f}\n"
        f"  coverage       : {coverage:.4f} of {usable} usable replicates (nominal {_LEVEL}),\n"
        f"                   exact 99% interval [{ci.low:.4f}, {ci.high:.4f}]\n"
        f"  open or censored: {tally['open']}   failed: {tally['failed']}\n"
        f"  seed           : {_SEED_COVERAGE}, datasets {_REPLICATES}, resamples {_RESAMPLES}"
    )

    assert usable >= 0.9 * _REPLICATES
    assert ci.high >= 0.93 and ci.low <= 0.97, (
        f"parametric_bootstrap coverage {coverage:.4f} (99% interval [{ci.low:.4f}, "
        f"{ci.high:.4f}]) is inconsistent with [0.93, 0.97] on the rich design for "
        f"{definition_name}; this is a method finding, not a flaky test"
    )


# ---------------------------------------------------------------------------------------------
# (b) the parent-starts refit against a full refit
# ---------------------------------------------------------------------------------------------


def _seam():  # type: ignore[no-untyped-def]
    """``robustmargin.threshold._bootstrap_replicates``, the private seam this test reads.

    ``_bootstrap_replicates(fit, *, definition, method, n_resamples, seed)`` returns one
    ``(data, theta)`` pair per replicate: the simulated ``Observations`` (binomial) or
    ``ContinuousObservations`` (Gaussian) in the fit's own direction, ready for
    ``fit_dose_response``, and the replicate's solved value on the theta scale (``log x`` on a log
    axis) from the parent-starts refit, ``-inf`` or ``+inf`` for an unreachable replicate by side,
    and ``nan`` for a failed refit.
    """
    # `robustmargin.threshold` the attribute is the function, so go through the module table.
    module = importlib.import_module("robustmargin.threshold")
    return module._bootstrap_replicates


@dataclass(frozen=True)
class _RefitDesign:
    name: str
    model: str
    upper: str | float
    lower: str | float
    replicates: int


_REFIT_DESIGNS = (
    _RefitDesign("binomial, upper estimated, rich", "binomial", "estimate", 0.0, 40),
    _RefitDesign("binomial, upper estimated, coarse", "binomial", "estimate", 0.0, 40),
    _RefitDesign("binomial, both estimated, rich", "binomial", "estimate", "estimate", 15),
    _RefitDesign("gaussian, both estimated", "gaussian", "estimate", "estimate", 15),
)


def _refit_parent(design: _RefitDesign) -> tuple[Fit, Definition]:
    rng = np.random.default_rng(_SEED_REFIT)
    if design.model == "gaussian":
        levels = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
        severity = [x for x in levels for _ in range(4)]
        x = np.asarray(severity)
        with np.errstate(divide="ignore"):
            z = (np.log(x) - math.log(0.8)) / 0.5
        share = np.where(x == 0.0, 0.0, 1.0 / (1.0 + np.exp(-z)))
        value = 10.0 - 9.0 * share + rng.normal(0.0, 0.3, size=x.size)
        gaussian = ContinuousObservations.from_observations(
            _AXIS,
            severity=severity,
            value=value.tolist(),
            outcome="response",
            direction="decreasing",
        )
        fit = fit_dose_response(
            gaussian, model="gaussian", link="logit", upper="estimate", lower="estimate"
        )
        definition = Definition.absolute(5.5)
    else:
        coarse = "coarse" in design.name
        severity_arr = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0]) if coarse else _RICH_SEVERITY
        trials = 60 if coarse else 400
        lower = 0.05 if design.lower == "estimate" else 0.0
        p = _p_decreasing(severity_arr, mu=0.0, s=_TRUE_S, upper=0.92, lower=lower)
        obs = Observations.from_counts(
            _AXIS,
            severity=severity_arr.tolist(),
            successes=rng.binomial(trials, p).tolist(),
            trials=[trials] * severity_arr.size,
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(
            obs, model="binomial", link="probit", upper=design.upper, lower=design.lower
        )
        definition = Definition.absolute(0.5)
    assert fit.status is Status.OK, f"parent fit for {design.name!r} is {fit.status}"
    return fit, definition


def _theta_from_params(refit: Fit, definition: Definition) -> float:
    """The replicate's theta read straight off its refit, by the same arithmetic the bootstrap
    uses: ``absolute(a)`` on a decreasing curve ``u - (u - l) F(z)`` solves ``F(z) = (u - a) /
    (u - l)`` and ``theta = mu + s z``; ``-inf`` when ``a >= u`` and ``+inf`` when ``a <= l``.

    Going through ``threshold()`` here would compare unlike things: its exact tests classify some
    replicate datasets as censored, which a bootstrap replicate never does.
    """
    assert refit.params is not None and definition.kind == "absolute"
    upper, lower = refit.params["upper"].value, refit.params["lower"].value
    target = definition.value
    if target >= upper:
        return -math.inf
    if target <= lower:
        return math.inf
    q = (upper - target) / (upper - lower)
    z = {
        "probit": lambda p: float(norm.ppf(p)),
        "logit": lambda p: math.log(p / (1.0 - p)),
        "cloglog": lambda p: math.log(-math.log(1.0 - p)),
    }[refit.link](q)
    return refit.params["mu"].value + refit.params["s"].value * z


def _full_refit_theta(parent: Fit, data: object, definition: Definition) -> float:
    assert parent.params is not None
    asymptotes = {
        name: ("estimate" if not parent.params[name].fixed else parent.params[name].value)
        for name in ("upper", "lower")
    }
    refit = fit_dose_response(
        data,  # type: ignore[call-overload]
        model=parent.model,
        link=parent.link,
        **asymptotes,
    )
    if refit.status is not Status.OK:
        return math.nan
    return _theta_from_params(refit, definition)


def _same_outcome(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b  # the same side
    return abs(a - b) <= 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("design", _REFIT_DESIGNS, ids=lambda d: d.name)
def test_parent_starts_refit_against_a_full_refit_is_recorded(design: _RefitDesign) -> None:
    parent, definition = _refit_parent(design)

    replicates = _seam()(
        parent,
        definition=definition,
        method="parametric_bootstrap",
        n_resamples=design.replicates,
        seed=_SEED_REFIT,
    )

    agree = 0
    finite_both = 0
    disagreements: list[tuple[float, float]] = []
    for data, theta in replicates:
        full = _full_refit_theta(parent, data, definition)
        if _same_outcome(theta, full):
            agree += 1
        else:
            disagreements.append((theta, full))
        finite_both += math.isfinite(theta) and math.isfinite(full)
    print(
        f"\nparent-starts refit vs full refit | {design.name}\n"
        f"  agreement      : {agree} of {len(replicates)} replicates "
        f"({agree / len(replicates):.3f}), {finite_both} finite in both\n"
        f"  disagreements  : {[(round(a, 4), round(b, 4)) for a, b in disagreements[:10]]}\n"
        f"  seed           : {_SEED_REFIT}"
    )

    # Recorded, not asserted: every replicate was compared, and each value is a plain float.
    assert len(replicates) == design.replicates
    assert all(isinstance(theta, float) for _, theta in replicates)


# ---------------------------------------------------------------------------------------------
# (c) isotonic coverage, recorded
# ---------------------------------------------------------------------------------------------

_ISOTONIC_REPLICATES = 300
_ISOTONIC_RESAMPLES = 199
_ISO_MU = math.log(0.8)


@dataclass(frozen=True)
class _IsotonicDesign:
    name: str
    severity: tuple[float, ...]
    trials: int


_ISOTONIC_DESIGNS = (
    _IsotonicDesign("rich (10 levels + control, n=400)", tuple(_RICH_SEVERITY.tolist()), 400),
    _IsotonicDesign("coarse (4 levels, n=100)", (0.25, 0.5, 1.0, 2.0), 100),
)


@pytest.mark.slow
@pytest.mark.parametrize("design", _ISOTONIC_DESIGNS, ids=lambda d: d.name)
def test_isotonic_cell_bootstrap_coverage_is_recorded(design: _IsotonicDesign) -> None:
    rng = np.random.default_rng(_SEED_ISOTONIC)
    severity = np.asarray(design.severity)
    p = _p_decreasing(severity, mu=_ISO_MU, s=_TRUE_S, upper=1.0, lower=0.0)
    truth = math.exp(_ISO_MU)  # F(z) = 0.5 at z = 0
    tally = {"covered": 0, "usable": 0, "open": 0, "unreachable": 0, "other": 0}

    for i in range(_ISOTONIC_REPLICATES):
        obs = Observations.from_counts(
            _AXIS,
            severity=severity.tolist(),
            successes=rng.binomial(design.trials, p).tolist(),
            trials=[design.trials] * severity.size,
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="isotonic")
        t = threshold(
            fit,
            definition=Definition.absolute(0.5),
            interval_method="cell_bootstrap",
            level=_LEVEL,
            n_resamples=_ISOTONIC_RESAMPLES,
            seed=_SEED_ISOTONIC + i,
        )
        if t.status is Status.UNREACHABLE:
            tally["unreachable"] += 1
        elif t.status is not Status.OK:
            tally["other"] += 1
        elif t.censoring is not Censoring.NONE or t.lo is None or t.hi is None:
            tally["open"] += 1
        else:
            tally["usable"] += 1
            tally["covered"] += t.lo <= truth <= t.hi

    usable = tally["usable"]
    coverage = tally["covered"] / usable if usable else float("nan")
    print(
        f"\nisotonic | {design.name} | cell_bootstrap\n"
        f"  true threshold : {truth:.6f}\n"
        f"  coverage       : {coverage:.4f} of {usable} usable replicates (nominal {_LEVEL})\n"
        f"  open or censored: {tally['open']}   unreachable: {tally['unreachable']}   "
        f"other: {tally['other']}\n"
        f"  seed           : {_SEED_ISOTONIC}, datasets {_ISOTONIC_REPLICATES}, "
        f"resamples {_ISOTONIC_RESAMPLES}"
    )

    # Recorded, not asserted: a threshold was produced for every dataset.
    assert sum(tally.values()) - tally["covered"] == _ISOTONIC_REPLICATES
