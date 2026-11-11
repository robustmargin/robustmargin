"""Coverage simulations for Gaussian (continuous-response) threshold intervals.

The reference tests prove robustmargin computes what R ``drc`` computes. They cannot say whether a
95% interval covers the truth 95% of the time. This file simulates from a known four-parameter
logistic curve (log axis, zero-severity control, homoscedastic Gaussian noise), fits it with the
correctly specified model, and counts how often the profile and delta intervals at 0.95 contain
the **true** threshold.

Marked ``slow`` and excluded from the default run, like the binomial coverage simulations. Run
locally with::

    pytest -m slow --no-cov tests/coverage/test_continuous_coverage.py

**What is asserted and what is only recorded.** The moderate design (8 levels with a control, 8
observations per level) asserts. The small design (6 levels with a control, 3 observations per
level) **records without asserting**: poor coverage there is a finding, not a bug.

**How the assertion is judged.** With a few hundred replicates the observed coverage carries
about a percentage point of simulation noise, so a bare ``0.93 <= observed <= 0.97`` would
fail or pass on the seed. The assertion instead builds the exact (Clopper-Pearson) 99% interval
of the observed coverage and fails only if it lies wholly outside ``[0.93, 0.97]``, i.e. only
when the simulation shows with 99% confidence that the true coverage is out of that band.

**Cost.** Each design simulates ``_REPLICATES`` datasets once; the fit is shared by both
definitions and both methods. The profile dominates. The estimate is 3 to 10 minutes for the
whole file on one CPU, to be re-measured once the estimator exists, and the replicate count
lowered if it runs over.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.stats import binomtest

from robustmargin import (
    Axis,
    Censoring,
    ContinuousObservations,
    Definition,
    Status,
    fit_dose_response,
    threshold,
)

# One seed per design, recorded so a reported number can be reproduced exactly. Changing a seed
# changes the reported numbers and is a deliberate act, not a tidy-up.
_SEED_SMALL = 20261109
_SEED_MODERATE = 20261110

_REPLICATES = 500
_LEVEL = 0.95

_AXIS = Axis(name="severity", unit="unit", scale="log")
_MU, _S, _U, _L, _SIGMA = math.log(0.8), 0.5, 10.0, 1.0, 0.5

_ABSOLUTE_TARGET = 7.0
_TRUTH = {
    "relative(0.5)": math.exp(_MU),
    # u - (u - l) F(z) = 7  ->  F(z) = 1/3, and logit(1/3) = -log(2).
    "absolute(7.0)": math.exp(_MU - _S * math.log(2.0)),
}
_DEFINITIONS = {
    "relative(0.5)": Definition.relative(0.5),
    "absolute(7.0)": Definition.absolute(_ABSOLUTE_TARGET),
}
_METHODS = ("profile", "delta")


@dataclass(frozen=True)
class _Design:
    name: str
    severity: tuple[float, ...]
    per_level: int
    seed: int
    asserted: bool


_SMALL = _Design(
    name="small (6 levels with a control, n=3 per level)",
    severity=(0.0, 0.2, 0.5, 1.0, 2.0, 5.0),
    per_level=3,
    seed=_SEED_SMALL,
    asserted=False,
)
_MODERATE = _Design(
    name="moderate (8 levels with a control, n=8 per level)",
    severity=(0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4),
    per_level=8,
    seed=_SEED_MODERATE,
    asserted=True,
)


def _mean_curve(severity: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        z = (np.log(severity) - _MU) / _S
    share = np.where(severity == 0.0, 0.0, 1.0 / (1.0 + np.exp(-z)))
    return np.asarray(_U - (_U - _L) * share)


@functools.cache
def _simulate(design_name: str) -> dict[tuple[str, str], dict[str, int]]:
    """Tallies per (definition, method), over one shared set of replicates.

    A replicate whose fit failed, whose threshold is not ``OK``, or whose interval is censored or
    open is **not** counted as covered and **not** silently dropped: each is tallied on its own,
    because "the interval missed" and "there was no interval" are different findings.
    """
    design = next(d for d in (_SMALL, _MODERATE) if d.name == design_name)
    rng = np.random.default_rng(design.seed)
    severity = np.repeat(np.asarray(design.severity), design.per_level)
    mean = _mean_curve(severity)
    tallies = {
        key: {"covered": 0, "usable": 0, "open": 0, "failed": 0}
        for key in ((d, m) for d in _DEFINITIONS for m in _METHODS)
    }

    for _ in range(_REPLICATES):
        value = mean + rng.normal(0.0, _SIGMA, size=mean.size)
        obs = ContinuousObservations.from_observations(
            _AXIS,
            severity=severity.tolist(),
            value=value.tolist(),
            outcome="response",
            direction="decreasing",
        )
        fit = fit_dose_response(
            obs, model="gaussian", link="logit", upper="estimate", lower="estimate"
        )
        for (definition_name, method), tally in tallies.items():
            if fit.status is not Status.OK:
                tally["failed"] += 1
                continue
            t = threshold(
                fit,
                definition=_DEFINITIONS[definition_name],
                interval_method=method,
                level=_LEVEL,
            )
            if t.status is not Status.OK:
                tally["failed"] += 1
            elif t.censoring is not Censoring.NONE or t.lo is None or t.hi is None:
                tally["open"] += 1
            else:
                tally["usable"] += 1
                tally["covered"] += t.lo <= _TRUTH[definition_name] <= t.hi
    return tallies


def _report(design: _Design, definition_name: str, method: str) -> dict[str, float]:
    tally = _simulate(design.name)[(definition_name, method)]
    usable = tally["usable"]
    interval = binomtest(tally["covered"], usable).proportion_ci(
        confidence_level=0.99, method="exact"
    )
    coverage = tally["covered"] / usable if usable else float("nan")
    print(
        f"\n{design.name} | {definition_name} | {method}\n"
        f"  true threshold : {_TRUTH[definition_name]:.6f}\n"
        f"  coverage       : {coverage:.4f} of {usable} usable replicates (nominal {_LEVEL}),\n"
        f"                   exact 99% interval [{interval.low:.4f}, {interval.high:.4f}]\n"
        f"  open or censored: {tally['open']}\n"
        f"  failed         : {tally['failed']}\n"
        f"  seed           : {design.seed}, replicates {_REPLICATES}"
    )
    return {"low": interval.low, "high": interval.high, "usable": usable, "coverage": coverage}


@pytest.mark.slow
@pytest.mark.parametrize("design", [_SMALL, _MODERATE], ids=lambda d: d.name)
@pytest.mark.parametrize("definition_name", sorted(_DEFINITIONS))
@pytest.mark.parametrize("method", _METHODS)
def test_interval_coverage(design: _Design, definition_name: str, method: str) -> None:
    result = _report(design, definition_name, method)

    # Only the moderate design is asserted. The small design is recorded.
    if design.asserted:
        assert result["usable"] >= 0.9 * _REPLICATES, (
            f"only {int(result['usable'])} of {_REPLICATES} replicates gave a closed interval on "
            f"the moderate design for {definition_name}/{method}; that is a method finding"
        )
        assert result["high"] >= 0.93 and result["low"] <= 0.97, (
            f"{method} coverage {result['coverage']:.4f} (99% interval "
            f"[{result['low']:.4f}, {result['high']:.4f}]) is inconsistent with [0.93, 0.97] on "
            f"the moderate design for {definition_name}; this is a method finding, not a flaky "
            "test"
        )
