"""Coverage simulations for threshold intervals.

These are the only tests in the suite that check the *method* rather than the implementation.
The reference tests in ``tests/reference/`` prove robustmargin computes the same numbers R does;
they cannot tell whether a 95% interval covers the truth 95% of the time. That is what this
file measures, by simulating from a known probit and counting how often the interval contains
the true threshold.

Marked ``slow`` and excluded from the default run by ``pyproject.toml``'s ``-m "not slow"``.
CI runs them once, in the ``coverage-simulations`` job, rather than on each of four matrix legs.
Run locally with::

    pytest -m slow --no-cov

**What is asserted and what is only recorded**: the rich design (a log-spaced grid with a
zero-severity control) asserts profile coverage in ``[0.93, 0.97]``. The coarse 4-level design
**records without asserting** -- poor coverage on a 4-level grid is a finding to report, not a
bug. The printed results are the numbers to quote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.stats import norm

from robustmargin import (
    Axis,
    Censoring,
    Definition,
    Observations,
    Status,
    fit_dose_response,
    threshold,
)

# One seed per design, recorded here so a reported number can be reproduced exactly. Changing a
# seed changes the reported numbers and is a deliberate act, not a tidy-up.
_SEED_COARSE = 20260921
_SEED_RICH = 20260922

_REPLICATES = 2000
_LEVEL = 0.95

# The simulation truth: a probit on a log axis, upper and lower asymptotes fixed at 1 and 0, so
# the fitted model is correctly specified and any coverage shortfall is the interval's, not the
# model's.
_TRUE_MU = 0.0
_TRUE_S = 0.5
_AXIS = Axis(name="severity", unit="unit", scale="log")


def _true_threshold(target: float) -> float:
    """The severity at which the true curve crosses ``target`` (with u=1, l=0)."""
    return float(math.exp(_TRUE_MU + _TRUE_S * norm.ppf(1.0 - target)))


def _true_success_probability(severity: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        z = (np.log(severity) - _TRUE_MU) / _TRUE_S
    p = 1.0 - norm.cdf(z)
    return np.where(severity == 0.0, 1.0, p)


@dataclass(frozen=True)
class _Design:
    name: str
    severity: np.ndarray
    trials: np.ndarray
    seed: int
    asserted: bool


_COARSE = _Design(
    name="coarse (4 levels, n=100-500)",
    severity=np.array([0.25, 0.5, 1.0, 2.0]),
    trials=np.array([500, 300, 200, 100]),
    seed=_SEED_COARSE,
    asserted=False,
)
_RICH = _Design(
    name="rich (10 log-spaced levels + control, n=400)",
    severity=np.concatenate([[0.0], np.logspace(-1.0, 1.0, 10)]),
    trials=np.full(11, 400),
    seed=_SEED_RICH,
    asserted=True,
)

_DEFINITIONS = {
    "absolute(0.5)": (Definition.absolute(0.5), 0.5),
    "baseline_fraction(0.8)": (Definition.baseline_fraction(0.8), 0.8),
}


def _coverage(
    design: _Design, definition: Definition, target: float, method: str
) -> dict[str, float]:
    """Fraction of replicates whose interval contains the true threshold.

    A replicate whose fit failed, or whose threshold is censored or has no bounds
    (no closed interval), is **not** counted as covered and **not** silently dropped: it is
    tallied separately, because "the interval missed" and "there was no interval" are different
    findings and averaging them together would hide both.
    """
    rng = np.random.default_rng(design.seed)
    truth = _true_threshold(target)
    probability = _true_success_probability(design.severity)

    covered = 0
    usable = 0
    censored = 0
    unbounded = 0
    failed = 0

    for _ in range(_REPLICATES):
        successes = rng.binomial(design.trials, probability)
        obs = Observations.from_counts(
            _AXIS,
            severity=design.severity.tolist(),
            successes=successes.tolist(),
            trials=design.trials.tolist(),
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        if fit.status is not Status.OK:
            failed += 1
            continue
        t = threshold(fit, definition=definition, interval_method=method, level=_LEVEL)
        if t.status is not Status.OK:
            failed += 1
            continue
        if t.censoring is not Censoring.NONE:
            censored += 1
            continue
        if t.lo is None or t.hi is None:
            unbounded += 1
            continue
        usable += 1
        if t.lo <= truth <= t.hi:
            covered += 1

    return {
        "coverage": covered / usable if usable else float("nan"),
        "usable": usable,
        "censored": censored,
        "unbounded": unbounded,
        "failed": failed,
        "truth": truth,
    }


@pytest.mark.slow
@pytest.mark.parametrize("design", [_COARSE, _RICH], ids=lambda d: d.name)
@pytest.mark.parametrize("definition_name", sorted(_DEFINITIONS))
@pytest.mark.parametrize("method", ["profile", "delta"])
def test_interval_coverage(design: _Design, definition_name: str, method: str) -> None:
    definition, target = _DEFINITIONS[definition_name]
    result = _coverage(design, definition, target, method)

    print(
        f"\n{design.name} | {definition_name} | {method}\n"
        f"  true threshold : {result['truth']:.6f}\n"
        f"  coverage       : {result['coverage']:.4f} of {int(result['usable'])} usable "
        f"replicates (nominal {_LEVEL})\n"
        f"  censored       : {int(result['censored'])}\n"
        f"  unbounded      : {int(result['unbounded'])}\n"
        f"  failed         : {int(result['failed'])}\n"
        f"  seed           : {design.seed}, replicates {_REPLICATES}"
    )

    # Only the rich design's profile coverage is asserted. The coarse design and the delta
    # cross-check are recorded.
    if design.asserted and method == "profile":
        assert 0.93 <= result["coverage"] <= 0.97, (
            f"profile coverage {result['coverage']:.4f} outside [0.93, 0.97] on the rich design "
            f"for {definition_name}; this is a method finding, not a flaky test"
        )
