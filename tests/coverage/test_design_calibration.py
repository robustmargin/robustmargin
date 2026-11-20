"""Calibration of the width planner against an independent direct simulation.

The planner is a simulation of an analysis; this checks it is the analysis it claims to be. One
design (fixed asymptotes, log axis, delta interval, independent trials) is run two ways:

* ``plan_threshold_width`` on the pilot fit, and
* a loop written here that draws binomial counts from the pilot's point estimates, refits with
  ``fit_dose_response`` and calls ``threshold``, scoring ``log(hi / lo)`` and counting any
  replicate without a bounded interval as an infinite width.

The two use different random streams, so the check is statistical: at several widths the share of
replicates at or under that width must agree within four combined Monte Carlo standard errors.

Marked ``slow`` and excluded from the default run. Run with ``pytest -m slow --no-cov``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from robustmargin import (
    Axis,
    Definition,
    Fit,
    Observations,
    Status,
    fit_dose_response,
    threshold,
)
from robustmargin.design import plan_threshold_width

pytestmark = pytest.mark.slow

_SEED_PLANNER = 20261121
_SEED_DIRECT = 20261122
_REPLICATES = 400
_LEVEL = 0.95
_TRIALS = 40
_AXIS = Axis(name="severity", unit="unit", scale="log")
_DESIGN = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
_DEFINITION = Definition.absolute(0.5)


def _pilot() -> Fit:
    levels = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8])
    p = 1.0 - norm.cdf(np.log(levels) / 0.5)
    obs = Observations.from_counts(
        _AXIS,
        severity=list(levels),
        successes=[int(round(200 * v)) for v in p],
        trials=[200] * len(levels),
        outcome="outcome",
        direction="decreasing",
    )
    return fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)


def _direct_widths(mu: float, s: float) -> list[float]:
    rng = np.random.default_rng(_SEED_DIRECT)
    x = np.array(_DESIGN)
    p = 1.0 - norm.cdf((np.log(x) - mu) / s)
    widths: list[float] = []
    for _ in range(_REPLICATES):
        k = rng.binomial(_TRIALS, p)
        obs = Observations.from_counts(
            _AXIS,
            severity=list(x),
            successes=[int(v) for v in k],
            trials=[_TRIALS] * len(x),
            outcome="outcome",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        if fit.status is not Status.OK:
            widths.append(math.inf)
            continue
        t = threshold(fit, definition=_DEFINITION, interval_method="delta", level=_LEVEL)
        if t.status is not Status.OK or t.lo is None or t.hi is None or t.lo <= 0.0:
            widths.append(math.inf)
        else:
            widths.append(math.log(t.hi / t.lo))
    return widths


def test_planner_width_distribution_matches_an_independent_direct_simulation() -> None:
    pilot = _pilot()
    assert pilot.params is not None
    result = plan_threshold_width(
        pilot,
        severity=_DESIGN,
        trials_per_level=_TRIALS,
        definition=_DEFINITION,
        interval_method="delta",
        level=_LEVEL,
        n_replicates=_REPLICATES,
        seed=_SEED_PLANNER,
    )
    direct = np.array(_direct_widths(pilot.params["mu"].value, pilot.params["s"].value))

    finite = direct[np.isfinite(direct)]
    assert finite.size > _REPLICATES // 2, "design too poor to calibrate against"
    for q in (0.25, 0.5, 0.75):
        width = float(np.quantile(finite, q))
        share_direct = float(np.mean(direct <= width))
        share_planner, se_planner = result.share_at_most(width)
        se_direct = math.sqrt(share_direct * (1.0 - share_direct) / _REPLICATES)
        combined = math.sqrt(se_planner**2 + se_direct**2)
        print(f"width {width:.4f}: planner {share_planner:.3f} direct {share_direct:.3f}")
        assert abs(share_planner - share_direct) <= 4.0 * max(combined, 1e-12)

    # The failure tally agrees too: the share of replicates with no finite width.
    none_direct = float(np.mean(~np.isfinite(direct)))
    none_planner = sum(w is None for w in result.widths) / _REPLICATES
    se = math.sqrt(2.0 * max(none_direct * (1.0 - none_direct), 0.01) / _REPLICATES)
    assert abs(none_planner - none_direct) <= 4.0 * se
