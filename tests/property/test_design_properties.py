"""Property of the width planner (hypothesis, derandomised, few examples).

More trials per level cannot widen the typical interval. With the pilot, seed and replicate
count fixed, quadrupling ``trials_per_level`` does not raise the median width. A replicate with
no finite width counts as an infinite width, so a design that fails less often moves the median
down, never up.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from robustmargin import Axis, Definition, Fit, Observations, fit_dose_response
from robustmargin.design import plan_threshold_width

_SETTINGS = settings(
    max_examples=4,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
_LOG = Axis(name="severity", unit="unit", scale="log")
_LEVELS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def _pilot(mu: float, s: float) -> Fit:
    levels = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8])
    p = 1.0 - norm.cdf((np.log(levels) - mu) / s)
    obs = Observations.from_counts(
        _LOG,
        severity=list(levels),
        successes=[int(round(200 * v)) for v in p],
        trials=[200] * len(levels),
        outcome="outcome",
        direction="decreasing",
    )
    return fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)


@_SETTINGS
@given(
    mu=st.sampled_from([-0.3, 0.0, 0.3]),
    s=st.sampled_from([0.4, 0.6]),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_quadrupling_trials_per_level_does_not_raise_the_median_width(
    mu: float, s: float, seed: int
) -> None:
    pilot = _pilot(mu, s)
    common: dict[str, Any] = {
        "severity": _LEVELS,
        "definition": Definition.absolute(0.5),
        "interval_method": "delta",
        "level": 0.95,
        "n_replicates": 15,
        "seed": seed,
    }
    small = plan_threshold_width(pilot, trials_per_level=50, **common)
    large = plan_threshold_width(pilot, trials_per_level=200, **common)
    assert large.quantile(0.5) <= small.quantile(0.5)
