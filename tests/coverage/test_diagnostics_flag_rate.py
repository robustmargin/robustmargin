"""Coverage simulation for the monotonicity check's false-positive rate.

Marked ``slow`` and excluded from the default run, mirroring
``tests/coverage/test_profile_coverage.py``. Run locally with::

    pytest -m slow --no-cov

Simulates from a genuinely monotone probit (``u=0.9``, ``l=0.1``, ``mu=0``, ``s=0.5``) on an
11-level design (an exact-zero control plus ``logspace(-2, 2, 10)``), ``n=400`` per level, and
counts how often :func:`~robustmargin.check_monotonicity`'s Holm-corrected check raises **any**
false flag across 2,000 replicates. Since the check needs only counts, every replicate is usable
regardless of whether a curve could be fit to it, so there is no "failed fit" category here at
all. This design measures roughly 2-3% with the Holm correction (a per-pair, uncorrected test
is roughly 25% on the same design); this test re-establishes that figure and asserts the bound
``<= 0.05``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from robustmargin import Axis, Observations, check_monotonicity

_SEED = 20260927
_REPLICATES = 2000
_TRUE_MU = 0.0
_TRUE_S = 0.5
_TRUE_UPPER = 0.9
_TRUE_LOWER = 0.1
_AXIS = Axis(name="severity", unit="unit", scale="log")

_SEVERITY = np.concatenate([[0.0], np.logspace(-2.0, 2.0, 10)])
_TRIALS = np.full(_SEVERITY.shape, 400)


def _true_success_probability(severity: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        z = (np.log(severity) - _TRUE_MU) / _TRUE_S
    p = _TRUE_UPPER - (_TRUE_UPPER - _TRUE_LOWER) * norm.cdf(z)
    return np.where(severity == 0.0, _TRUE_UPPER, p)


def _any_flag_rate() -> dict[str, float]:
    rng = np.random.default_rng(_SEED)
    probability = _true_success_probability(_SEVERITY)

    flagged = 0

    for _ in range(_REPLICATES):
        successes = rng.binomial(_TRIALS, probability)
        obs = Observations.from_counts(
            _AXIS,
            severity=_SEVERITY.tolist(),
            successes=successes.tolist(),
            trials=_TRIALS.tolist(),
            outcome="success",
            direction="decreasing",
        )
        check = check_monotonicity(obs)
        if check.any_flagged:
            flagged += 1

    return {"rate": flagged / _REPLICATES, "replicates": _REPLICATES}


@pytest.mark.slow
def test_false_flag_rate_on_a_correctly_monotone_curve_is_at_most_5_percent() -> None:
    result = _any_flag_rate()

    print(
        f"\nmonotonicity false-flag rate (11-level rich design, n=400, {_REPLICATES} replicates)\n"
        f"  rate    : {result['rate']:.4f} of {int(result['replicates'])} replicates\n"
        f"  seed    : {_SEED}\n"
        "  roughly 0.02-0.03 is expected for this design; this run re-establishes "
        "that figure rather than assuming it still holds."
    )

    assert result["rate"] <= 0.05, (
        f"false-flag rate {result['rate']:.4f} exceeds the 0.05 bound on a correctly "
        "monotone curve -- this is a method finding (the Holm correction), not a flaky test"
    )
