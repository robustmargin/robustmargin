"""Properties of ``interval_method="parametric_bootstrap"`` (hypothesis, derandomised).

Fixed-asymptote binomial fits only: a GLM refit is milliseconds, so a property can run many
replicates per example. Each property holds the seed fixed and varies the data or its presentation.

* **Unit scaling.** Rescaling every severity by ``c > 0`` scales ``value``, ``lo`` and ``hi`` by
  ``c`` and leaves the status, censoring and the bootstrap summary alone. The simulated draws
  depend on the fitted probabilities, which are invariant under the rescaling, so the same seed
  gives the same replicate outcomes. Tolerance is relative ``1e-6``: statsmodels reports the
  rescaled fit to its own convergence tolerance, not bit for bit.
* **Row permutation.** The order of the input rows changes nothing, bounds included.
* **Direction flip.** An increasing fit on events ``k`` and a decreasing fit on the complement
  ``n - k`` give the same bounds under one seed (``absolute(0.5)`` is its own complement, and the
  fixed asymptotes ``(1, 0)`` mirror to themselves). Tolerance is relative ``1e-9``.

Designs are drawn so that no level separates and the rates trend across the design, with five to
six levels of 80 to 200 trials. Whatever status results is compared too, so a design that fails
is checked for failing identically.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from robustmargin import (
    Axis,
    Definition,
    Observations,
    Threshold,
    fit_dose_response,
    threshold,
)

_SETTINGS = settings(
    max_examples=10,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
_LOG = Axis(name="severity", unit="unit", scale="log")
_LEVEL = 0.95
_RESAMPLES = 29
_SEED = 20261118

Counts = tuple[list[float], list[int], list[int]]


@st.composite
def _falling_counts(draw: st.DrawFn) -> Counts:
    """Successes out of trials, falling with severity across most of the unit range, with every
    level keeping a success and a failure."""
    n = draw(st.integers(min_value=5, max_value=6))
    severities = sorted(
        draw(
            st.lists(
                st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
    )
    trials = draw(st.lists(st.integers(min_value=80, max_value=200), min_size=n, max_size=n))
    top = draw(st.floats(min_value=0.80, max_value=0.95))
    bottom = draw(st.floats(min_value=0.05, max_value=0.20))
    step = (top - bottom) / (n - 1)
    successes = [min(max(int(round((top - i * step) * t)), 1), t - 1) for i, t in enumerate(trials)]
    return severities, successes, trials


def _solve(
    severity: list[float],
    successes: list[int],
    trials: list[int],
    *,
    direction: str = "decreasing",
) -> Threshold:
    obs = Observations.from_counts(
        _LOG,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="outcome",
        direction=direction,
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
    return threshold(
        fit,
        definition=Definition.absolute(0.5),
        interval_method="parametric_bootstrap",
        level=_LEVEL,
        n_resamples=_RESAMPLES,
        seed=_SEED,
    )


def _assert_same(a: Threshold, b: Threshold, *, rel: float, scale: float = 1.0) -> None:
    """``b``'s severities equal ``scale`` times ``a``'s; everything else is equal."""
    assert a.status is b.status and a.censoring is b.censoring
    assert a.interval_method == b.interval_method
    assert a.bootstrap == b.bootstrap
    for left, right in ((a.value, b.value), (a.lo, b.lo), (a.hi, b.hi)):
        if left is None or right is None:
            assert left is None and right is None
        else:
            assert right == pytest.approx(scale * left, rel=rel)


@_SETTINGS
@given(
    counts=_falling_counts(),
    scale=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
)
def test_rescaling_the_severities_rescales_the_bounds_under_one_seed(
    counts: Counts, scale: float
) -> None:
    severity, successes, trials = counts

    original = _solve(severity, successes, trials)
    rescaled = _solve([x * scale for x in severity], successes, trials)

    _assert_same(original, rescaled, rel=1e-6, scale=scale)


@_SETTINGS
@given(counts=_falling_counts(), order=st.data())
def test_row_order_changes_nothing(counts: Counts, order: st.DataObject) -> None:
    severity, successes, trials = counts
    permutation = order.draw(st.permutations(range(len(severity))))

    original = _solve(severity, successes, trials)
    shuffled = _solve(
        [severity[i] for i in permutation],
        [successes[i] for i in permutation],
        [trials[i] for i in permutation],
    )

    _assert_same(original, shuffled, rel=1e-12)


@_SETTINGS
@given(counts=_falling_counts())
def test_an_increasing_fit_equals_the_mirrored_decreasing_one_under_one_seed(
    counts: Counts,
) -> None:
    severity, successes, trials = counts
    events = [n - k for k, n in zip(successes, trials, strict=True)]

    decreasing = _solve(severity, successes, trials)
    increasing = _solve(severity, events, trials, direction="increasing")

    assert decreasing.direction == "decreasing" and increasing.direction == "increasing"
    _assert_same(decreasing, increasing, rel=1e-9)
