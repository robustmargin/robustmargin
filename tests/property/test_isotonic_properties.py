"""Properties of the isotonic fit and its point-estimate threshold, checked with hypothesis.

Each property is stated on the public API only.

* **Unit scaling.** Rescaling every nonzero severity by ``c > 0`` scales the value, and any exact
  bound, by exactly ``c`` (to float rounding), and changes no status or censoring. The control
  stays at 0.
* **Row permutation changes nothing.** Rows at one severity are pooled, so their order is
  irrelevant, and the cell estimates are bit-identical.
* **Aggregation does not matter.** A cell given as ``(successes, trials)`` and the same cell given
  as one 0/1 row per trial are one series: same estimates, same value.
* **Direction flip.** An increasing outcome is the complementary decreasing outcome
  (``trials - successes``) with the target complemented, and gives the same value.
* **Monotone fit, monotone in the criterion.** ``cell_estimates`` never reverses direction, and a
  lower target never moves the threshold of a decreasing outcome inward.

The bootstrap runs a few resamples with a fixed seed: only the point value and the status are
compared, which do not depend on the draws. Every strategy is derandomised.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from robustmargin import (
    Axis,
    Definition,
    Fit,
    Observations,
    Status,
    Threshold,
    fit_dose_response,
    threshold,
)

_SEVERITY = [0.0, 1.0, 2.0, 4.0, 8.0]
_TRIALS = 20
_PROFILE = settings(
    derandomize=True,
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_successes = st.lists(st.integers(0, _TRIALS), min_size=5, max_size=5)
_scale = st.sampled_from([1e-3, 0.37, 2.5, 1.0e3])
_targets = st.sampled_from([0.3, 0.5, 0.7])


def _axis(scale: str = "log") -> Axis:
    return Axis(name="severity", unit="unit", scale=scale)  # type: ignore[arg-type]


def _obs(
    successes: list[int],
    *,
    direction: str = "decreasing",
    factor: float = 1.0,
    scale: str = "log",
) -> Observations:
    return Observations.from_counts(
        _axis(scale),
        severity=[x * factor for x in _SEVERITY],
        successes=successes,
        trials=[_TRIALS] * 5,
        outcome="outcome",
        direction=direction,
    )


def _fit(obs: Observations) -> Fit:
    fit = fit_dose_response(obs, model="isotonic")
    assert fit.status is Status.OK
    return fit


def _solve(fit: Fit, target: float) -> Threshold:
    return threshold(
        fit,
        definition=Definition.absolute(target),
        interval_method="cell_bootstrap",
        level=0.95,
        n_resamples=9,
        seed=1,
    )


def _close(a: float | None, b: float | None, *, rel: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= rel * max(abs(a), abs(b))


@_PROFILE
@given(successes=_successes, factor=_scale, target=_targets)
def test_rescaling_severity_scales_the_value_and_bounds_by_the_same_factor(
    successes: list[int], factor: float, target: float
) -> None:
    base = _solve(_fit(_obs(successes)), target)
    scaled = _solve(_fit(_obs(successes, factor=factor)), target)

    assert scaled.status is base.status and scaled.censoring is base.censoring
    assert _close(scaled.value, None if base.value is None else base.value * factor)
    assert _close(scaled.lo, None if base.lo is None else base.lo * factor)
    assert _close(scaled.hi, None if base.hi is None else base.hi * factor)


@_PROFILE
@given(successes=_successes, factor=_scale)
def test_rescaling_severity_leaves_the_cell_estimates_unchanged(
    successes: list[int], factor: float
) -> None:
    assert _fit(_obs(successes, factor=factor)).cell_estimates == (
        _fit(_obs(successes)).cell_estimates
    )


@_PROFILE
@given(successes=_successes, target=_targets, data=st.data())
def test_row_order_changes_neither_the_estimates_nor_the_value(
    successes: list[int], target: float, data: st.DataObject
) -> None:
    rows = [(x, k // 2, _TRIALS // 2) for x, k in zip(_SEVERITY, successes, strict=True)] + [
        (x, k - k // 2, _TRIALS - _TRIALS // 2) for x, k in zip(_SEVERITY, successes, strict=True)
    ]
    shuffled = data.draw(st.permutations(rows))

    def build(order: list[tuple[float, int, int]]) -> Observations:
        return Observations.from_counts(
            _axis(),
            severity=[r[0] for r in order],
            successes=[r[1] for r in order],
            trials=[r[2] for r in order],
            outcome="outcome",
            direction="decreasing",
        )

    a, b = _fit(build(rows)), _fit(build(shuffled))

    assert a.cell_estimates == b.cell_estimates
    assert a.log_likelihood == b.log_likelihood
    ta, tb = _solve(a, target), _solve(b, target)
    assert (ta.status, ta.censoring, ta.value) == (tb.status, tb.censoring, tb.value)


@_PROFILE
@given(successes=_successes, target=_targets)
def test_aggregated_and_per_row_input_give_the_same_fit_and_value(
    successes: list[int], target: float
) -> None:
    aggregated = _obs(successes)
    severity: list[float] = []
    success: list[int] = []
    for x, k in zip(_SEVERITY, successes, strict=True):
        severity += [x] * _TRIALS
        success += [1] * k + [0] * (_TRIALS - k)
    per_row = Observations.from_observations(
        _axis(), severity=severity, success=success, outcome="outcome", direction="decreasing"
    )

    a, b = _fit(aggregated), _fit(per_row)

    assert a.cell_estimates == b.cell_estimates
    assert a.log_likelihood == b.log_likelihood
    ta, tb = _solve(a, target), _solve(b, target)
    assert (ta.status, ta.censoring, ta.value) == (tb.status, tb.censoring, tb.value)


@_PROFILE
@given(successes=_successes, target=_targets)
def test_an_increasing_outcome_equals_its_complementary_decreasing_outcome(
    successes: list[int], target: float
) -> None:
    increasing = _fit(_obs(successes, direction="increasing"))
    complement = _fit(_obs([_TRIALS - k for k in successes]))

    assert increasing.cell_estimates is not None and complement.cell_estimates is not None
    assert np.allclose(
        increasing.cell_estimates, [1.0 - p for p in complement.cell_estimates], rtol=0, atol=1e-12
    )
    a = _solve(increasing, target)
    b = _solve(complement, 1.0 - target)
    assert (a.status, a.censoring) == (b.status, b.censoring)
    assert _close(a.value, b.value, rel=1e-12)


@_PROFILE
@given(successes=_successes, direction=st.sampled_from(["decreasing", "increasing"]))
def test_cell_estimates_never_reverse_direction(successes: list[int], direction: str) -> None:
    fit = _fit(_obs(successes, direction=direction))

    assert fit.cell_estimates is not None
    steps = np.diff(fit.cell_estimates)
    assert np.all(steps <= 1e-15) if direction == "decreasing" else np.all(steps >= -1e-15)
    assert all(0.0 <= p <= 1.0 for p in fit.cell_estimates)


@_PROFILE
@given(successes=_successes)
def test_a_lower_target_never_moves_a_decreasing_threshold_inward(successes: list[int]) -> None:
    fit = _fit(_obs(successes))
    high, low = _solve(fit, 0.7), _solve(fit, 0.3)
    assume(high.value is not None and low.value is not None)

    assert high.value is not None and low.value is not None
    assert low.value >= high.value * (1.0 - 1e-12)


@_PROFILE
@given(successes=_successes, target=_targets)
def test_a_point_estimate_lies_within_the_tested_severities_and_a_censored_one_has_none(
    successes: list[int], target: float
) -> None:
    t = _solve(_fit(_obs(successes)), target)

    if t.status is not Status.OK or t.censoring.value in ("RIGHT", "LEFT"):
        assert t.value is None
    if t.value is not None:
        assert _SEVERITY[1] <= t.value <= _SEVERITY[-1]
