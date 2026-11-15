"""``threshold()`` on an isotonic fit: the point estimate, and what it refuses.

An isotonic threshold interpolates the pooled cell estimates, so its rules differ from a
parametric curve's:

* **Interpolation.** The value is the crossing of the piecewise-linear interpolant of
  ``cell_estimates`` over the tested severities, on the axis scale (``log x`` on a log axis, ``x``
  on a linear one). On a log axis the control has no coordinate and is not a node; on a linear
  axis it is one.
* **Generalised inverse.** The value is the smallest ``x`` with ``P(x) <= P*``, so a pooled flat
  block that equals the target gives the block's first severity.
* **No extrapolation.** A target the interpolant never reaches inside its nodes is
  ``UNREACHABLE`` with no number and a warning that names the side.
* **Exact claims for counts only.** Counts keep the exact one-sided tests first
  (``FAILS_AT_BASELINE``, ``RIGHT``, ``LEFT``); continuous isotonic makes none.
* **Definitions and methods.** ``relative`` raises; ``baseline_fraction`` and
  ``baseline_multiple`` need a control; the only interval method is ``cell_bootstrap``.

An ``OK`` result needs ``interval_method="cell_bootstrap"``, so these tests pass a small
``n_resamples`` and a fixed ``seed`` and assert the point value and the status only. The
interval is pinned elsewhere.
"""

from __future__ import annotations

import math

import pytest

from robustmargin import (
    Axis,
    Censoring,
    ContinuousObservations,
    Definition,
    Fit,
    Observations,
    Status,
    Threshold,
    fit_dose_response,
    threshold,
)

_LOG = Axis(name="severity", unit="unit", scale="log")
_LINEAR = Axis(name="severity", unit="unit", scale="linear")
_SEVERITY = [0.0, 1.0, 2.0, 4.0, 8.0]
_LEVEL = 0.95
_N_RESAMPLES = 49
_SEED = 20261114


def _counts_fit(
    successes: list[int],
    trials: list[int],
    *,
    axis: Axis = _LOG,
    direction: str = "decreasing",
    severity: list[float] | None = None,
) -> Fit:
    obs = Observations.from_counts(
        axis,
        severity=severity if severity is not None else _SEVERITY[: len(successes)],
        successes=successes,
        trials=trials,
        outcome="outcome",
        direction=direction,
    )
    fit = fit_dose_response(obs, model="isotonic")
    assert fit.status is Status.OK
    return fit


def _continuous_fit(
    mean: list[float],
    *,
    n: int = 4,
    axis: Axis = _LOG,
    direction: str = "decreasing",
    severity: list[float] | None = None,
) -> Fit:
    obs = ContinuousObservations.from_summaries(
        axis,
        severity=severity if severity is not None else _SEVERITY[: len(mean)],
        mean=mean,
        sd=[0.5] * len(mean),
        n=[n] * len(mean),
        outcome="response",
        direction=direction,
    )
    fit = fit_dose_response(obs, model="isotonic")
    assert fit.status is Status.OK
    return fit


def _solve(
    fit: Fit, definition: Definition, *, method: str = "cell_bootstrap", seed: int = _SEED
) -> Threshold:
    return threshold(
        fit,
        definition=definition,
        interval_method=method,
        level=_LEVEL,
        n_resamples=_N_RESAMPLES if method == "cell_bootstrap" else None,
        seed=seed if method == "cell_bootstrap" else None,
    )


def _no_number(t: Threshold) -> None:
    assert (t.value, t.lo, t.hi) == (None, None, None)


# Rates .95 .70 .30 .05 at severities 1, 2, 4, 8 (the control at 1.0): the curve crosses .5
# between severities 2 and 4, halfway between them in log x, at 2 * sqrt(2).
_CROSSING_SUCCESSES = [100, 95, 70, 30, 5]
_CROSSING_TRIALS = [100] * 5

# Continuous means 10 8 5 2 1 at severities 0 .5 1 2 4.
_MEANS = [10.0, 8.0, 5.0, 2.0, 1.0]
_MEAN_SEVERITY = [0.0, 0.5, 1.0, 2.0, 4.0]


class TestInterpolation:
    def test_a_counts_crossing_is_interpolated_linearly_in_log_severity(self) -> None:
        fit = _counts_fit(_CROSSING_SUCCESSES, _CROSSING_TRIALS)

        t = _solve(fit, Definition.absolute(0.5))

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.value == pytest.approx(2.0 * math.sqrt(2.0), rel=1e-9)

    def test_a_continuous_crossing_is_interpolated_in_log_severity(self) -> None:
        fit = _continuous_fit(_MEANS, severity=_MEAN_SEVERITY)

        t = _solve(fit, Definition.absolute(6.5))

        # Between (0.5, 8) and (1, 5): halfway down is halfway in log x.
        assert t.status is Status.OK
        assert t.value == pytest.approx(math.sqrt(0.5), rel=1e-9)

    def test_a_linear_axis_interpolates_in_severity_itself(self) -> None:
        fit = _continuous_fit(
            [10.0, 8.0, 5.0, 2.0, 1.0], axis=_LINEAR, severity=[0.0, 1.0, 2.0, 4.0, 8.0]
        )

        t = _solve(fit, Definition.absolute(6.5))

        # Between (1, 8) and (2, 5): the target is halfway, so x = 1.5.
        assert t.status is Status.OK
        assert t.value == pytest.approx(1.5, rel=1e-9)

    def test_on_a_linear_axis_the_control_is_an_interpolation_node(self) -> None:
        fit = _counts_fit([10, 7, 4, 2, 1], [10] * 5, axis=_LINEAR)

        t = _solve(fit, Definition.absolute(0.84))

        # Between the control (1.0 at x = 0) and (0.7 at x = 1): (1 - .84) / .3 of the way.
        assert t.status is Status.OK
        assert t.value == pytest.approx(0.16 / 0.30, rel=1e-9)

    def test_on_a_log_axis_the_control_is_not_a_node(self) -> None:
        fit = _counts_fit([10, 7, 4, 2, 1], [10] * 5)

        t = _solve(fit, Definition.absolute(0.84))

        # Same data and target: the curve is already below .84 at the first tested severity.
        assert t.status is Status.UNREACHABLE
        _no_number(t)

    def test_the_value_does_not_depend_on_the_bootstrap_seed(self) -> None:
        fit = _counts_fit(_CROSSING_SUCCESSES, _CROSSING_TRIALS)

        a = _solve(fit, Definition.absolute(0.5), seed=1)
        b = _solve(fit, Definition.absolute(0.5), seed=2)

        assert a.value == b.value

    def test_a_pooled_flat_block_at_the_target_gives_the_block_start(self) -> None:
        # Rates 1.0 .9 .4 .6 .2: the middle pair pools to .5, a flat block at severities 2 and 4.
        fit = _counts_fit([10, 9, 4, 6, 2], [10] * 5)
        assert fit.cell_estimates == pytest.approx([1.0, 0.9, 0.5, 0.5, 0.2])

        t = _solve(fit, Definition.absolute(0.5))

        # The smallest x with P(x) <= .5 is the start of the block, not its end.
        assert t.status is Status.OK
        assert t.value == pytest.approx(2.0, rel=1e-9)

    def test_a_target_in_the_interior_of_a_flat_block_is_not_reached_below_it(self) -> None:
        fit = _counts_fit([10, 9, 4, 6, 2], [10] * 5)

        t = _solve(fit, Definition.absolute(0.45))

        # The block sits at .5 > .45, so the crossing is between severities 4 and 8.
        assert t.status is Status.OK
        assert t.value is not None and 4.0 < t.value < 8.0

    def test_a_target_equal_to_a_node_estimate_returns_that_node(self) -> None:
        fit = _continuous_fit(_MEANS, severity=_MEAN_SEVERITY)

        t = _solve(fit, Definition.absolute(5.0))

        assert t.value == pytest.approx(1.0, rel=1e-9)


class TestUnreachable:
    def test_a_target_the_curve_never_falls_to_is_unreachable_above_the_highest(self) -> None:
        # Estimates stay at .8 or above; the exact test cannot call it RIGHT with 10 trials.
        fit = _counts_fit([10, 10, 9, 9, 8], [10] * 5)

        t = _solve(fit, Definition.absolute(0.5))

        assert t.status is Status.UNREACHABLE
        _no_number(t)
        assert t.interval_method == "exact_bound"
        assert t.censoring is Censoring.NONE
        assert t.bootstrap is None
        text = " ".join(t.warnings)
        assert "above the highest tested severity" in text
        assert "does not extrapolate" in text

    def test_a_log_axis_curve_already_below_the_target_is_unreachable_below_the_lowest(
        self,
    ) -> None:
        fit = _counts_fit([10, 7, 4, 2, 1], [10] * 5)

        t = _solve(fit, Definition.absolute(0.84))

        assert t.status is Status.UNREACHABLE
        _no_number(t)
        assert t.interval_method == "exact_bound"
        text = " ".join(t.warnings)
        assert "below the lowest tested severity" in text
        assert "does not extrapolate" in text
        assert "above the highest" not in text

    def test_continuous_unreachable_carries_no_number_and_no_bootstrap(self) -> None:
        fit = _continuous_fit([10.0, 9.0, 8.0, 7.0, 6.0])

        t = _solve(fit, Definition.absolute(0.5))

        assert t.status is Status.UNREACHABLE and t.censoring is Censoring.NONE
        _no_number(t)
        assert t.bootstrap is None

    @pytest.mark.parametrize("target", [0.0, 1.0])
    def test_a_counts_target_at_or_beyond_the_rate_range_is_unreachable(
        self, target: float
    ) -> None:
        fit = _counts_fit(_CROSSING_SUCCESSES, _CROSSING_TRIALS)

        t = _solve(fit, Definition.absolute(target))

        assert t.status is Status.UNREACHABLE
        _no_number(t)

    def test_an_unreachable_isotonic_threshold_is_not_a_failed_fit(self) -> None:
        fit = _counts_fit([10, 10, 9, 9, 8], [10] * 5)

        t = _solve(fit, Definition.absolute(0.5))

        assert t.fit.status is Status.OK and t.status is not Status.NOT_CONVERGED


class TestCountsExactClaimsStillTakePrecedence:
    def test_right_censoring_keeps_its_exact_lower_bound_and_no_value(self) -> None:
        fit = _counts_fit([200, 200, 199, 199, 198], [200] * 5)

        t = _solve(fit, Definition.absolute(0.5))

        assert t.status is Status.OK and t.censoring is Censoring.RIGHT
        assert t.value is None and t.hi is None
        assert t.lo == 8.0
        assert t.interval_method == "exact_bound"
        assert t.bootstrap is None

    def test_left_censoring_keeps_its_exact_upper_bound_and_no_value(self) -> None:
        fit = _counts_fit([200, 20, 10, 5, 2], [200] * 5)

        t = _solve(fit, Definition.absolute(0.5))

        assert t.status is Status.OK and t.censoring is Censoring.LEFT
        assert t.value is None and t.lo is None
        assert t.hi == 1.0
        assert t.interval_method == "exact_bound"
        assert t.bootstrap is None

    def test_a_control_below_the_gate_fails_at_baseline(self) -> None:
        fit = _counts_fit([100, 80, 50, 20, 5], [200] * 5)

        t = _solve(fit, Definition.absolute(0.95))

        assert t.status is Status.FAILS_AT_BASELINE
        _no_number(t)
        assert t.bootstrap is None

    def test_a_right_censored_result_never_carries_x_max_as_a_value(self) -> None:
        fit = _counts_fit([200, 200, 199, 199, 198], [200] * 5)

        t = _solve(fit, Definition.absolute(0.5))

        assert t.value is None


class TestContinuousMakesNoExactClaims:
    @pytest.mark.parametrize(
        "means",
        [
            [10.0, 9.0, 8.0, 7.0, 6.0],
            [10.0, 8.0, 5.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ],
    )
    @pytest.mark.parametrize("target", [0.5, 5.0, 20.0])
    def test_no_target_gives_right_left_or_fails_at_baseline(
        self, means: list[float], target: float
    ) -> None:
        fit = _continuous_fit(means)

        t = _solve(fit, Definition.absolute(target))

        assert t.status in (Status.OK, Status.UNREACHABLE)
        assert t.censoring in (Censoring.NONE, Censoring.OPEN_UPPER, Censoring.OPEN_LOWER)
        assert t.status is not Status.FAILS_AT_BASELINE
        assert t.censoring not in (Censoring.RIGHT, Censoring.LEFT)

    def test_a_target_above_the_whole_curve_is_unreachable_not_left(self) -> None:
        fit = _continuous_fit([10.0, 9.0, 8.0, 7.0, 6.0])

        t = _solve(fit, Definition.absolute(20.0))

        assert t.status is Status.UNREACHABLE and t.censoring is Censoring.NONE

    def test_a_flat_series_has_no_crossing(self) -> None:
        fit = _continuous_fit([5.0] * 5)

        t = _solve(fit, Definition.absolute(3.0))

        assert t.status is Status.UNREACHABLE


class TestDefinitions:
    def test_relative_raises_value_error(self) -> None:
        fit = _counts_fit(_CROSSING_SUCCESSES, _CROSSING_TRIALS)

        with pytest.raises(ValueError):
            _solve(fit, Definition.relative(0.5))

    def test_relative_raises_value_error_for_continuous_too(self) -> None:
        fit = _continuous_fit(_MEANS, severity=_MEAN_SEVERITY)

        with pytest.raises(ValueError):
            _solve(fit, Definition.relative(0.5))

    def test_baseline_fraction_without_a_control_raises(self) -> None:
        fit = _counts_fit([95, 70, 30, 5], [100] * 4, severity=[1.0, 2.0, 4.0, 8.0])

        with pytest.raises(ValueError):
            _solve(fit, Definition.baseline_fraction(0.5))

    def test_baseline_multiple_without_a_control_raises(self) -> None:
        fit = _counts_fit(
            [5, 30, 70, 95], [100] * 4, severity=[1.0, 2.0, 4.0, 8.0], direction="increasing"
        )

        with pytest.raises(ValueError):
            _solve(fit, Definition.baseline_multiple(2.0))

    def test_baseline_fraction_uses_the_control_estimate_as_the_baseline(self) -> None:
        fit = _continuous_fit(_MEANS, severity=_MEAN_SEVERITY)

        by_fraction = _solve(fit, Definition.baseline_fraction(0.65))
        by_target = _solve(fit, Definition.absolute(6.5))

        assert by_fraction.status is Status.OK
        assert by_fraction.value == pytest.approx(by_target.value, rel=1e-12)

    def test_the_baseline_is_the_pooled_control_estimate_not_the_raw_control_mean(self) -> None:
        # Means 8 10 5 2 1: the control (8) is below the next cell (10), so the pair pools to 9.
        fit = _continuous_fit([8.0, 10.0, 5.0, 2.0, 1.0], severity=_MEAN_SEVERITY)
        assert fit.cell_estimates is not None and fit.cell_estimates[0] == pytest.approx(9.0)

        by_fraction = _solve(fit, Definition.baseline_fraction(0.5))
        by_target = _solve(fit, Definition.absolute(4.5))

        assert by_fraction.value == pytest.approx(by_target.value, rel=1e-12)

    def test_baseline_multiple_on_an_increasing_series_uses_the_control_estimate(self) -> None:
        fit = _continuous_fit(
            [1.0, 2.0, 5.0, 8.0, 10.0], direction="increasing", severity=_MEAN_SEVERITY
        )

        t = _solve(fit, Definition.baseline_multiple(6.5))

        # Target 6.5 * 1.0, between (1.0, 5) and (2.0, 8): halfway in log x.
        assert t.status is Status.OK
        assert t.value == pytest.approx(math.sqrt(2.0), rel=1e-9)


class TestMethodsRefused:
    @pytest.mark.parametrize("method", ["profile", "delta", "parametric_bootstrap"])
    @pytest.mark.parametrize("which", ["counts", "continuous"])
    def test_a_parametric_interval_method_raises_value_error(self, method: str, which: str) -> None:
        fit = (
            _counts_fit(_CROSSING_SUCCESSES, _CROSSING_TRIALS)
            if which == "counts"
            else _continuous_fit(_MEANS, severity=_MEAN_SEVERITY)
        )
        target = 0.5 if which == "counts" else 6.5

        with pytest.raises(ValueError):
            threshold(
                fit, definition=Definition.absolute(target), interval_method=method, level=_LEVEL
            )

    def test_cell_bootstrap_is_accepted(self) -> None:
        fit = _counts_fit(_CROSSING_SUCCESSES, _CROSSING_TRIALS)

        t = _solve(fit, Definition.absolute(0.5))

        assert t.status is Status.OK
        assert t.interval_method == "cell_bootstrap"
        assert t.bootstrap is not None
        assert (t.bootstrap.n_resamples, t.bootstrap.seed) == (_N_RESAMPLES, _SEED)


class TestIncreasingIsTheMirroredDecreasing:
    def test_an_increasing_counts_threshold_equals_the_complementary_decreasing_one(self) -> None:
        successes = [2, 10, 30, 70, 95]
        trials = [100] * 5
        increasing = _counts_fit(successes, trials, direction="increasing")
        complement = _counts_fit([n - k for n, k in zip(trials, successes, strict=True)], trials)

        a = _solve(increasing, Definition.absolute(0.5))
        b = _solve(complement, Definition.absolute(0.5))

        assert a.status is Status.OK and b.status is Status.OK
        assert a.direction == "increasing" and b.direction == "decreasing"
        assert a.value == pytest.approx(b.value, rel=1e-12)

    def test_an_increasing_target_is_complemented_on_the_mirrored_series(self) -> None:
        successes = [2, 10, 30, 70, 95]
        trials = [100] * 5
        increasing = _counts_fit(successes, trials, direction="increasing")
        complement = _counts_fit([n - k for n, k in zip(trials, successes, strict=True)], trials)

        a = _solve(increasing, Definition.absolute(0.2))
        b = _solve(complement, Definition.absolute(0.8))

        assert a.value is not None
        assert a.value == pytest.approx(b.value, rel=1e-12)

    def test_an_increasing_continuous_threshold_equals_the_negated_decreasing_one(self) -> None:
        rising = [1.0, 2.0, 5.0, 8.0, 10.0]
        increasing = _continuous_fit(rising, direction="increasing", severity=_MEAN_SEVERITY)
        negated = _continuous_fit([-v for v in rising], severity=_MEAN_SEVERITY)

        a = _solve(increasing, Definition.absolute(6.5))
        b = _solve(negated, Definition.absolute(-6.5))

        assert a.status is Status.OK and b.status is Status.OK
        assert a.value == pytest.approx(b.value, rel=1e-12)

    def test_the_increasing_unreachable_side_is_reported_on_the_counted_outcome(self) -> None:
        increasing = _counts_fit([0, 0, 1, 2, 3], [10] * 5, direction="increasing")

        t = _solve(increasing, Definition.absolute(0.5))

        assert t.status is Status.UNREACHABLE
        _no_number(t)
