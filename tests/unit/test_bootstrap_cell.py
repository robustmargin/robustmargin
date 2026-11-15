"""``interval_method="cell_bootstrap"``: resample each severity level from its own data.

Counts: ``k* ~ Binomial(n_i, k_i / n_i)``. Continuous cells keep only ``n``, ``mean`` and ``sd``, so
a cell mean is drawn from ``N(mean_i, sd_i^2 / n_i)`` (the normal approximation to resampling
responses with replacement), and a cell of one observation has no spread to resample and raises.

It is the only interval an isotonic fit accepts, and a binomial fit accepts it too. This file pins:

* the interval on a binomial fit and on isotonic counts and continuous fits (bracketing the value,
  reproducible under a seed, an integer seed recorded when none is given);
* the isotonic point estimate the bootstrap brackets: the crossing of the piecewise-linear
  interpolant of the isotonic cell values over the tested severities, on the axis scale, computed
  here by an oracle that uses ``scipy.optimize.isotonic_regression`` directly;
* the isotonic definitions: ``absolute``, and ``baseline_fraction`` read from the control cell's
  isotonic value; ``relative`` raises, and so does a baseline definition without a control;
* an isotonic curve that does not cross the target between the interpolation nodes is
  ``UNREACHABLE`` with no number, and a bootstrap replicate in that position is counted by the
  side it falls on, a side opening when more than ``alpha / 2`` of the replicates land on it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import isotonic_regression
from scipy.stats import norm

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
_LEVEL = 0.95
_LEVELS = [0.0, *np.logspace(-1.0, 1.0, 10).tolist()]
_MU, _S = math.log(2.5), 0.5


def _rich_counts(upper: float = 1.0, seed: int = 20261114) -> tuple[list[int], list[int]]:
    x = np.asarray(_LEVELS)
    with np.errstate(divide="ignore"):
        z = (np.log(x) - _MU) / _S
    p = np.where(x == 0.0, upper, upper * (1.0 - norm.cdf(z)))
    return np.random.default_rng(seed).binomial(400, p).tolist(), [400] * x.size


def _counts_obs(
    severity: list[float], successes: list[int], trials: list[int], axis: Axis = _LOG
) -> Observations:
    return Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )


def _isotonic_counts(severity: list[float], successes: list[int], trials: list[int]) -> Fit:
    fit = fit_dose_response(_counts_obs(severity, successes, trials), model="isotonic")
    assert fit.status is Status.OK
    return fit


def _rich_isotonic() -> Fit:
    successes, trials = _rich_counts(upper=0.95)
    return _isotonic_counts(_LEVELS, successes, trials)


def _rich_binomial() -> Fit:
    successes, trials = _rich_counts()
    fit = fit_dose_response(
        _counts_obs(_LEVELS, successes, trials),
        model="binomial",
        link="probit",
        upper=1.0,
        lower=0.0,
    )
    assert fit.status is Status.OK
    return fit


def _continuous_obs(*, reps: int = 8, n_one: bool = False) -> ContinuousObservations:
    rng = np.random.default_rng(20261117)
    levels = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
    severity = [x for x in levels for _ in range(reps)]
    x = np.asarray(severity)
    with np.errstate(divide="ignore"):
        z = (np.log(x) - math.log(0.8)) / 0.5
    share = np.where(x == 0.0, 0.0, 1.0 / (1.0 + np.exp(-z)))
    value = 10.0 - 9.0 * share + rng.normal(0.0, 0.3, size=x.size)
    obs = ContinuousObservations.from_observations(
        _LOG, severity=severity, value=value.tolist(), outcome="response", direction="decreasing"
    )
    if not n_one:
        return obs
    cells = obs.cells()
    return ContinuousObservations.from_summaries(
        _LOG,
        severity=[c.severity for c in cells],
        mean=[c.mean for c in cells],
        sd=[0.0 if i == 3 else c.sd for i, c in enumerate(cells)],
        n=[1 if i == 3 else c.n for i, c in enumerate(cells)],
        outcome="response",
        direction="decreasing",
    )


def _boot(
    fit: Fit,
    definition: Definition,
    method: str = "cell_bootstrap",
    *,
    n_resamples: int | None = 199,
    seed: int | None = 5,
) -> Threshold:
    return threshold(
        fit,
        definition=definition,
        interval_method=method,
        level=_LEVEL,
        n_resamples=n_resamples,
        seed=seed,
    )


def _crossing(
    severity: list[float], rate: list[float], weight: list[float], target: float
) -> float | str:
    """Oracle for the isotonic point estimate on a log axis: PAVA on the cells in order (control
    first), then the smallest ``log x`` where the interpolant over the nonzero levels reaches
    ``target``. ``"low"`` / ``"high"`` when it does not cross within the nodes."""
    fitted = isotonic_regression(np.asarray(rate), weights=np.asarray(weight), increasing=False).x
    keep = [i for i, x in enumerate(severity) if x > 0.0]
    lx = [math.log(severity[i]) for i in keep]
    p = [float(fitted[i]) for i in keep]
    if p[0] <= target:
        return "low"
    for j in range(1, len(lx)):
        if p[j] <= target:
            frac = (p[j - 1] - target) / (p[j - 1] - p[j])
            return math.exp(lx[j - 1] + frac * (lx[j] - lx[j - 1]))
    return "high"


# ---------------------------------------------------------------------------------------------
# Binomial fit
# ---------------------------------------------------------------------------------------------


class TestCellBootstrapOnABinomialFit:
    def test_the_interval_brackets_the_value_and_records_the_summary(self) -> None:
        t = _boot(_rich_binomial(), Definition.absolute(0.5))

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.interval_method == "cell_bootstrap"
        assert t.lo is not None and t.value is not None and t.hi is not None
        assert t.lo <= t.value <= t.hi
        assert t.bootstrap is not None
        assert (t.bootstrap.n_resamples, t.bootstrap.seed, t.bootstrap.failed) == (199, 5, 0)

    def test_the_point_estimate_is_the_one_the_model_intervals_report(self) -> None:
        fit = _rich_binomial()
        definition = Definition.absolute(0.5)

        delta = threshold(fit, definition=definition, interval_method="delta", level=_LEVEL)

        assert _boot(fit, definition).value == delta.value

    def test_the_same_seed_gives_a_bit_identical_threshold(self) -> None:
        fit = _rich_binomial()

        assert _boot(fit, Definition.absolute(0.5)) == _boot(fit, Definition.absolute(0.5))

    def test_no_seed_records_an_int_and_rerunning_with_it_reproduces_the_result(self) -> None:
        fit = _rich_binomial()

        unseeded = _boot(fit, Definition.absolute(0.5), seed=None)

        assert unseeded.bootstrap is not None and isinstance(unseeded.bootstrap.seed, int)
        replay = _boot(fit, Definition.absolute(0.5), seed=unseeded.bootstrap.seed)
        assert replay == unseeded

    def test_it_draws_differently_from_the_parametric_bootstrap_but_agrees_in_width(self) -> None:
        fit = _rich_binomial()
        definition = Definition.absolute(0.5)

        cell = _boot(fit, definition, "cell_bootstrap")
        parametric = _boot(fit, definition, "parametric_bootstrap")

        assert cell.lo is not None and cell.hi is not None
        assert parametric.lo is not None and parametric.hi is not None
        assert (cell.lo, cell.hi) != (parametric.lo, parametric.hi)
        # Both estimate one standard error; with 199 replicates each, a factor of 1.5 is generous.
        ratio = math.log(cell.hi / cell.lo) / math.log(parametric.hi / parametric.lo)
        assert 1.0 / 1.5 < ratio < 1.5


# ---------------------------------------------------------------------------------------------
# Isotonic counts
# ---------------------------------------------------------------------------------------------


class TestIsotonicCounts:
    def test_the_value_is_the_crossing_of_the_interpolated_isotonic_cell_values(self) -> None:
        successes, trials = _rich_counts(upper=0.95)
        fit = _isotonic_counts(_LEVELS, successes, trials)
        rate = [k / n for k, n in zip(successes, trials, strict=True)]

        t = _boot(fit, Definition.absolute(0.5))

        assert t.status is Status.OK
        assert t.value == pytest.approx(_crossing(_LEVELS, rate, trials, 0.5), rel=1e-9)

    def test_the_interval_brackets_the_value(self) -> None:
        t = _boot(_rich_isotonic(), Definition.absolute(0.5), n_resamples=499)

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.interval_method == "cell_bootstrap"
        assert t.lo is not None and t.value is not None and t.hi is not None
        assert 0.0 < t.lo <= t.value <= t.hi
        assert t.bootstrap is not None
        assert t.bootstrap.failed == 0
        assert t.bootstrap.unreachable_low == 0 and t.bootstrap.unreachable_high == 0

    def test_the_interval_is_in_severity_units_and_sensible_in_width(self) -> None:
        t = _boot(_rich_isotonic(), Definition.absolute(0.5), n_resamples=499)

        assert t.lo is not None and t.value is not None and t.hi is not None
        assert 0.6 * t.value < t.lo and t.hi < 1.6 * t.value

    def test_the_same_seed_gives_a_bit_identical_threshold(self) -> None:
        fit = _rich_isotonic()

        assert _boot(fit, Definition.absolute(0.5)) == _boot(fit, Definition.absolute(0.5))

    def test_a_different_seed_gives_a_different_interval(self) -> None:
        fit = _rich_isotonic()

        a = _boot(fit, Definition.absolute(0.5), seed=1)
        b = _boot(fit, Definition.absolute(0.5), seed=2)

        assert a.value == b.value and (a.lo, a.hi) != (b.lo, b.hi)

    def test_no_seed_records_an_int_and_rerunning_with_it_reproduces_the_result(self) -> None:
        fit = _rich_isotonic()

        unseeded = _boot(fit, Definition.absolute(0.5), seed=None)

        assert unseeded.bootstrap is not None
        seed = unseeded.bootstrap.seed
        assert isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
        assert _boot(fit, Definition.absolute(0.5), seed=seed) == unseeded

    def test_the_default_is_999_resamples(self) -> None:
        t = _boot(_rich_isotonic(), Definition.absolute(0.5), n_resamples=None)

        assert t.bootstrap is not None and t.bootstrap.n_resamples == 999

    def test_baseline_fraction_is_a_fraction_of_the_control_cells_isotonic_value(self) -> None:
        successes, trials = _rich_counts(upper=0.95)
        fit = _isotonic_counts(_LEVELS, successes, trials)
        assert fit.cell_estimates is not None
        rate = [k / n for k, n in zip(successes, trials, strict=True)]
        target = 0.8 * fit.cell_estimates[0]

        t = _boot(fit, Definition.baseline_fraction(0.8))

        assert t.status is Status.OK
        assert t.value == pytest.approx(_crossing(_LEVELS, rate, trials, target), rel=1e-9)

    def test_relative_raises_because_an_isotonic_fit_has_no_lower_asymptote(self) -> None:
        with pytest.raises(ValueError):
            _boot(_rich_isotonic(), Definition.relative(0.5))

    def test_a_baseline_definition_without_a_control_raises(self) -> None:
        nonzero = _LEVELS[1:]
        successes, trials = _rich_counts(upper=0.95)
        fit = _isotonic_counts(nonzero, successes[1:], trials[1:])

        with pytest.raises(ValueError):
            _boot(fit, Definition.baseline_fraction(0.8))

    def test_the_unreachable_target_of_a_curve_that_stays_above_it_has_no_number(self) -> None:
        # 45/200 at the highest level: above 0.2, but not significantly, so no exact claim.
        fit = _isotonic_counts(
            [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
            [200, 196, 190, 170, 120, 70, 45],
            [200] * 7,
        )

        t = _boot(fit, Definition.absolute(0.2))

        assert t.status is Status.UNREACHABLE
        assert (t.value, t.lo, t.hi) == (None, None, None)
        assert t.interval_method == "exact_bound" and t.bootstrap is None
        assert any("above the highest tested severity" in w for w in t.warnings)
        assert any("extrapolate" in w for w in t.warnings)

    def test_a_curve_already_below_the_target_at_the_first_level_has_no_number(self) -> None:
        fit = _isotonic_counts(
            [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
            [200, 36, 30, 20, 10, 5, 2],
            [200] * 7,
        )

        t = _boot(fit, Definition.absolute(0.2))

        assert t.status is Status.UNREACHABLE and t.value is None
        assert t.bootstrap is None
        assert any("below the lowest tested severity" in w for w in t.warnings)


class TestReplicatesOutsideTheNodesCountBySide:
    def test_replicates_that_never_reach_the_target_open_the_upper_side(self) -> None:
        # The highest level sits just under the target (0.19 against 0.2), so about a third of the
        # replicates stay above it everywhere: no crossing inside the nodes, on the high side.
        successes = [200, 196, 190, 170, 120, 70, 38]
        fit = _isotonic_counts([0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2], successes, [200] * 7)

        t = _boot(fit, Definition.absolute(0.2), n_resamples=999)

        assert t.status is Status.OK and t.censoring is Censoring.OPEN_UPPER
        assert t.hi is None and t.lo is not None and t.value is not None
        assert t.lo <= t.value
        assert t.interval_method == "cell_bootstrap"
        assert t.bootstrap is not None
        assert t.bootstrap.unreachable_high > 0.025 * 999
        assert t.bootstrap.unreachable_low == 0

    def test_replicates_already_below_the_target_at_the_first_level_open_the_lower_side(
        self,
    ) -> None:
        # 0.915 at the first level against a target of 0.9: about a quarter of the replicates are
        # at or below it there, with no crossing between the control and the first level.
        successes = [200, 183, 165, 140, 100, 60, 20]
        fit = _isotonic_counts([0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2], successes, [200] * 7)

        t = _boot(fit, Definition.absolute(0.9), n_resamples=999)

        assert t.status is Status.OK and t.censoring is Censoring.OPEN_LOWER
        assert t.lo is None and t.hi is not None and t.value is not None
        assert t.value <= t.hi
        assert t.bootstrap is not None
        assert t.bootstrap.unreachable_low > 0.025 * 999
        assert t.bootstrap.unreachable_high == 0


# ---------------------------------------------------------------------------------------------
# Isotonic continuous
# ---------------------------------------------------------------------------------------------


class TestIsotonicContinuous:
    def _fit(self, **kwargs: bool) -> Fit:
        fit = fit_dose_response(_continuous_obs(**kwargs), model="isotonic")
        assert fit.status is Status.OK
        return fit

    def test_the_value_is_the_crossing_of_the_interpolated_isotonic_cell_means(self) -> None:
        obs = _continuous_obs()
        cells = obs.cells()

        t = _boot(self._fit(), Definition.absolute(5.5))

        expected = _crossing(
            [c.severity for c in cells], [c.mean for c in cells], [float(c.n) for c in cells], 5.5
        )
        assert t.status is Status.OK
        assert t.value == pytest.approx(expected, rel=1e-9)

    def test_the_interval_brackets_the_value_and_makes_no_exact_claim(self) -> None:
        t = _boot(self._fit(), Definition.absolute(5.5), n_resamples=499)

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.interval_method == "cell_bootstrap"
        assert t.lo is not None and t.value is not None and t.hi is not None
        assert 0.0 < t.lo <= t.value <= t.hi
        assert t.baseline is None and t.grid is None

    def test_the_same_seed_gives_a_bit_identical_threshold(self) -> None:
        fit = self._fit()

        assert _boot(fit, Definition.absolute(5.5)) == _boot(fit, Definition.absolute(5.5))

    def test_a_cell_of_one_observation_raises(self) -> None:
        fit = self._fit(n_one=True)

        with pytest.raises(ValueError, match="observation|n=1|n == 1|n_i"):
            _boot(fit, Definition.absolute(5.5))

    def test_a_continuous_isotonic_fit_never_gives_an_exact_censoring_label(self) -> None:
        for target in (5.5, 9.9, 0.5, 100.0):
            t = _boot(self._fit(), Definition.absolute(target), n_resamples=99)

            assert t.status in (Status.OK, Status.UNREACHABLE)
            assert t.censoring in (Censoring.NONE, Censoring.OPEN_UPPER, Censoring.OPEN_LOWER)
            assert t.baseline is None and t.grid is None
