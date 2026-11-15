"""``interval_method="parametric_bootstrap"``: simulate from the fit, refit, take quantiles.

The bootstrap draws new data from the fitted binomial or Gaussian model at the observed design
(binomial ``k* ~ Binomial(n_i, P_hat(x_i))``; Gaussian cell means ``N(f_hat(x_i), sigma2/n_i)`` and
within-cell sums of squares ``sigma2 * chi2(n_i - 1)``), refits the same model, link and
fixed/estimated asymptotes, solves each replicate's threshold, and reports percentiles of the
replicate thresholds on the severity axis's own scale.

What this file pins:

* **Determinism.** One seed gives a bit-identical ``Threshold``. With no seed an integer is drawn
  and recorded, and re-running with that integer reproduces the result.
* **Shape.** ``lo <= value <= hi``, the point estimate is the same one the other interval methods
  report, and the bounds are in natural severity (never log severity).
* **Agreement with a model interval.** On a rich, well-identified design the bootstrap interval is
  close to the profile interval. The two are different estimators of one quantity, so they are
  compared loosely: each endpoint within 15% of the profile endpoint on the log scale, and the
  widths within a factor of 1.4. With 400 trials at each of ten levels the standard error of the
  log threshold is about 0.02, so a 15% band is far wider than any genuine difference, but a
  bootstrap that returned the point estimate, or a log-scale value, or an interval of the wrong
  width would still miss it.
* **Direction.** An increasing fit and its mirrored decreasing fit use identical draws.

Estimated-asymptote cases use few resamples: a generic refit is far slower than a GLM refit.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from scipy.stats import norm

from robustmargin import (
    Axis,
    Censoring,
    ContinuousObservations,
    Definition,
    Observations,
    Status,
    Threshold,
    fit_dose_response,
    threshold,
)

_LOG = Axis(name="severity", unit="unit", scale="log")
_LINEAR = Axis(name="severity", unit="unit", scale="linear")
_LEVELS = [0.0, *np.logspace(-1.0, 1.0, 10).tolist()]
_MU, _S = math.log(2.5), 0.5
_LEVEL = 0.95


def _counts(
    *, seed: int = 20261114, upper: float = 1.0, levels: list[float] | None = None
) -> tuple[list[float], list[int], list[int]]:
    """Counts from a probit on a log axis: lower asymptote 0, upper asymptote ``upper``."""
    x = np.asarray(_LEVELS if levels is None else levels)
    with np.errstate(divide="ignore"):
        z = (np.log(x) - _MU) / _S
    p = np.where(x == 0.0, upper, upper * (1.0 - norm.cdf(z)))
    trials = [400] * x.size
    return x.tolist(), np.random.default_rng(seed).binomial(400, p).tolist(), trials


def _binomial_fit(
    *, upper: object = 1.0, direction: str = "decreasing", axis: Axis = _LOG
) -> object:
    severity, successes, trials = _counts(upper=0.95 if upper == "estimate" else 1.0)
    if direction == "increasing":
        successes = [n - k for k, n in zip(successes, trials, strict=True)]
    obs = Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="outcome",
        direction=direction,
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper=upper, lower=0.0)  # type: ignore[arg-type]
    assert fit.status is Status.OK
    return fit


def _gaussian_fit() -> object:
    rng = np.random.default_rng(20261115)
    levels = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
    severity = [x for x in levels for _ in range(4)]
    x = np.asarray(severity)
    with np.errstate(divide="ignore"):
        z = (np.log(x) - math.log(0.8)) / 0.5
    share = np.where(x == 0.0, 0.0, 1.0 / (1.0 + np.exp(-z)))
    value = 10.0 - 9.0 * share + rng.normal(0.0, 0.3, size=x.size)
    obs = ContinuousObservations.from_observations(
        _LOG, severity=severity, value=value.tolist(), outcome="response", direction="decreasing"
    )
    fit = fit_dose_response(obs, model="gaussian", link="logit", upper=10.0, lower=1.0)
    assert fit.status is Status.OK
    return fit


# case name -> (fit builder, definition, n_resamples)
_CASES = {
    "binomial_fixed": (lambda: _binomial_fit(), Definition.absolute(0.5), 49),
    "binomial_upper_estimate": (
        lambda: _binomial_fit(upper="estimate"),
        Definition.absolute(0.5),
        11,
    ),
    "gaussian_fixed": (lambda: _gaussian_fit(), Definition.absolute(5.5), 49),
}


@functools.cache
def _fit_for(case: str):  # type: ignore[no-untyped-def]
    return _CASES[case][0]()


def _solve(case: str, seed: int | None, **overrides: object) -> Threshold:
    _, definition, n_resamples = _CASES[case]
    kwargs = {"n_resamples": n_resamples, "seed": seed, **overrides}
    return threshold(
        _fit_for(case),
        definition=definition,
        interval_method="parametric_bootstrap",
        level=_LEVEL,
        **kwargs,  # type: ignore[arg-type]
    )


@functools.cache
def _cached(case: str, seed: int) -> Threshold:
    return _solve(case, seed)


@pytest.fixture(params=sorted(_CASES))
def case(request: pytest.FixtureRequest) -> str:
    return str(request.param)


class TestDeterminism:
    def test_the_same_seed_gives_a_bit_identical_threshold(self, case: str) -> None:
        first = _cached(case, 7)

        again = _solve(case, 7)

        assert again == first
        assert (again.value, again.lo, again.hi) == (first.value, first.lo, first.hi)
        assert again.bootstrap == first.bootstrap

    def test_a_different_seed_gives_a_different_interval(self) -> None:
        a, b = _cached("binomial_fixed", 7), _cached("binomial_fixed", 8)

        assert a.value == b.value
        assert (a.lo, a.hi) != (b.lo, b.hi)

    def test_no_seed_records_an_int_seed_and_rerunning_with_it_reproduces_the_result(
        self, case: str
    ) -> None:
        unseeded = _solve(case, None)

        assert unseeded.bootstrap is not None
        seed = unseeded.bootstrap.seed
        assert isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
        assert _solve(case, seed) == unseeded

    def test_two_unseeded_runs_record_different_seeds(self) -> None:
        a, b = _solve("binomial_fixed", None), _solve("binomial_fixed", None)

        assert a.bootstrap is not None and b.bootstrap is not None
        assert a.bootstrap.seed != b.bootstrap.seed

    def test_the_recorded_seed_is_the_one_passed(self, case: str) -> None:
        t = _cached(case, 7)

        assert t.bootstrap is not None and t.bootstrap.seed == 7


class TestShape:
    def test_the_interval_brackets_the_value_with_the_requested_method_and_level(
        self, case: str
    ) -> None:
        t = _cached(case, 7)

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.interval_method == "parametric_bootstrap" and t.level == _LEVEL
        assert t.lo is not None and t.value is not None and t.hi is not None
        assert t.lo <= t.value <= t.hi
        assert t.lo < t.hi

    def test_the_summary_records_the_resamples_and_a_clean_run(self, case: str) -> None:
        t = _cached(case, 7)

        assert t.bootstrap is not None
        assert t.bootstrap.n_resamples == _CASES[case][2]
        assert t.bootstrap.failed == 0
        assert t.bootstrap.unreachable_low == 0 and t.bootstrap.unreachable_high == 0
        assert not any(w.startswith("BOOTSTRAP_FAILURES:") for w in t.warnings)

    def test_the_point_estimate_is_the_one_the_model_intervals_report(self, case: str) -> None:
        _, definition, _ = _CASES[case]

        delta = threshold(
            _fit_for(case), definition=definition, interval_method="delta", level=_LEVEL
        )

        assert _cached(case, 7).value == delta.value

    def test_the_result_is_solved_from_the_fit_it_was_given(self, case: str) -> None:
        t = _cached(case, 7)

        assert t.fit is _fit_for(case)
        assert t.definition == _CASES[case][1]

    def test_a_wider_level_gives_a_wider_interval_under_one_seed(self) -> None:
        fit = _fit_for("binomial_fixed")

        def at(level: float) -> Threshold:
            return threshold(
                fit,
                definition=Definition.absolute(0.5),
                interval_method="parametric_bootstrap",
                level=level,
                n_resamples=99,
                seed=3,
            )

        narrow, wide = at(0.80), at(0.99)

        assert narrow.lo is not None and narrow.hi is not None
        assert wide.lo is not None and wide.hi is not None
        assert wide.lo <= narrow.lo and narrow.hi <= wide.hi
        assert (wide.hi - wide.lo) > (narrow.hi - narrow.lo)


class TestNaturalSeverity:
    def test_the_bounds_are_severities_not_log_severities(self) -> None:
        t = _cached("binomial_fixed", 7)

        assert t.lo is not None and t.value is not None and t.hi is not None
        # The curve crosses 0.5 at 2.5. A log-scale bound would sit near log(2.5) = 0.92.
        assert 0.5 * t.value < t.lo < t.value < t.hi < 2.0 * t.value

    def test_a_linear_axis_gives_positive_bounds_in_severity_units(self) -> None:
        levels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        x = np.asarray(levels)
        p = 1.0 - norm.cdf((x - 2.5) / 1.0)
        successes = np.random.default_rng(20261116).binomial(100, p).tolist()
        obs = Observations.from_counts(
            _LINEAR,
            severity=levels,
            successes=successes,
            trials=[100] * len(levels),
            outcome="outcome",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        t = threshold(
            fit,
            definition=Definition.absolute(0.5),
            interval_method="parametric_bootstrap",
            level=_LEVEL,
            n_resamples=99,
            seed=4,
        )

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.lo is not None and t.value is not None and t.hi is not None
        assert 0.0 < t.lo <= t.value <= t.hi
        assert abs(t.value - 2.5) < 0.5
        assert t.hi - t.lo < 2.5


class TestAgreementWithTheProfileInterval:
    def test_on_a_rich_design_the_bootstrap_endpoints_are_close_to_the_profile_endpoints(
        self,
    ) -> None:
        fit = _fit_for("binomial_fixed")
        definition = Definition.absolute(0.5)

        boot = threshold(
            fit,
            definition=definition,
            interval_method="parametric_bootstrap",
            level=_LEVEL,
            n_resamples=299,
            seed=2026,
        )
        prof = threshold(fit, definition=definition, interval_method="profile", level=_LEVEL)

        assert boot.status is Status.OK and prof.status is Status.OK
        assert boot.lo is not None and boot.hi is not None
        assert prof.lo is not None and prof.hi is not None
        assert abs(math.log(boot.lo / prof.lo)) < 0.15
        assert abs(math.log(boot.hi / prof.hi)) < 0.15
        width_ratio = math.log(boot.hi / boot.lo) / math.log(prof.hi / prof.lo)
        assert 1.0 / 1.4 < width_ratio < 1.4


class TestDirection:
    def test_an_increasing_fit_equals_the_mirror_of_the_decreasing_one_under_one_seed(
        self,
    ) -> None:
        # Increasing events = failures of the decreasing fit; the asymptotes mirror to the same
        # fixed values (1, 0), and absolute(0.5) is its own complement.
        decreasing = _fit_for("binomial_fixed")
        increasing = _binomial_fit(direction="increasing")
        definition = Definition.absolute(0.5)

        a = threshold(
            decreasing,
            definition=definition,
            interval_method="parametric_bootstrap",
            level=_LEVEL,
            n_resamples=49,
            seed=9,
        )
        b = threshold(
            increasing,
            definition=definition,
            interval_method="parametric_bootstrap",
            level=_LEVEL,
            n_resamples=49,
            seed=9,
        )

        assert a.status is b.status is Status.OK
        assert a.direction == "decreasing" and b.direction == "increasing"
        for left, right in ((a.value, b.value), (a.lo, b.lo), (a.hi, b.hi)):
            assert left == pytest.approx(right, rel=1e-9)
        assert a.bootstrap == b.bootstrap
