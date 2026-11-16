"""Bootstrap replicate rules, the cell draws and the isotonic likelihood, as corrected in review.

* **Failed replicates are not dropped.** A replicate whose refit fails, or whose baseline is
  undefined (a replicate control estimate or fitted baseline at or below zero), enters the
  percentiles as ``-inf`` for ``lo`` and ``+inf`` for ``hi``. A side is open when its unreachable
  count plus the failed count exceeds ``alpha / 2`` of **all** ``n_resamples`` replicates.
* **Count draws are smoothed.** A cell of ``k`` out of ``n`` is resampled as
  ``Binomial(n, (k + 0.5) / (n + 1))``, for binomial and isotonic fits, so a cell at 0 or at 1 is
  not frozen. A bootstrap whose finite interval collapses to one point reports no bounds and the
  warning ``BOOTSTRAP_ZERO_WIDTH:``.
* **Continuous cell means** are drawn with variance ``sd_i**2 * (n_i - 1) / n_i**2``, the variance
  of a mean under resampling with replacement.
* **A log-axis flat block** equal to the target and starting at the first tested severity is
  ``UNREACHABLE`` (below the lowest tested severity); on a linear axis the control is a node and
  the crossing is at ``x_1``.
* **The isotonic binomial log-likelihood** includes ``log C(n, k)``, so it is comparable to the
  parametric binomial fits' value on the same counts, and the continuous one follows the Gaussian
  fit's convention.
* **Warnings** (excluded estimate with ``z0``, few resamples, the unreachable-side wording) say
  nothing false.

The per-replicate thetas are replaced where a rule needs an exact mixture; everything else runs on
real data.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
from scipy.special import gammaln
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
from robustmargin.intervals import _BootstrapDesign

_LOG = Axis(name="severity", unit="unit", scale="log")
_LINEAR = Axis(name="severity", unit="unit", scale="linear")
_LEVEL = 0.95
_LEVELS = [0.0, *np.logspace(-1.0, 1.0, 10).tolist()]


def _counts_obs(
    severity: list[float],
    successes: list[int],
    trials: list[int],
    *,
    axis: Axis = _LOG,
    direction: str = "decreasing",
) -> Observations:
    return Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="outcome",
        direction=direction,
    )


def _isotonic(
    severity: list[float],
    successes: list[int],
    trials: list[int],
    *,
    axis: Axis = _LOG,
    direction: str = "decreasing",
) -> Fit:
    fit = fit_dose_response(
        _counts_obs(severity, successes, trials, axis=axis, direction=direction),
        model="isotonic",
    )
    assert fit.status is Status.OK
    return fit


def _rich_counts(upper: float = 1.0) -> tuple[list[int], list[int]]:
    x = np.asarray(_LEVELS)
    with np.errstate(divide="ignore"):
        z = (np.log(x) - math.log(2.5)) / 0.5
    p = np.where(x == 0.0, upper, upper * (1.0 - norm.cdf(z)))
    return np.random.default_rng(20261114).binomial(400, p).tolist(), [400] * x.size


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


def _boot(
    fit: Fit,
    definition: Definition,
    method: str,
    *,
    n_resamples: int = 100,
    seed: int = 1,
    level: float = _LEVEL,
) -> Threshold:
    return threshold(
        fit,
        definition=definition,
        interval_method=method,
        level=level,
        n_resamples=n_resamples,
        seed=seed,
    )


def _point(fit: Fit, method: str) -> Threshold:
    """The point estimate of ``absolute(0.5)``, which every interval method reports unchanged."""
    if method == "cell_bootstrap":
        return _boot(fit, Definition.absolute(0.5), method, n_resamples=49)
    return threshold(fit, definition=Definition.absolute(0.5), interval_method=method, level=_LEVEL)


def _patch_thetas(monkeypatch: pytest.MonkeyPatch, thetas: list[float]) -> None:
    monkeypatch.setattr(
        _BootstrapDesign, "theta", lambda self, first, second, index: float(thetas[index])
    )


def _seam():  # type: ignore[no-untyped-def]
    # `robustmargin.threshold` the attribute is the function, so go through the module table.
    return importlib.import_module("robustmargin.threshold")._bootstrap_replicates


# ---------------------------------------------------------------------------------------------
# Failed replicates count as -inf for lo and +inf for hi
# ---------------------------------------------------------------------------------------------


class TestFailedReplicatesAreNotDropped:
    def test_failures_above_half_of_alpha_open_both_sides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fit = _rich_binomial()
        point = _point(fit, "delta")
        assert point.value is not None
        spread = np.linspace(-0.05, 0.05, 97)
        thetas = [math.nan] * 3 + (math.log(point.value) + spread).tolist()
        _patch_thetas(monkeypatch, thetas)

        t = _boot(fit, Definition.absolute(0.5), "parametric_bootstrap", n_resamples=100)

        # 3 of 100 failed: 3 > 0.025 * 100 on both sides, so neither end is bounded.
        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.value is not None and t.lo is None and t.hi is None
        assert t.bootstrap is not None and t.bootstrap.failed == 3
        assert any(w.startswith("BOOTSTRAP_FAILURES:") and "3/100" in w for w in t.warnings)

    def test_a_single_failure_in_a_hundred_leaves_the_interval_closed_and_unwarned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fit = _rich_binomial()
        point = _point(fit, "delta")
        assert point.value is not None
        spread = np.linspace(-0.05, 0.05, 99)
        _patch_thetas(monkeypatch, [math.nan, *(math.log(point.value) + spread).tolist()])

        t = _boot(fit, Definition.absolute(0.5), "parametric_bootstrap", n_resamples=100)

        assert t.lo is not None and t.hi is not None and t.lo <= t.value <= t.hi  # type: ignore[operator]
        assert t.bootstrap is not None and t.bootstrap.failed == 1
        assert not any(w.startswith("BOOTSTRAP_FAILURES:") for w in t.warnings)

    def test_failures_add_to_one_sided_unreachable_replicates_against_the_same_cut_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fit = _rich_binomial()
        point = _point(fit, "delta")
        assert point.value is not None
        spread = (math.log(point.value) + np.linspace(-0.05, 0.05, 94)).tolist()
        # 2 failed + 4 unreachable-low: the lower count is 6 > 2.5, the upper count is only 2.
        _patch_thetas(monkeypatch, [math.nan] * 2 + [-math.inf] * 4 + spread)

        t = _boot(fit, Definition.absolute(0.5), "parametric_bootstrap", n_resamples=100)

        assert t.bootstrap is not None
        assert (t.bootstrap.failed, t.bootstrap.unreachable_low) == (2, 4)
        assert t.lo is None  # the lower side is open

    def test_sparse_counts_where_a_few_percent_of_refits_separate_have_no_bounds(self) -> None:
        # About 6% of the refits separate: more than alpha / 2 of all replicates on each side.
        fit = fit_dose_response(
            _counts_obs([0.0, 0.5, 1.0, 2.0, 4.0], [6, 5, 3, 2, 0], [6] * 5),
            model="binomial",
            link="probit",
            upper=1.0,
            lower=0.0,
        )

        t = _boot(fit, Definition.absolute(0.5), "parametric_bootstrap", n_resamples=400, seed=2026)

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.value is not None and t.lo is None and t.hi is None
        assert t.bootstrap is not None
        assert 0.025 * 400 < t.bootstrap.failed <= 0.10 * 400
        assert any(w.startswith("BOOTSTRAP_FAILURES:") for w in t.warnings)

    def test_an_undefined_replicate_baseline_counts_as_failed_and_the_warning_names_it(
        self,
    ) -> None:
        # Control 2/150: about 8% of the draws put 0 events at the control, so the replicate's
        # baseline is zero and `baseline_multiple` has no target.
        fit = _isotonic(
            [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
            [2, 6, 25, 70, 120, 145],
            [150] * 6,
            direction="increasing",
        )

        t = _boot(fit, Definition.baseline_multiple(5), "cell_bootstrap", n_resamples=400, seed=6)

        assert t.status is Status.OK
        assert t.bootstrap is not None and 0.01 * 400 < t.bootstrap.failed <= 0.10 * 400
        assert t.lo is None  # failures widen the lower percentile to -inf and open the side
        messages = [w for w in t.warnings if w.startswith("BOOTSTRAP_FAILURES:")]
        assert len(messages) == 1
        assert "baseline" in messages[0].lower()
        assert f"{t.bootstrap.failed}/400" in messages[0]

    @pytest.mark.slow
    def test_estimated_lower_asymptote_replicates_no_longer_leave_a_closed_lower_bound(
        self,
    ) -> None:
        # The review's case: 3/150 events at the control, `lower` estimated, baseline_multiple(5).
        # Replicates whose fitted baseline is not positive failed, were dropped, and left an OK
        # interval with a lower bound near 0.835. They now count as -inf on the lower side.
        fit = fit_dose_response(
            _counts_obs(
                [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
                [3, 6, 25, 70, 120, 145],
                [150] * 6,
                direction="increasing",
            ),
            model="binomial",
            link="probit",
            upper=1.0,
            lower="estimate",
        )
        assert fit.status is Status.OK

        t = _boot(
            fit, Definition.baseline_multiple(5), "parametric_bootstrap", n_resamples=400, seed=5
        )

        assert t.status is Status.OK
        assert t.bootstrap is not None and t.bootstrap.failed > 0
        assert t.lo is None
        assert t.censoring in (Censoring.OPEN_LOWER, Censoring.NONE)
        assert any(w.startswith("BOOTSTRAP_FAILURES:") for w in t.warnings)


# ---------------------------------------------------------------------------------------------
# Smoothed count draws, and the zero-width guard
# ---------------------------------------------------------------------------------------------


_CUT = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
_CUT_SUCCESSES = [20, 20, 20, 20, 0, 0]


class TestSmoothedCountDraws:
    def test_isotonic_cells_at_zero_and_at_all_trials_are_drawn_from_a_half_count_rate(
        self,
    ) -> None:
        fit = _isotonic(_CUT, _CUT_SUCCESSES, [20] * 6)

        replicates = _seam()(
            fit,
            definition=Definition.absolute(0.5),
            method="cell_bootstrap",
            n_resamples=3000,
            seed=1,
        )

        means = np.mean(
            [[c.successes for c in data.cells()] for data, _ in replicates],  # type: ignore[union-attr]
            axis=0,
        )
        # Binomial(20, 0.5 / 21) has mean 0.476, and Binomial(20, 20.5 / 21) has mean 19.524.
        assert means[0] == pytest.approx(20.0 * 20.5 / 21.0, abs=0.06)
        assert means[4] == pytest.approx(20.0 * 0.5 / 21.0, abs=0.06)
        assert means[5] == pytest.approx(20.0 * 0.5 / 21.0, abs=0.06)

    def test_a_mid_rate_cell_keeps_its_rate(self) -> None:
        fit = _isotonic([0.0, 1.0, 2.0], [20, 10, 2], [20] * 3)

        replicates = _seam()(
            fit,
            definition=Definition.absolute(0.5),
            method="cell_bootstrap",
            n_resamples=3000,
            seed=1,
        )

        means = np.mean(
            [[c.successes for c in data.cells()] for data, _ in replicates],  # type: ignore[union-attr]
            axis=0,
        )
        assert means[1] == pytest.approx(20.0 * 10.5 / 21.0, abs=0.12)

    def test_a_binomial_fits_cell_bootstrap_uses_the_same_draw(self) -> None:
        fit = fit_dose_response(
            _counts_obs(
                [0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
                [20, 20, 19, 10, 2, 0],
                [20] * 6,
            ),
            model="binomial",
            link="probit",
            upper=1.0,
            lower=0.0,
        )
        assert fit.status is Status.OK

        replicates = _seam()(
            fit,
            definition=Definition.absolute(0.5),
            method="cell_bootstrap",
            n_resamples=500,
            seed=1,
        )

        last = np.mean([data.cells()[-1].successes for data, _ in replicates])  # type: ignore[union-attr]
        # Binomial(20, 0.5 / 21) has mean 0.476; a draw from k / n = 0 would give exactly 0.
        assert last > 0.3

    def test_all_success_then_all_failure_cells_no_longer_give_a_zero_width_interval(self) -> None:
        fit = _isotonic(_CUT, _CUT_SUCCESSES, [20] * 6)

        t = _boot(fit, Definition.absolute(0.5), "cell_bootstrap", n_resamples=999)

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.lo is not None and t.hi is not None and t.value is not None
        assert t.lo < t.hi
        assert t.lo <= t.value <= t.hi
        assert not any(w.startswith("BOOTSTRAP_ZERO_WIDTH:") for w in t.warnings)


class TestZeroWidthGuard:
    @pytest.mark.parametrize("method", ["parametric_bootstrap", "cell_bootstrap"])
    def test_a_collapsed_finite_interval_reports_no_bounds_and_a_warning(
        self, method: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fit = _rich_binomial()
        point = _point(fit, "delta" if method == "parametric_bootstrap" else "cell_bootstrap")
        assert point.value is not None
        _patch_thetas(monkeypatch, [math.log(point.value)] * 100)

        t = _boot(fit, Definition.absolute(0.5), method, n_resamples=100)

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.value == point.value
        assert t.lo is None and t.hi is None
        assert t.interval_method == method and t.bootstrap is not None
        assert any(w.startswith("BOOTSTRAP_ZERO_WIDTH:") for w in t.warnings)

    def test_an_isotonic_fit_is_guarded_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fit = _isotonic(_CUT, _CUT_SUCCESSES, [20] * 6)
        point = _point(fit, "cell_bootstrap")
        assert point.value is not None
        _patch_thetas(monkeypatch, [math.log(point.value)] * 100)

        t = _boot(fit, Definition.absolute(0.5), "cell_bootstrap", n_resamples=100)

        assert t.lo is None and t.hi is None and t.censoring is Censoring.NONE
        assert any(w.startswith("BOOTSTRAP_ZERO_WIDTH:") for w in t.warnings)

    def test_a_spread_interval_is_not_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fit = _rich_binomial()
        point = _point(fit, "delta")
        assert point.value is not None
        spread = math.log(point.value) + np.linspace(-0.05, 0.05, 100)
        _patch_thetas(monkeypatch, spread.tolist())

        t = _boot(fit, Definition.absolute(0.5), "parametric_bootstrap", n_resamples=100)

        assert t.lo is not None and t.hi is not None
        assert not any(w.startswith("BOOTSTRAP_ZERO_WIDTH:") for w in t.warnings)


# ---------------------------------------------------------------------------------------------
# Continuous cell means
# ---------------------------------------------------------------------------------------------


class TestContinuousCellVariance:
    def test_a_resampled_cell_mean_has_the_variance_of_a_mean_under_resampling(self) -> None:
        rng = np.random.default_rng(20261117)
        levels = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
        severity = [x for x in levels for _ in range(4)]
        x = np.asarray(severity)
        with np.errstate(divide="ignore"):
            z = (np.log(x) - math.log(0.8)) / 0.5
        share = np.where(x == 0.0, 0.0, 1.0 / (1.0 + np.exp(-z)))
        value = 10.0 - 9.0 * share + rng.normal(0.0, 0.6, size=x.size)
        obs = ContinuousObservations.from_observations(
            _LOG,
            severity=severity,
            value=value.tolist(),
            outcome="response",
            direction="decreasing",
        )
        cells = obs.cells()
        fit = fit_dose_response(obs, model="isotonic")

        replicates = _seam()(
            fit,
            definition=Definition.absolute(5.5),
            method="cell_bootstrap",
            n_resamples=4000,
            seed=1,
        )

        draws = np.array(
            [[c.mean for c in data.cells()] for data, _ in replicates]  # type: ignore[union-attr]
        )
        expected = np.array([c.sd**2 * (c.n - 1) / c.n**2 for c in cells])
        ratio = draws.var(axis=0, ddof=1) / expected
        # 4 observations per cell: the variance is 3/16 of sd^2, not the 1/4 of the earlier draw.
        assert float(np.mean(ratio)) == pytest.approx(1.0, abs=0.05)
        assert draws.mean(axis=0) == pytest.approx([c.mean for c in cells], abs=0.05)


# ---------------------------------------------------------------------------------------------
# Log-axis tie
# ---------------------------------------------------------------------------------------------


class TestFlatBlockEqualToTheTarget:
    def test_on_a_log_axis_a_block_at_the_target_from_the_first_level_is_unreachable(self) -> None:
        fit = _isotonic([0.0, 1.0, 2.0, 4.0], [20, 10, 10, 2], [20] * 4)

        t = _boot(fit, Definition.absolute(0.5), "cell_bootstrap")

        assert t.status is Status.UNREACHABLE
        assert (t.value, t.lo, t.hi) == (None, None, None)
        assert t.interval_method == "exact_bound" and t.bootstrap is None
        assert any("below the lowest tested severity" in w for w in t.warnings)

    def test_on_a_linear_axis_the_control_is_a_node_and_the_crossing_is_the_first_level(
        self,
    ) -> None:
        fit = _isotonic([0.0, 1.0, 2.0, 4.0], [20, 10, 10, 2], [20] * 4, axis=_LINEAR)

        t = _boot(fit, Definition.absolute(0.5), "cell_bootstrap", n_resamples=99)

        assert t.status is Status.OK and t.value == pytest.approx(1.0, rel=1e-12)

    def test_a_block_at_the_target_that_starts_after_the_first_level_is_a_value(self) -> None:
        # Rates 1, .8, .5, .5, .1: the smallest x with P(x) <= .5 is the start of the block, x = 2.
        fit = _isotonic([0.0, 1.0, 2.0, 4.0, 8.0], [20, 16, 10, 10, 2], [20] * 5)

        t = _boot(fit, Definition.absolute(0.5), "cell_bootstrap", n_resamples=99)

        assert t.status is Status.OK and t.value == pytest.approx(2.0, rel=1e-12)


# ---------------------------------------------------------------------------------------------
# Isotonic log-likelihood
# ---------------------------------------------------------------------------------------------


class TestIsotonicLogLikelihoodIsComparable:
    def test_binomial_value_is_at_least_the_probit_fits_on_the_same_counts(self) -> None:
        successes, trials = _rich_counts()
        obs = _counts_obs(_LEVELS, successes, trials)

        isotonic = fit_dose_response(obs, model="isotonic")
        probit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert probit.status is Status.OK and probit.log_likelihood is not None
        assert isotonic.log_likelihood is not None
        # The monotone nonparametric fit nests every monotone curve, so it cannot fit worse.
        assert isotonic.log_likelihood >= probit.log_likelihood - 1e-9
        # Both carry log C(n, k), so the gap is a few likelihood units, not hundreds.
        assert isotonic.log_likelihood - probit.log_likelihood < 15.0

    def test_binomial_value_is_the_full_binomial_log_likelihood_at_the_cell_estimates(
        self,
    ) -> None:
        successes, trials = _rich_counts()
        fit = _isotonic(_LEVELS, successes, trials)
        assert fit.cell_estimates is not None
        p = np.clip(np.asarray(fit.cell_estimates), 1e-300, 1.0 - 1e-16)
        k, n = np.asarray(successes, float), np.asarray(trials, float)

        expected = float(
            np.sum(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + k * np.log(p))
            + np.sum(np.where(n - k > 0, (n - k) * np.log1p(-p), 0.0))
        )

        assert fit.log_likelihood == pytest.approx(expected, rel=1e-9)

    def test_continuous_value_follows_the_gaussian_fits_convention(self) -> None:
        rng = np.random.default_rng(20261117)
        levels = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
        severity = [x for x in levels for _ in range(6)]
        x = np.asarray(severity)
        with np.errstate(divide="ignore"):
            z = (np.log(x) - math.log(0.8)) / 0.5
        share = np.where(x == 0.0, 0.0, 1.0 / (1.0 + np.exp(-z)))
        value = 10.0 - 9.0 * share + rng.normal(0.0, 0.3, size=x.size)
        obs = ContinuousObservations.from_observations(
            _LOG,
            severity=severity,
            value=value.tolist(),
            outcome="response",
            direction="decreasing",
        )

        isotonic = fit_dose_response(obs, model="isotonic")
        gaussian = fit_dose_response(
            obs, model="gaussian", link="logit", upper="estimate", lower="estimate"
        )

        assert gaussian.status is Status.OK and gaussian.log_likelihood is not None
        assert isotonic.log_likelihood is not None
        # Same constants (a free variance at its maximum, the 2 pi term included), so the monotone
        # fit, with more freedom, is not lower, and is higher by about its extra parameters.
        assert isotonic.log_likelihood >= gaussian.log_likelihood - 1e-9
        assert isotonic.log_likelihood - gaussian.log_likelihood < 8.0


# ---------------------------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------------------------


class TestExcludedEstimateWarningStatesZ0:
    def test_the_warning_gives_the_bias_correction_from_the_valid_replicates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fit = _rich_binomial()
        point = _point(fit, "delta")
        assert point.value is not None
        centre = math.log(point.value)
        # One replicate of 100 below the estimate, the rest well above it.
        thetas = [centre - 0.2] + (centre + 0.5 + np.linspace(0.0, 0.1, 99)).tolist()
        _patch_thetas(monkeypatch, thetas)

        t = _boot(fit, Definition.absolute(0.5), "parametric_bootstrap", n_resamples=100)

        messages = [w for w in t.warnings if w.startswith("BOOTSTRAP_EXCLUDES_ESTIMATE:")]
        assert len(messages) == 1
        assert "z0" in messages[0]
        assert f"{norm.ppf(0.01):.2f}" in messages[0] or f"{norm.ppf(0.01):.3f}" in messages[0]


class TestFewResamples:
    @pytest.mark.parametrize(("n_resamples", "warned"), [(19, True), (38, True), (39, False)])
    def test_fewer_than_two_over_alpha_minus_one_resamples_warn_at_level_095(
        self, n_resamples: int, warned: bool
    ) -> None:
        fit = _isotonic(_LEVELS, *_rich_counts())

        t = _boot(fit, Definition.absolute(0.5), "cell_bootstrap", n_resamples=n_resamples)

        assert any(w.startswith("BOOTSTRAP_FEW_RESAMPLES:") for w in t.warnings) is warned

    @pytest.mark.parametrize(("n_resamples", "warned"), [(198, True), (199, False)])
    def test_the_cut_off_moves_with_the_level(self, n_resamples: int, warned: bool) -> None:
        fit = _isotonic(_LEVELS, *_rich_counts())

        t = _boot(
            fit, Definition.absolute(0.5), "cell_bootstrap", n_resamples=n_resamples, level=0.99
        )

        assert any(w.startswith("BOOTSTRAP_FEW_RESAMPLES:") for w in t.warnings) is warned


class TestUnreachableWording:
    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_a_one_sided_pile_of_infinities_is_not_described_as_unreachable_on_each_side(
        self, sign: float, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fit = _rich_binomial()
        point = _point(fit, "delta")
        assert point.value is not None
        _patch_thetas(monkeypatch, [sign * math.inf] * 99 + [math.log(point.value)])

        t = _boot(fit, Definition.absolute(0.5), "parametric_bootstrap", n_resamples=100)

        assert t.lo is None and t.hi is None and t.censoring is Censoring.NONE
        assert t.bootstrap is not None
        assert (t.bootstrap.unreachable_low == 0) or (t.bootstrap.unreachable_high == 0)
        for w in t.warnings:
            assert "on each side" not in w
            assert "each side" not in w
