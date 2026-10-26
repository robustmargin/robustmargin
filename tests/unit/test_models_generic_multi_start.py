"""The multi-start rule of the direct-likelihood path (``_fit_generic``).

The pilot run (BFGS then Newton from the pilot start) is always made. BFGS also runs from the
five fixed starts in ``models._ALTERNATIVE_STARTS``. If the best alternative's BFGS
log-likelihood exceeds the pilot's best by more than ``_NEWTON_DECREMENT_TOLERANCE``, that start
is Newton-polished and every existing check runs on it. Otherwise the pilot result is checked as
before, bit-identically. With both asymptotes fixed at (1, 0) the problem is concave and no
alternative start is run.

A fit whose log-likelihood does not exceed the closed-form supremum on the ``s -> infinity``
boundary (``models._flat_supremum_log_likelihood``: the binomial log-likelihood at the pooled
rate, clipped to the feasible asymptote constants) is ``NOT_CONVERGED`` with the wrong-sign/flat
reason.

Basis of every log-likelihood below: the FULL binomial log-likelihood, ``lchoose`` terms included,
the same basis as ``Fit.log_likelihood``. The reference values were computed with an independent
test-side oracle (``scipy.optimize.minimize``, Nelder-Mead from a 5 x 5 grid of (mu, log s)
starts crossed with asymptote-logit starts, a second Nelder-Mead pass from each result, own
likelihood in numpy/scipy, no code from ``robustmargin``). Results:

* R21 design (linear, logit, u 0.95, l 0.05): llf -6.556781, mu 89.513299, s 60.713128.
* R19 design (log, probit, u 0.95, l 0.3): llf -13.482347, mu 2.388397, s 0.532689 (log units).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from statsmodels.base.model import GenericLikelihoodModel

from robustmargin import Axis, Observations, Status, fit_dose_response
from robustmargin import models as models_module

_LINEAR = Axis(name="severity", unit="unit", scale="linear")
_LOG = Axis(name="severity", unit="unit", scale="log")

_R21_SEVERITY = [54.966, 84.166, 95.205, 95.73]
_R21_SUCCESSES = [6, 6, 6, 3]
_R21_LLF = -6.556780871935
_R21_MU = 89.51329854831
_R21_S = 60.71312786512

_R19_SEVERITY = [3.4479, 29.4978, 320.0891, 2859.2341, 6797.8529, 15781.2882]
_R19_SUCCESSES = [47, 16, 19, 15, 15, 11]
_R19_LLF = -13.48234727954
_R19_MU = 2.38839701661
_R19_S = 0.53268889447
# The step supremum of the R19 design, -13.5294 (below the interior maximum).
_R19_STEP_SUPREMUM = -13.5294


def _r21(*, direction: str = "decreasing") -> Observations:
    successes = _R21_SUCCESSES if direction == "decreasing" else [10 - k for k in _R21_SUCCESSES]
    return Observations.from_counts(
        _LINEAR,
        severity=_R21_SEVERITY,
        successes=successes,
        trials=[10] * 4,
        outcome="success",
        direction=direction,
    )


def _r19() -> Observations:
    return Observations.from_counts(
        _LOG,
        severity=_R19_SEVERITY,
        successes=_R19_SUCCESSES,
        trials=[50] * 6,
        outcome="success",
        direction="decreasing",
    )


def _fit_r21(direction: str = "decreasing") -> Any:
    return fit_dose_response(
        _r21(direction=direction), model="binomial", link="logit", upper=0.95, lower=0.05
    )


def _fit_r19() -> Any:
    return fit_dose_response(_r19(), model="binomial", link="probit", upper=0.95, lower=0.3)


def _value(fit: Any, name: str) -> float:
    return float(fit.params[name].value)


# --- the interface the rule is specified through -------------------------------------------------


def test_alternative_starts_are_the_five_fixed_starts_on_the_centred_scaled_covariate() -> None:
    assert models_module._ALTERNATIVE_STARTS == (
        (0.0, 1.0),
        (-1.0, 0.25),
        (1.0, 0.25),
        (-1.0, 4.0),
        (1.0, 4.0),
    )


# --- the reported local-maximum and plateau designs ----------------------------------------------


def test_fit_at_a_local_maximum_moves_to_the_global_maximum_of_the_full_log_likelihood() -> None:
    # Reported as an OK fit at llf -16.591 (mu 95.374, s 0.373), a local maximum in the near-step
    # basin; the global maximum is -6.5568.
    fit = _fit_r21()

    assert fit.status is Status.OK, fit.warnings
    assert fit.log_likelihood == pytest.approx(_R21_LLF, abs=1e-4)
    assert _value(fit, "mu") == pytest.approx(_R21_MU, rel=1e-3)
    assert _value(fit, "s") == pytest.approx(_R21_S, rel=1e-3)
    assert _value(fit, "lower") == 0.05
    assert _value(fit, "upper") == 0.95


def test_increasing_mirror_of_the_local_maximum_design_gives_the_mirrored_fit() -> None:
    # Successes n - k with the asymptote pair (1 - upper, 1 - lower) = (0.05, 0.95) is the
    # complement of the decreasing design.
    fit = _fit_r21(direction="increasing")

    assert fit.status is Status.OK, fit.warnings
    assert fit.log_likelihood == pytest.approx(_R21_LLF, abs=1e-4)
    assert _value(fit, "mu") == pytest.approx(_R21_MU, rel=1e-3)
    assert _value(fit, "s") == pytest.approx(_R21_S, rel=1e-3)


def test_design_reported_as_a_step_plateau_is_ok_at_its_interior_maximum() -> None:
    # Reported as NOT_CONVERGED (step plateau) although an interior maximum, -13.4823, exceeds
    # the step supremum, -13.5294.
    fit = _fit_r19()

    assert fit.status is Status.OK, fit.warnings
    assert fit.log_likelihood is not None
    assert fit.log_likelihood > _R19_STEP_SUPREMUM
    assert fit.log_likelihood == pytest.approx(_R19_LLF, abs=1e-4)
    assert _value(fit, "mu") == pytest.approx(_R19_MU, rel=1e-3)
    assert _value(fit, "s") == pytest.approx(_R19_S, rel=1e-3)


def test_multi_start_fit_is_deterministic() -> None:
    first = _fit_r21()
    second = _fit_r21()

    assert first.log_likelihood == second.log_likelihood
    assert {k: v.value for k, v in first.params.items()} == {
        k: v.value for k, v in second.params.items()
    }


# --- the switch rule leaves a fit that is already at the global maximum alone --------------------


def _smooth_estimated_design() -> Observations:
    return Observations.from_counts(
        _LOG,
        severity=[0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        successes=[190, 186, 170, 120, 60, 20, 10],
        trials=[200] * 7,
        outcome="success",
        direction="decreasing",
    )


def test_fit_already_at_the_global_maximum_is_bit_identical_with_alternative_starts_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs = _smooth_estimated_design()
    with_starts = fit_dose_response(
        obs, model="binomial", link="logit", upper="estimate", lower="estimate"
    )
    monkeypatch.setattr(models_module, "_ALTERNATIVE_STARTS", ())

    pilot_only = fit_dose_response(
        obs, model="binomial", link="logit", upper="estimate", lower="estimate"
    )

    assert with_starts.status is Status.OK
    assert pilot_only.status is Status.OK
    assert with_starts.log_likelihood == pilot_only.log_likelihood
    assert {k: v.value for k, v in with_starts.params.items()} == {
        k: v.value for k, v in pilot_only.params.items()
    }
    assert with_starts.covariance == pilot_only.covariance


# --- no extra starts on the concave (1, 0) problem -----------------------------------------------


def _count_fit_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls: dict[str, int] = {}
    real_fit = GenericLikelihoodModel.fit

    def counting(self: Any, *args: Any, **kwargs: Any) -> Any:
        method = str(kwargs.get("method"))
        calls[method] = calls.get(method, 0) + 1
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(GenericLikelihoodModel, "fit", counting)
    return calls


def test_direct_likelihood_refit_with_asymptotes_fixed_at_one_and_zero_runs_one_bfgs_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs = Observations.from_counts(
        _LOG,
        severity=[1.0, 2.0, 4.0, 8.0, 16.0],
        successes=[48, 40, 25, 12, 3],
        trials=[50] * 5,
        outcome="success",
        direction="decreasing",
    )
    calls = _count_fit_calls(monkeypatch)

    refit = models_module._refit_by_direct_likelihood(
        obs,
        cells=obs.cells(),
        cluster_ids=None,
        link="probit",
        reason="test",
        mirrored=False,
    )

    assert refit.status is Status.OK, refit.warnings
    assert calls["bfgs"] == 1


def test_estimated_asymptote_fit_runs_the_pilot_and_every_alternative_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A direct `_fit_generic` call: `fit_dose_response` also refits the other links for its
    # diagnostics, which would multiply the count.
    obs = _r21()
    calls = _count_fit_calls(monkeypatch)

    fit = models_module._fit_generic(
        obs,
        cells=obs.cells(),
        cluster_ids=None,
        link="logit",
        upper_is_estimate=False,
        upper_value=0.95,
        lower_is_estimate=False,
        lower_value=0.05,
        mirrored=False,
    )

    assert fit.status is Status.OK, fit.warnings
    assert calls["bfgs"] == 1 + len(models_module._ALTERNATIVE_STARTS)


# --- the flat supremum, closed form --------------------------------------------------------------

_FLAT_SUCCESSES = [6, 6, 6, 3]
_FLAT_TRIALS = [10, 10, 10, 10]


def _flat_expected(p: float) -> float:
    lchoose = sum(
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        for k, n in zip(_FLAT_SUCCESSES, _FLAT_TRIALS, strict=True)
    )
    return lchoose + 21 * math.log(p) + 19 * math.log(1.0 - p)


def _flat(
    *,
    estimate_lower: bool,
    estimate_upper: bool,
    lower_value: float = 0.0,
    upper_value: float = 1.0,
) -> float:
    return float(
        models_module._flat_supremum_log_likelihood(
            np.asarray(_R21_SEVERITY, dtype=np.float64),
            np.asarray(_FLAT_SUCCESSES, dtype=np.float64),
            np.asarray(_FLAT_TRIALS, dtype=np.float64),
            estimate_lower=estimate_lower,
            estimate_upper=estimate_upper,
            lower_value=lower_value,
            upper_value=upper_value,
            scale="linear",
        )
    )


def test_flat_supremum_with_both_asymptotes_fixed_is_the_pooled_rate_inside_the_range() -> None:
    value = _flat(estimate_lower=False, estimate_upper=False, lower_value=0.05, upper_value=0.95)

    assert value == pytest.approx(_flat_expected(21 / 40), abs=1e-9)


def test_flat_supremum_with_both_asymptotes_fixed_clips_the_pooled_rate_up_to_lower() -> None:
    value = _flat(estimate_lower=False, estimate_upper=False, lower_value=0.6, upper_value=0.9)

    assert value == pytest.approx(_flat_expected(0.6), abs=1e-9)


def test_flat_supremum_with_both_asymptotes_fixed_clips_the_pooled_rate_down_to_upper() -> None:
    value = _flat(estimate_lower=False, estimate_upper=False, lower_value=0.1, upper_value=0.4)

    assert value == pytest.approx(_flat_expected(0.4), abs=1e-9)


def test_flat_supremum_with_upper_estimated_is_clipped_to_the_fixed_lower() -> None:
    value = _flat(estimate_lower=False, estimate_upper=True, lower_value=0.7)

    assert value == pytest.approx(_flat_expected(0.7), abs=1e-9)


def test_flat_supremum_with_upper_estimated_is_the_pooled_rate_when_lower_is_below_it() -> None:
    value = _flat(estimate_lower=False, estimate_upper=True, lower_value=0.2)

    assert value == pytest.approx(_flat_expected(21 / 40), abs=1e-9)


def test_flat_supremum_with_lower_estimated_is_clipped_to_the_fixed_upper() -> None:
    value = _flat(estimate_lower=True, estimate_upper=False, upper_value=0.4)

    assert value == pytest.approx(_flat_expected(0.4), abs=1e-9)


def test_flat_supremum_with_lower_estimated_is_the_pooled_rate_when_upper_is_above_it() -> None:
    value = _flat(estimate_lower=True, estimate_upper=False, upper_value=0.95)

    assert value == pytest.approx(_flat_expected(21 / 40), abs=1e-9)


def test_flat_supremum_with_both_asymptotes_estimated_is_the_pooled_rate() -> None:
    value = _flat(estimate_lower=True, estimate_upper=True)

    assert value == pytest.approx(_flat_expected(21 / 40), abs=1e-9)


def test_flat_supremum_of_all_failures_with_lower_estimated_is_the_lchoose_total() -> None:
    value = models_module._flat_supremum_log_likelihood(
        np.asarray([1.0, 2.0], dtype=np.float64),
        np.asarray([0.0, 0.0], dtype=np.float64),
        np.asarray([10.0, 10.0], dtype=np.float64),
        estimate_lower=True,
        estimate_upper=True,
        lower_value=0.0,
        upper_value=1.0,
        scale="linear",
    )

    # Pooled rate 0: every cell fits exactly, so the log-likelihood is 0 (lchoose(n, 0) = 0).
    assert value == pytest.approx(0.0, abs=1e-12)


# --- the flat supremum with a log-axis zero-severity control --------------------------------------
#
# On a log axis the curve pins P(0) = u for every (mu, s), so on the s -> infinity boundary the
# control stays at u and only the nonzero levels share one constant p. Reference values below were
# checked against a test-side brute-force grid over (u, p) with the control at u (and, for the
# linear axis, with every cell at p); see each test.

_CONTROL_SEVERITY = [0.0, 1.0, 2.0, 4.0]
_CONTROL_TRIALS = [10, 10, 10, 10]
_HIGH_CONTROL_SUCCESSES = [9, 6, 5, 3]  # control rate 0.9; nonzero pooled rate 14/30
_LOW_CONTROL_SUCCESSES = [3, 8, 7, 9]  # control rate 0.3; nonzero pooled rate 24/30


def _binomial_part(k: int, n: int, p: float) -> float:
    return k * math.log(p) + (n - k) * math.log(1.0 - p)


def _coefficients(successes: list[int], trials: list[int]) -> float:
    return sum(
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        for k, n in zip(successes, trials, strict=True)
    )


def _flat_with_control(
    successes: list[int],
    *,
    scale: str,
    estimate_lower: bool,
    estimate_upper: bool,
    lower_value: float = 0.0,
    upper_value: float = 1.0,
) -> float:
    return float(
        models_module._flat_supremum_log_likelihood(
            np.asarray(_CONTROL_SEVERITY, dtype=np.float64),
            np.asarray(successes, dtype=np.float64),
            np.asarray(_CONTROL_TRIALS, dtype=np.float64),
            estimate_lower=estimate_lower,
            estimate_upper=estimate_upper,
            lower_value=lower_value,
            upper_value=upper_value,
            scale=scale,
        )
    )


def test_flat_supremum_log_axis_control_stays_at_fixed_upper_and_nonzero_levels_pool() -> None:
    value = _flat_with_control(
        _HIGH_CONTROL_SUCCESSES,
        scale="log",
        estimate_lower=False,
        estimate_upper=False,
        lower_value=0.1,
        upper_value=0.9,
    )

    # Control (9 of 10) at u = 0.9; nonzero levels (14 of 30) at their pooled rate, inside
    # [0.1, 0.9]. Brute-force grid reference: -6.011917.
    expected = (
        _coefficients(_HIGH_CONTROL_SUCCESSES, _CONTROL_TRIALS)
        + _binomial_part(9, 10, 0.9)
        + _binomial_part(14, 30, 14 / 30)
    )
    assert expected == pytest.approx(-6.011917, abs=1e-5)
    assert value == pytest.approx(expected, abs=1e-9)


def test_flat_supremum_log_axis_control_stays_at_fixed_upper_with_lower_estimated() -> None:
    value = _flat_with_control(
        _HIGH_CONTROL_SUCCESSES,
        scale="log",
        estimate_lower=True,
        estimate_upper=False,
        upper_value=0.9,
    )

    # Same configuration with low = 0; the pooled nonzero rate is unchanged. Grid: -6.011917.
    expected = (
        _coefficients(_HIGH_CONTROL_SUCCESSES, _CONTROL_TRIALS)
        + _binomial_part(9, 10, 0.9)
        + _binomial_part(14, 30, 14 / 30)
    )
    assert value == pytest.approx(expected, abs=1e-9)


def test_flat_supremum_log_axis_fixed_upper_clips_the_nonzero_pooled_rate_to_upper() -> None:
    value = _flat_with_control(
        _LOW_CONTROL_SUCCESSES,
        scale="log",
        estimate_lower=False,
        estimate_upper=False,
        lower_value=0.1,
        upper_value=0.6,
    )

    # Nonzero pooled rate 24/30 = 0.8 is clipped down to u = 0.6; the control (3 of 10) sits at 0.6.
    expected = (
        _coefficients(_LOW_CONTROL_SUCCESSES, _CONTROL_TRIALS)
        + _binomial_part(3, 10, 0.6)
        + _binomial_part(24, 30, 0.6)
    )
    assert value == pytest.approx(expected, abs=1e-9)


def test_flat_supremum_log_axis_upper_estimated_keeps_control_rate_when_it_is_at_least_p() -> None:
    value = _flat_with_control(
        _HIGH_CONTROL_SUCCESSES,
        scale="log",
        estimate_lower=False,
        estimate_upper=True,
        lower_value=0.2,
    )

    # p = max(14/30, 0.2) = 14/30; control rate 0.9 >= p, so u = 0.9 and p is kept.
    # Brute-force grid over (u, p): -6.011917.
    expected = (
        _coefficients(_HIGH_CONTROL_SUCCESSES, _CONTROL_TRIALS)
        + _binomial_part(9, 10, 0.9)
        + _binomial_part(14, 30, 14 / 30)
    )
    assert value == pytest.approx(expected, abs=1e-9)


def test_flat_supremum_log_axis_upper_estimated_collapses_to_all_cells_pooled_rate() -> None:
    value = _flat_with_control(
        _LOW_CONTROL_SUCCESSES,
        scale="log",
        estimate_lower=True,
        estimate_upper=True,
    )

    # Control rate 0.3 < p = 0.8, so u = p = pooled rate of all cells, 27/40.
    # Brute-force grid over (u, p): -9.539010.
    expected = _coefficients(_LOW_CONTROL_SUCCESSES, _CONTROL_TRIALS) + _binomial_part(
        27, 40, 27 / 40
    )
    assert expected == pytest.approx(-9.539010, abs=1e-5)
    assert value == pytest.approx(expected, abs=1e-9)


def test_flat_supremum_log_axis_collapse_is_clipped_up_to_the_fixed_lower() -> None:
    value = _flat_with_control(
        _LOW_CONTROL_SUCCESSES,
        scale="log",
        estimate_lower=False,
        estimate_upper=True,
        lower_value=0.7,
    )

    # Control rate 0.3 < p = max(0.8, 0.7) = 0.8, so u = p = max(27/40, 0.7) = 0.7.
    # Brute-force grid over (u, p) with p >= 0.7: -9.597639.
    expected = _coefficients(_LOW_CONTROL_SUCCESSES, _CONTROL_TRIALS) + _binomial_part(27, 40, 0.7)
    assert expected == pytest.approx(-9.597639, abs=1e-5)
    assert value == pytest.approx(expected, abs=1e-9)


def test_flat_supremum_linear_axis_zero_severity_is_not_pinned_to_upper() -> None:
    value = _flat_with_control(
        _HIGH_CONTROL_SUCCESSES,
        scale="linear",
        estimate_lower=False,
        estimate_upper=False,
        lower_value=0.1,
        upper_value=0.9,
    )

    # On a linear axis severity 0 is an ordinary level: every cell pools, 23/40. Grid: -9.307571.
    expected = _coefficients(_HIGH_CONTROL_SUCCESSES, _CONTROL_TRIALS) + _binomial_part(
        23, 40, 23 / 40
    )
    assert expected == pytest.approx(-9.307571, abs=1e-5)
    assert value == pytest.approx(expected, abs=1e-9)


# --- end to end: a log-axis control must not make the flat guard fire or miss ---------------------
#
# Independent oracle: own numpy likelihood with P(0) = u pinned, Nelder-Mead from a 13 x 11 grid of
# (mu, log s) on the standardised log covariate, then a second pass. Flat supremum from a brute
# (u, p) grid with the control at u.

_FALSE_FLAT_SEVERITY = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
_FALSE_FLAT_SUCCESSES = [50, 45, 42, 40, 37, 34, 30]
_FALSE_FLAT_LLF = -68.635337
_FALSE_FLAT_SUPREMUM = -71.794304

_LOW_GUARD_SEVERITY = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0]
_LOW_GUARD_SUCCESSES = [46, 30, 28, 27, 25, 24]
_LOW_GUARD_LLF = -12.914522
_LOW_GUARD_SUPREMUM = -13.817315


def _log_axis_observations(
    severity: list[float], successes: list[int], trials: int
) -> Observations:
    return Observations.from_counts(
        _LOG,
        severity=severity,
        successes=successes,
        trials=[trials] * len(severity),
        outcome="success",
        direction="decreasing",
    )


def _flat_supremum_of(
    severity: list[float], successes: list[int], trials: int, *, lower: float, upper: float
) -> float:
    return float(
        models_module._flat_supremum_log_likelihood(
            np.asarray(severity, dtype=np.float64),
            np.asarray(successes, dtype=np.float64),
            np.asarray([trials] * len(severity), dtype=np.float64),
            estimate_lower=False,
            estimate_upper=False,
            lower_value=lower,
            upper_value=upper,
            scale="log",
        )
    )


def test_log_axis_control_fit_beating_the_pinned_flat_supremum_is_ok_not_flat_plateau() -> None:
    # Reported as a false NOT_CONVERGED: the unpinned pooled-rate supremum (-23.24) is higher than
    # any fit with P(0) = u can reach, but the supremum with the control at u is -71.79 and the
    # fit, -68.635, beats it.
    assert _flat_supremum_of(
        _FALSE_FLAT_SEVERITY, _FALSE_FLAT_SUCCESSES, 100, lower=0.05, upper=0.9
    ) == pytest.approx(_FALSE_FLAT_SUPREMUM, abs=1e-4)

    fit = fit_dose_response(
        _log_axis_observations(_FALSE_FLAT_SEVERITY, _FALSE_FLAT_SUCCESSES, 100),
        model="binomial",
        link="logit",
        upper=0.9,
        lower=0.05,
    )

    assert fit.status is Status.OK, fit.warnings
    assert fit.log_likelihood == pytest.approx(_FALSE_FLAT_LLF, abs=1e-3)
    assert fit.log_likelihood > _FALSE_FLAT_SUPREMUM


def test_log_axis_control_flat_supremum_is_not_set_too_low_by_pooling_the_control() -> None:
    # Pooling the control with the other levels gives -28.74, below the true supremum -13.82 with
    # the control at u = 0.95. The interior maximum (-12.9145) beats the true supremum, so the fit
    # is OK; a guard set at -28.74 would also say OK, but one that read the guard as -13.82 must
    # not be a plateau here.
    helper = _flat_supremum_of(
        _LOW_GUARD_SEVERITY, _LOW_GUARD_SUCCESSES, 50, lower=0.05, upper=0.95
    )
    assert helper == pytest.approx(_LOW_GUARD_SUPREMUM, abs=1e-4)
    assert helper > -28.74 + 10.0

    fit = fit_dose_response(
        _log_axis_observations(_LOW_GUARD_SEVERITY, _LOW_GUARD_SUCCESSES, 50),
        model="binomial",
        link="logit",
        upper=0.95,
        lower=0.05,
    )

    assert fit.status is Status.OK, fit.warnings
    assert fit.log_likelihood == pytest.approx(_LOW_GUARD_LLF, abs=1e-3)
    assert fit.log_likelihood > helper


# --- the flat guard ------------------------------------------------------------------------------


def test_fit_not_above_the_flat_supremum_is_not_converged_with_the_wrong_sign_or_flat_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With the alternative starts disabled the pilot run stops at the near-step local maximum,
    # llf -16.591. The flat supremum of this design (pooled rate 21/40, inside [0.05, 0.95]) is
    # -6.847, so that point is not an interior maximum and must not be reported OK.
    monkeypatch.setattr(models_module, "_ALTERNATIVE_STARTS", ())
    flat = _flat_expected(21 / 40)
    assert flat > -16.5913  # the pilot's llf is below the flat supremum

    fit = _fit_r21()

    assert fit.status is Status.NOT_CONVERGED
    assert fit.params is None
    assert fit.log_likelihood is None
    assert models_module._wrong_sign_or_flat_warning(mirrored=False) in fit.warnings


def test_flat_guard_names_the_increasing_direction_for_an_increasing_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models_module, "_ALTERNATIVE_STARTS", ())

    fit = _fit_r21(direction="increasing")

    assert fit.status is Status.NOT_CONVERGED
    assert models_module._wrong_sign_or_flat_warning(mirrored=True) in fit.warnings


def test_flat_guard_does_not_fire_on_a_fit_that_exceeds_the_flat_supremum() -> None:
    fit = _fit_r21()

    assert fit.status is Status.OK
    assert fit.log_likelihood is not None
    assert fit.log_likelihood > _flat_expected(21 / 40) + 0.2
