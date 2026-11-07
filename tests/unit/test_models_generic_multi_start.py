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

import itertools
import math
from typing import Any

import numpy as np
import pytest
from statsmodels.base.model import GenericLikelihoodModel

from robustmargin import Axis, Definition, Observations, Status, fit_dose_response, threshold
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
    returned: list[int] = []
    real_scan = models_module._scan_starts

    def recording_scan(*args: Any, **kwargs: Any) -> Any:
        starts = real_scan(*args, **kwargs)
        returned.append(len(starts))
        return starts

    monkeypatch.setattr(models_module, "_scan_starts", recording_scan)

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
    # The pilot, every fixed alternative start, and one start per point the scan returned.
    assert len(returned) == 1
    assert calls["bfgs"] == 1 + len(models_module._ALTERNATIVE_STARTS) + returned[0]


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
    # The scan is disabled too, so the guard itself is exercised.
    monkeypatch.setattr(models_module, "_scan_starts", lambda *a, **k: [])
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
    monkeypatch.setattr(models_module, "_scan_starts", lambda *a, **k: [])

    fit = _fit_r21(direction="increasing")

    assert fit.status is Status.NOT_CONVERGED
    assert models_module._wrong_sign_or_flat_warning(mirrored=True) in fit.warnings


def test_flat_guard_does_not_fire_on_a_fit_that_exceeds_the_flat_supremum() -> None:
    fit = _fit_r21()

    assert fit.status is Status.OK
    assert fit.log_likelihood is not None
    assert fit.log_likelihood > _flat_expected(21 / 40) + 0.2


# --- a refusal as flat or step must not hide an interior maximum ---------------------------------
#
# Log axis with a zero-severity control, cloglog, upper 0.95, lower estimated, decreasing. Every
# optimiser start has a location near the centre of the nonzero levels; the interior maximum lies
# in the gap between the pinned control and the first nonzero level. Independent oracle
# (Nelder-Mead on the centred-and-scaled covariate, own numpy likelihood): interior maximum
# -47.545652 (mu' -3.97, s' 2.1, lower 0.504); step supremum -47.547900; flat -47.593179.

_GAP_SEVERITY = [
    0.0,
    15.0,
    37.7879274206032,
    58.9841931089883,
    59.3788140504317,
    84.39271170536418,
]
_GAP_SUCCESSES = [1, 26, 14, 27, 1, 12]
_GAP_TRIALS = [10, 49, 26, 47, 20, 14]
_GAP_LLF = -47.545652
_GAP_STEP_SUPREMUM = -47.547900
_GAP_FLAT_SUPREMUM = -47.593179
_FLAT_OR_STEP_FRAGMENT = "a step between adjacent tested levels"


def _gap_observations(
    *, severity_factor: float = 1.0, direction: str = "decreasing"
) -> Observations:
    successes = (
        _GAP_SUCCESSES
        if direction == "decreasing"
        else [n - k for k, n in zip(_GAP_SUCCESSES, _GAP_TRIALS, strict=True)]
    )
    return Observations.from_counts(
        _LOG,
        severity=[x * severity_factor for x in _GAP_SEVERITY],
        successes=successes,
        trials=_GAP_TRIALS,
        outcome="success",
        direction=direction,
    )


def _fit_gap(*, severity_factor: float = 1.0) -> Any:
    return fit_dose_response(
        _gap_observations(severity_factor=severity_factor),
        model="binomial",
        link="cloglog",
        upper=0.95,
        lower="estimate",
    )


def _fit_gap_increasing() -> Any:
    return fit_dose_response(
        _gap_observations(direction="increasing"),
        model="binomial",
        link="cloglog",
        upper="estimate",
        lower=0.05,
    )


def _is_flat_or_step_refusal(fit: Any) -> bool:
    if fit.status is not Status.NOT_CONVERGED or not fit.warnings:
        return False
    first = fit.warnings[0]
    return (
        first == models_module._wrong_sign_or_flat_warning(mirrored=False)
        or first == models_module._wrong_sign_or_flat_warning(mirrored=True)
        or _FLAT_OR_STEP_FRAGMENT in first
    )


def test_interior_maximum_in_the_gap_after_the_control_is_not_refused_as_flat_or_step() -> None:
    fit = _fit_gap()

    assert not _is_flat_or_step_refusal(fit), fit.warnings
    assert fit.status is Status.OK, fit.warnings


def test_interior_maximum_in_the_gap_after_the_control_reaches_the_oracle_log_likelihood() -> None:
    fit = _fit_gap()

    assert fit.log_likelihood is not None
    assert fit.log_likelihood >= _GAP_LLF - 1e-6
    assert fit.log_likelihood > _GAP_STEP_SUPREMUM
    assert fit.log_likelihood > _GAP_FLAT_SUPREMUM


def test_gap_design_increasing_mirror_gives_the_mirrored_fit() -> None:
    decreasing = _fit_gap()
    increasing = _fit_gap_increasing()

    assert increasing.status is Status.OK, increasing.warnings
    assert decreasing.status is Status.OK, decreasing.warnings
    assert increasing.log_likelihood == pytest.approx(decreasing.log_likelihood, abs=1e-6)
    assert _value(increasing, "mu") == pytest.approx(_value(decreasing, "mu"), rel=1e-3, abs=1e-3)
    assert _value(increasing, "s") == pytest.approx(_value(decreasing, "s"), rel=1e-3)
    assert _value(increasing, "upper") == pytest.approx(1.0 - _value(decreasing, "lower"), abs=1e-4)
    assert _value(increasing, "lower") == pytest.approx(0.05, abs=1e-12)


def test_gap_design_is_unchanged_by_rescaling_the_severity_units() -> None:
    base = _fit_gap()
    scaled = _fit_gap(severity_factor=1000.0)

    assert scaled.status is Status.OK, scaled.warnings
    assert base.status is Status.OK, base.warnings
    assert scaled.log_likelihood == pytest.approx(base.log_likelihood, abs=1e-6)
    # Log axis: mu is a log-severity, so it shifts by log(1000); s and the asymptotes do not move.
    assert _value(scaled, "mu") == pytest.approx(_value(base, "mu") + math.log(1000.0), abs=1e-3)
    assert _value(scaled, "s") == pytest.approx(_value(base, "s"), rel=1e-3)
    assert _value(scaled, "lower") == pytest.approx(_value(base, "lower"), abs=1e-4)


def test_gap_design_repeat_run_is_identical() -> None:
    first = _fit_gap()
    second = _fit_gap()

    assert first.status is second.status
    assert first.log_likelihood == second.log_likelihood
    assert {k: v.value for k, v in first.params.items()} == {
        k: v.value for k, v in second.params.items()
    }


# --- the scan runs once on every fit except asymptotes fixed at (1, 0) -------------------------


def _count_scans(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real = models_module._scan_starts

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(models_module, "_scan_starts", counting)
    return calls


def _generic(
    scale: str,
    link: str,
    severity: list[float],
    successes: list[int],
    trials: list[int],
    *,
    upper: Any,
    lower: Any,
) -> Any:
    obs = Observations.from_counts(
        _LOG if scale == "log" else _LINEAR,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )
    return models_module._fit_generic(
        obs,
        cells=obs.cells(),
        cluster_ids=None,
        link=link,
        upper_is_estimate=upper == "estimate",
        upper_value=1.0 if upper == "estimate" else float(upper),
        lower_is_estimate=lower == "estimate",
        lower_value=0.0 if lower == "estimate" else float(lower),
        mirrored=False,
    )


def test_scan_runs_exactly_once_for_a_fit_that_needed_it(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_scans(monkeypatch)

    fit = _generic(
        "log",
        "cloglog",
        _GAP_SEVERITY,
        _GAP_SUCCESSES,
        _GAP_TRIALS,
        upper=0.95,
        lower="estimate",
    )

    assert fit.status is Status.OK, fit.warnings
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("scale", "link", "severity", "successes", "trials", "upper", "lower"),
    [
        ("linear", "logit", _R21_SEVERITY, _R21_SUCCESSES, [10] * 4, 0.95, 0.05),
        ("log", "probit", _R19_SEVERITY, _R19_SUCCESSES, [50] * 6, 0.95, 0.3),
        (
            "log",
            "logit",
            [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
            [50, 45, 42, 40, 37, 34, 30],
            [100] * 7,
            0.9,
            0.05,
        ),
    ],
    ids=["local-maximum-design", "step-plateau-design", "control-design"],
)
def test_scan_runs_exactly_once_for_a_fit_that_is_ok(
    monkeypatch: pytest.MonkeyPatch,
    scale: str,
    link: str,
    severity: list[float],
    successes: list[int],
    trials: list[int],
    upper: Any,
    lower: Any,
) -> None:
    calls = _count_scans(monkeypatch)

    fit = _generic(scale, link, severity, successes, trials, upper=upper, lower=lower)

    assert fit.status is Status.OK, fit.warnings
    assert len(calls) == 1


def test_scan_does_not_run_at_asymptotes_fixed_at_one_and_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A design refused as flat with both asymptotes fixed at (1, 0): the problem is concave, so
    # every local maximum is global and no scan is made.
    calls = _count_scans(monkeypatch)

    fit = _generic(
        "linear",
        "logit",
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5, 15, 5, 15, 10],
        [20] * 5,
        upper=1.0,
        lower=0.0,
    )

    assert fit.status is Status.NOT_CONVERGED
    assert calls == []


# --- the batch evaluator used by the search equals the per-point objective -----------------------

_ASYMPTOTE_CONFIGS: list[tuple[Any, Any]] = [
    ("estimate", "estimate"),
    ("estimate", 0.0),
    (1.0, "estimate"),
    (0.95, 0.05),
    (0.9, 0.3),
    (0.95, "estimate"),
]


def _objective(
    scale: str, link: str, config: tuple[Any, Any], severity: list[float]
) -> models_module._GenericLogLikelihood:
    upper, lower = config
    sev = np.asarray(severity, dtype=np.float64)
    center, spread = models_module._centering_and_scale(sev, scale=scale)
    return models_module._GenericLogLikelihood(
        severity=sev,
        successes=np.asarray([9.0, 7.0, 5.0, 2.0, 1.0][: sev.size], dtype=np.float64),
        trials=np.asarray([10.0] * sev.size, dtype=np.float64),
        scale=scale,
        link=link,
        estimate_lower=lower == "estimate",
        estimate_upper=upper == "estimate",
        lower_value=0.0 if lower == "estimate" else float(lower),
        upper_value=1.0 if upper == "estimate" else float(upper),
        center=center,
        spread=spread,
    )


def _parameter_grid(objective: models_module._GenericLogLikelihood) -> np.ndarray:
    # Locations at the level positions and in the tails (|z| up to 40 per s), very small and very
    # large scales, and asymptote logits from the middle to the extremes.
    transformed = models_module._transform(
        objective.severity[objective.severity > 0.0]
        if objective.scale == "log"
        else objective.severity,
        scale=objective.scale,
    )
    z_levels = (transformed - objective.center) / objective.spread
    log_s_values = (-14.0, -6.0, -1.0, 0.0, 2.5, 9.0, 20.0)
    tail_widths = (-40.0, -20.0, -3.0, -0.5, 0.0, 0.5, 3.0, 20.0, 40.0)
    logits = (-12.0, -2.0, 0.0, 2.0, 12.0)
    n_logits = int(objective.estimate_lower) + int(objective.estimate_upper)
    rows = []
    for log_s in log_s_values:
        s = math.exp(log_s)
        locations = [float(z) for z in z_levels] + [
            float(np.mean(z_levels) + s * w) for w in tail_widths
        ]
        for mu in locations:
            for extra in itertools.product(logits, repeat=n_logits):
                rows.append([mu, log_s, *extra])
    return np.asarray(rows, dtype=np.float64)


_PARITY_SEVERITIES = {
    "linear": [0.0, 1.0, 2.5, 4.0, 9.0],
    "log": [0.0, 1.0, 2.5, 4.0, 9.0],
}


@pytest.mark.parametrize("scale", ["linear", "log"])
@pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
@pytest.mark.parametrize("config", _ASYMPTOTE_CONFIGS, ids=str)
def test_batch_log_likelihood_equals_the_per_point_objective(
    scale: str, link: str, config: tuple[Any, Any]
) -> None:
    for severity in (_PARITY_SEVERITIES[scale], _PARITY_SEVERITIES[scale][1:]):
        # With and without a zero-severity level (a pinned control on the log axis).
        objective = _objective(scale, link, config, severity)
        grid = _parameter_grid(objective)

        with np.errstate(all="ignore"):
            batch = objective.batch_log_likelihood(grid)
            single = np.asarray([objective(row) for row in grid])

        assert batch.shape == (grid.shape[0],)
        finite = np.isfinite(single)
        assert np.array_equal(finite, np.isfinite(batch))
        assert np.all(
            np.abs(batch[finite] - single[finite]) <= 1e-10 + 1e-12 * np.abs(single[finite])
        )


def test_batch_log_likelihood_of_a_single_row_has_one_value() -> None:
    objective = _objective("log", "logit", ("estimate", 0.05), [0.0, 1.0, 2.5, 4.0, 9.0])
    row = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64)

    batch = objective.batch_log_likelihood(row)

    assert batch.shape == (1,)
    assert batch[0] == pytest.approx(objective(row[0]), abs=1e-10)


# --- the local-maximum rule moved to the model module --------------------------------------------


def _intervals_top_local_maxima(profile: Any) -> list[int]:
    from robustmargin import intervals

    return list(intervals._top_local_maxima(profile))


@pytest.mark.parametrize(
    "profile",
    [
        [0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0],  # more maxima than the cap
        [1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 0.0],  # plateaus are one maximum each
        [5.0, 1.0, 2.0, 3.0],  # a maximum at either end
        [1.0, 2.0, 3.0, 4.0],  # monotone rising
        [4.0, 3.0, 2.0, 1.0],  # monotone falling
        [np.nan, 1.0, np.inf, 0.5, -np.inf, 2.0, np.nan],  # non-finite entries count as -inf
        [3.0, 3.0, 3.0],  # constant
        [np.nan, np.nan],  # nothing finite
        [2.0, 1.0, 2.0, 1.0, 2.0],  # ties ordered by index
        [7.0],
    ],
    ids=[
        "cap",
        "plateaus",
        "ends",
        "rising",
        "falling",
        "non-finite",
        "constant",
        "all-non-finite",
        "ties",
        "single",
    ],
)
def test_top_local_maxima_in_models_with_count_four_matches_the_interval_rule(
    profile: list[float],
) -> None:
    array = np.asarray(profile, dtype=np.float64)

    assert list(models_module._top_local_maxima(array, 4)) == _intervals_top_local_maxima(array)


def test_top_local_maxima_in_models_honours_the_count() -> None:
    array = np.asarray([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0], dtype=np.float64)

    assert list(models_module._top_local_maxima(array, 2)) == [5, 3]
    assert list(models_module._top_local_maxima(array, 0)) == []


# --- a genuinely flat design keeps its refusal ---------------------------------------------------
#
# Linear axis, logit, upper 0.9, lower 0.3, rates about one half at every level. Independent
# oracle (Nelder-Mead over the interior, own numpy likelihood): interior maximum -10.578121,
# below the flat supremum -10.577563, so there is no interior maximum to find.

_TRULY_FLAT_SEVERITY = [1.0, 2.0, 4.0, 8.0, 16.0]
_TRULY_FLAT_SUCCESSES = [19, 19, 21, 20, 21]
_TRULY_FLAT_TRIALS = [40] * 5


def test_a_genuinely_flat_design_is_still_refused_as_flat_with_the_same_warnings() -> None:
    fit = _generic(
        "linear",
        "logit",
        _TRULY_FLAT_SEVERITY,
        _TRULY_FLAT_SUCCESSES,
        _TRULY_FLAT_TRIALS,
        upper=0.9,
        lower=0.3,
    )

    assert fit.status is Status.NOT_CONVERGED
    assert fit.params is None
    assert fit.log_likelihood is None
    assert tuple(fit.warnings) == (models_module._wrong_sign_or_flat_warning(mirrored=False),)


def test_a_genuinely_flat_design_runs_the_scan_once_and_stays_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _count_scans(monkeypatch)

    fit = _generic(
        "linear",
        "logit",
        _TRULY_FLAT_SEVERITY,
        _TRULY_FLAT_SUCCESSES,
        _TRULY_FLAT_TRIALS,
        upper=0.9,
        lower=0.3,
    )

    assert fit.status is Status.NOT_CONVERGED
    assert len(calls) == 1


def test_scan_runs_exactly_once_for_a_boundary_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    # The estimated lower asymptote runs to zero (the last level has no successes): refused for
    # the boundary, not for a flat or step curve.
    calls = _count_scans(monkeypatch)

    fit = _generic(
        "linear",
        "probit",
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [28, 24, 19, 12, 6, 0],
        [30] * 6,
        upper=0.95,
        lower="estimate",
    )

    assert fit.status is Status.NOT_CONVERGED
    assert "asymptote reached the parameter boundary" in fit.warnings[0]
    assert len(calls) == 1


# --- an interior maximum on the OK path that a start inside the tested range misses ---------------
#
# Linear axis, probit, upper 0.95, lower estimated. Independent oracle (Nelder-Mead on the
# centred-and-scaled covariate, own numpy likelihood): maximum -67.594356. A fit that settles at
# -70.871766 is a local maximum.

_LOCAL_SEVERITY = [
    14.299959791393428,
    15.464441152713276,
    21.936092912297216,
    36.82439493273207,
    62.07312890366481,
    75.0685043940403,
]
_LOCAL_SUCCESSES = [24, 30, 1, 40, 1, 11]
_LOCAL_TRIALS = [39, 39, 34, 49, 46, 13]
_LOCAL_LLF = -67.594356


def test_ok_fit_at_a_local_maximum_moves_to_the_global_maximum_of_the_full_log_likelihood() -> None:
    obs = Observations.from_counts(
        _LINEAR,
        severity=_LOCAL_SEVERITY,
        successes=_LOCAL_SUCCESSES,
        trials=_LOCAL_TRIALS,
        outcome="success",
        direction="decreasing",
    )

    fit = fit_dose_response(obs, model="binomial", link="probit", upper=0.95, lower="estimate")

    assert fit.status is Status.OK, fit.warnings
    assert fit.log_likelihood is not None
    assert fit.log_likelihood >= _LOCAL_LLF - 1e-6


# --- an increasing outcome whose fit sat at a local maximum and broke the profile threshold -------
#
# Log axis, logit, upper 1.0, lower estimated, direction increasing. Without the log-likelihood
# scan the fit is OK at -20.825183 and a profile threshold at absolute(0.25) raises because the
# constrained fit exceeds the reported maximum. An independent multi-start oracle (decreasing
# frame, complement counts, upper estimated, lower 0) gives -15.957757288094086.

_INCREASING_LOCAL_SEVERITY = [
    2.1493569652877778,
    5.732817965660452,
    16.82073058851421,
    16.92798363449072,
    17.31976265313372,
]
_INCREASING_LOCAL_EVENTS = [9, 23, 37, 50, 64]
_INCREASING_LOCAL_LLF = -15.957757288094086


def _fit_increasing_local_maximum() -> Any:
    obs = Observations.from_counts(
        _LOG,
        severity=_INCREASING_LOCAL_SEVERITY,
        successes=_INCREASING_LOCAL_EVENTS,
        trials=[80] * 5,
        outcome="success",
        direction="increasing",
    )
    return fit_dose_response(obs, model="binomial", link="logit", upper=1.0, lower="estimate")


def test_increasing_ok_fit_at_a_local_maximum_reaches_the_oracle_log_likelihood() -> None:
    fit = _fit_increasing_local_maximum()

    assert fit.status is Status.OK, fit.warnings
    assert fit.log_likelihood is not None
    assert fit.log_likelihood >= _INCREASING_LOCAL_LLF - 1e-6


def test_increasing_local_maximum_fit_gives_an_ok_profile_threshold_not_a_value_error() -> None:
    fit = _fit_increasing_local_maximum()

    result = threshold(
        fit, definition=Definition.absolute(0.25), interval_method="profile", level=0.95
    )

    assert result.status is Status.OK, result.warnings


# --- a maximum barely above the step supremum ----------------------------------------------------
#
# The gap design with successes [8, 26, 14, 28, 2, 12]. Oracle interior maximum -23.405629; step
# supremum -23.405738; flat supremum -23.409279.


def test_shallow_interior_maximum_above_the_step_supremum_is_not_refused() -> None:
    obs = Observations.from_counts(
        _LOG,
        severity=_GAP_SEVERITY,
        successes=[8, 26, 14, 28, 2, 12],
        trials=_GAP_TRIALS,
        outcome="success",
        direction="decreasing",
    )

    fit = fit_dose_response(obs, model="binomial", link="cloglog", upper=0.95, lower="estimate")

    assert not _is_flat_or_step_refusal(fit), fit.warnings
    assert fit.status is Status.OK, fit.warnings
    assert fit.log_likelihood is not None
    assert fit.log_likelihood >= -23.405629 - 1e-6


# --- fits that were already at the global maximum are unchanged ----------------------------------
#
# The exact claim is the same-process comparison with the scan disabled. The pinned values were
# recorded from a fit made before the scan existed, on one platform, so they are a cross-check,
# not an exact oracle. The log-likelihood is flat at the maximum, so it agrees closely across
# platforms (1e-12); the parameters are determined only to the optimizer's convergence
# tolerance, so they agree to 1e-6. The covariance is the inverse of a finite-difference Hessian,
# whose roundoff alone moves it by a few parts in a million when the log-likelihood changes in
# its last bit, so it is checked to 1e-4.

_PINNED: dict[str, dict[str, Any]] = {
    "local-maximum-design": {
        "llf": -6.556780871935378,
        "params": {
            "mu": 89.51329900580501,
            "s": 60.71313047723067,
            "lower": 0.05,
            "upper": 0.95,
        },
        "names": ("mu", "s"),
        "matrix": (
            (537.8165931474387, 683.6712038983972),
            (683.6712038983972, 6569.282846491438),
        ),
    },
    "step-plateau-design": {
        "llf": -13.482347279540964,
        "params": {
            "mu": 2.3883970117463096,
            "s": 0.5326888922833916,
            "lower": 0.3,
            "upper": 0.95,
        },
        "names": ("mu", "s"),
        "matrix": (
            (0.28281765458541774, 0.022300110960383108),
            (0.022300110960383108, 0.06842808089683874),
        ),
    },
    "control-design": {
        "llf": -68.6353366107693,
        "params": {
            "mu": -0.7865549352262233,
            "s": 7.88349701052622,
            "lower": 0.05,
            "upper": 0.9,
        },
        "names": ("mu", "s"),
        "matrix": (
            (2.6126235004738505, -4.481664792269459),
            (-4.481664792269459, 10.029025609954138),
        ),
    },
}


@pytest.mark.parametrize(
    ("design", "scale", "link", "severity", "successes", "trials", "upper", "lower"),
    [
        (
            "local-maximum-design",
            "linear",
            "logit",
            _R21_SEVERITY,
            _R21_SUCCESSES,
            [10] * 4,
            0.95,
            0.05,
        ),
        (
            "step-plateau-design",
            "log",
            "probit",
            _R19_SEVERITY,
            _R19_SUCCESSES,
            [50] * 6,
            0.95,
            0.3,
        ),
        (
            "control-design",
            "log",
            "logit",
            [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
            [50, 45, 42, 40, 37, 34, 30],
            [100] * 7,
            0.9,
            0.05,
        ),
    ],
)
def test_a_fit_already_at_the_global_maximum_is_bit_identical_with_and_without_the_scan(
    monkeypatch: pytest.MonkeyPatch,
    design: str,
    scale: str,
    link: str,
    severity: list[float],
    successes: list[int],
    trials: list[int],
    upper: Any,
    lower: Any,
) -> None:
    obs = Observations.from_counts(
        _LOG if scale == "log" else _LINEAR,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )
    pinned = _PINNED[design]

    fit = fit_dose_response(obs, model="binomial", link=link, upper=upper, lower=lower)
    with monkeypatch.context() as patch:
        patch.setattr(models_module, "_scan_starts", lambda *a, **k: [])
        unscanned = fit_dose_response(obs, model="binomial", link=link, upper=upper, lower=lower)

    assert fit.status is Status.OK
    assert fit.status is unscanned.status
    assert tuple(fit.warnings) == ()
    assert tuple(fit.warnings) == tuple(unscanned.warnings)
    assert fit.log_likelihood == unscanned.log_likelihood
    assert fit.params.keys() == unscanned.params.keys()
    for name, param in fit.params.items():
        assert param.value == unscanned.params[name].value
        assert param.fixed is unscanned.params[name].fixed
    assert fit.covariance.names == unscanned.covariance.names
    assert fit.covariance.matrix == unscanned.covariance.matrix

    assert fit.log_likelihood == pytest.approx(pinned["llf"], rel=1e-12)
    assert {name: value.value for name, value in fit.params.items()} == pytest.approx(
        pinned["params"], rel=1e-6
    )
    assert fit.covariance.names == pinned["names"]
    assert np.asarray(fit.covariance.matrix) == pytest.approx(
        np.asarray(pinned["matrix"]), rel=1e-4
    )


def _step(upper: float, lower: float) -> models_module._StepSupremum:
    return models_module._StepSupremum(0.0, False, upper, lower)


def _chart_asymptotes(
    objective: models_module._GenericLogLikelihood, lower_share: float, upper_share: float
) -> tuple[float, float]:
    """The (lower, upper) asymptotes the objective's own transform gives for the shares."""
    params = [0.0, 0.0]
    if objective.estimate_lower:
        params.append(math.log(lower_share / (1.0 - lower_share)))
    if objective.estimate_upper:
        params.append(math.log(upper_share / (1.0 - upper_share)))
    _, _, lower, upper = models_module._unpack_optimizer_params(
        np.asarray(params, dtype=np.float64),
        estimate_lower=objective.estimate_lower,
        estimate_upper=objective.estimate_upper,
        lower_value=objective.lower_value,
        upper_value=objective.upper_value,
    )
    return lower, upper


@pytest.mark.parametrize(
    ("config", "step_upper", "step_lower"),
    [
        (("estimate", "estimate"), 0.8, 0.2),
        (("estimate", 0.2), 0.8, 0.2),
        ((0.9, "estimate"), 0.9, 0.3),
    ],
)
def test_step_shares_round_trip_through_the_optimizer_chart(
    config: tuple[Any, Any], step_upper: float, step_lower: float
) -> None:
    objective = _objective("linear", "logit", config, [1.0, 2.0, 3.0, 4.0])
    if not objective.estimate_upper:
        step_upper = objective.upper_value
    if not objective.estimate_lower:
        step_lower = objective.lower_value
    step = _step(step_upper, step_lower)

    lower_share, upper_share = models_module._step_shares(objective, step)
    lower, upper = _chart_asymptotes(objective, lower_share, upper_share)

    assert lower == pytest.approx(step_lower, rel=1e-12, abs=1e-12)
    assert upper == pytest.approx(step_upper, rel=1e-12, abs=1e-12)


def test_step_shares_lower_only_chart_is_the_fraction_of_the_fixed_upper() -> None:
    objective = _objective("linear", "logit", (0.9, "estimate"), [1.0, 2.0, 3.0, 4.0])
    assert objective.estimate_lower
    assert not objective.estimate_upper

    lower_share, upper_share = models_module._step_shares(objective, _step(0.9, 0.3))

    assert lower_share == pytest.approx(0.3 / 0.9)
    assert math.isnan(upper_share)


def test_step_shares_upper_only_chart_is_the_fraction_of_the_range_above_the_fixed_lower() -> None:
    objective = _objective("linear", "logit", ("estimate", 0.2), [1.0, 2.0, 3.0, 4.0])
    assert objective.estimate_upper
    assert not objective.estimate_lower

    lower_share, upper_share = models_module._step_shares(objective, _step(0.8, 0.2))

    assert math.isnan(lower_share)
    assert upper_share == pytest.approx((0.8 - 0.2) / (1.0 - 0.2))


@pytest.mark.parametrize(
    ("config", "step_upper", "step_lower"),
    [
        (("estimate", "estimate"), 1.0, 1.0),
        (("estimate", "estimate"), 0.5, 0.0),
        (("estimate", "estimate"), 0.0, 0.0),
        (("estimate", "estimate"), 0.7, 0.7),
        ((0.9, "estimate"), 0.9, 0.0),
        ((0.9, "estimate"), 0.9, 0.9),
        (("estimate", 0.0), 0.0, 0.0),
        (("estimate", 0.2), 1.0, 1.0),
        (("estimate", "estimate"), math.nan, math.nan),
    ],
)
def test_step_share_edge_cases_give_nan_or_a_finite_clipped_logit(
    config: tuple[Any, Any], step_upper: float, step_lower: float
) -> None:
    objective = _objective("linear", "logit", config, [1.0, 2.0, 3.0, 4.0])

    shares = models_module._step_shares(objective, _step(step_upper, step_lower))

    for share in shares:
        logits = models_module._share_logits(0.5, (share,))
        assert all(math.isfinite(value) for value in logits)
        if not math.isfinite(share):
            assert logits == models_module._share_logits(0.5)


def test_step_shares_undefined_asymptotes_are_nan_and_add_no_logit() -> None:
    objective = _objective("linear", "logit", ("estimate", "estimate"), [1.0, 2.0, 3.0, 4.0])

    shares = models_module._step_shares(objective, _step(math.nan, math.nan))

    assert all(math.isnan(share) for share in shares)
    for share in shares:
        assert models_module._share_logits(0.4, (share,)) == models_module._share_logits(0.4)


# --- the asymptote rows of the scan ------------------------------------------------------------


def _scan_batches(
    monkeypatch: pytest.MonkeyPatch,
    objective: models_module._GenericLogLikelihood,
    selected_params: list[float],
    step: models_module._StepSupremum,
) -> list[np.ndarray]:
    """The rows `_scan_starts` hands the objective, one array per log(s) of its grid."""
    batches: list[np.ndarray] = []
    real = models_module._GenericLogLikelihood.batch_log_likelihood

    def spy(self: Any, params: Any) -> Any:
        batches.append(np.array(params, dtype=np.float64))
        return real(self, params)

    monkeypatch.setattr(models_module._GenericLogLikelihood, "batch_log_likelihood", spy)
    models_module._scan_starts(objective, np.asarray(selected_params, dtype=np.float64), step)
    assert batches
    return batches


def _pairs(batch: np.ndarray) -> set[tuple[float, ...]]:
    return {tuple(round(float(v), 9) for v in row) for row in batch[:, 2:]}


def _logit_of_share(share: float) -> float:
    edge = models_module._FIT_SHARE_EDGE
    return float(models_module._logit(float(np.clip(share, edge, 1.0 - edge))))


_FIT_LOWER_SHARE = 0.35
_FIT_UPPER_SHARE = 0.65


def _both_estimated_expected(
    step: models_module._StepSupremum,
) -> tuple[list[float], list[float], list[float]]:
    objective = _objective("linear", "logit", ("estimate", "estimate"), [1.0, 2.0, 3.0, 4.0])
    lower_share, upper_share = models_module._step_shares(objective, step)
    lower_logits = models_module._share_logits(_FIT_LOWER_SHARE)
    upper_logits = models_module._share_logits(_FIT_UPPER_SHARE)
    return lower_logits, upper_logits, [lower_share, upper_share]


def _both_selected() -> list[float]:
    return [
        0.0,
        0.0,
        float(models_module._logit(_FIT_LOWER_SHARE)),
        float(models_module._logit(_FIT_UPPER_SHARE)),
    ]


def test_scan_rows_with_both_asymptotes_estimated_are_the_product_plus_the_three_step_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = _objective("linear", "logit", ("estimate", "estimate"), [1.0, 2.0, 3.0, 4.0])
    step = _step(0.9, 0.25)
    lower_logits, upper_logits, (lower_share, upper_share) = _both_estimated_expected(step)
    a_step, b_step = _logit_of_share(lower_share), _logit_of_share(upper_share)
    expected = {
        tuple(round(v, 9) for v in pair) for pair in itertools.product(lower_logits, upper_logits)
    }
    assert len(expected) == 144
    expected |= {
        (round(a_step, 9), round(b_step, 9)),
        (round(a_step, 9), round(upper_logits[0], 9)),
        (round(lower_logits[0], 9), round(b_step, 9)),
    }
    assert len(expected) == 147

    for batch in _scan_batches(monkeypatch, objective, _both_selected(), step):
        assert _pairs(batch) == expected


def test_scan_rows_with_both_asymptotes_estimated_include_the_step_step_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = _objective("linear", "logit", ("estimate", "estimate"), [1.0, 2.0, 3.0, 4.0])
    step = _step(0.9, 0.25)
    _, _, (lower_share, upper_share) = _both_estimated_expected(step)
    step_row = (round(_logit_of_share(lower_share), 9), round(_logit_of_share(upper_share), 9))

    for batch in _scan_batches(monkeypatch, objective, _both_selected(), step):
        assert step_row in _pairs(batch)


def test_scan_rows_with_the_upper_step_share_undefined_add_only_the_step_and_fit_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = _objective("linear", "logit", ("estimate", "estimate"), [1.0, 2.0, 3.0, 4.0])
    step = _step(math.nan, 0.25)
    lower_logits, upper_logits, (lower_share, upper_share) = _both_estimated_expected(step)
    assert math.isfinite(lower_share)
    assert math.isnan(upper_share)
    expected = {
        tuple(round(v, 9) for v in pair) for pair in itertools.product(lower_logits, upper_logits)
    }
    expected.add((round(_logit_of_share(lower_share), 9), round(upper_logits[0], 9)))
    assert len(expected) == 145

    for batch in _scan_batches(monkeypatch, objective, _both_selected(), step):
        assert _pairs(batch) == expected


def test_scan_rows_with_every_step_share_undefined_are_the_product_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = _objective("linear", "logit", ("estimate", "estimate"), [1.0, 2.0, 3.0, 4.0])
    step = _step(0.9, math.nan)
    lower_logits, upper_logits, shares = _both_estimated_expected(step)
    assert all(math.isnan(share) for share in shares)
    expected = {
        tuple(round(v, 9) for v in pair) for pair in itertools.product(lower_logits, upper_logits)
    }

    for batch in _scan_batches(monkeypatch, objective, _both_selected(), step):
        assert _pairs(batch) == expected


def test_scan_rows_with_one_asymptote_estimated_add_the_step_share_to_that_coordinate_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = _objective("linear", "logit", (0.9, "estimate"), [1.0, 2.0, 3.0, 4.0])
    step = _step(0.9, 0.3)
    (step_share, _) = models_module._step_shares(objective, step)
    fit_share = 0.35
    logits = models_module._share_logits(fit_share, (step_share,))
    expected = {(round(v, 9),) for v in logits}
    assert len(expected) == 13
    assert (round(_logit_of_share(step_share), 9),) in expected
    selected = [0.0, 0.0, float(models_module._logit(fit_share))]

    for batch in _scan_batches(monkeypatch, objective, selected, step):
        assert batch.shape[1] == 3
        assert _pairs(batch) == expected


def test_scan_first_row_at_each_log_s_carries_the_selected_results_own_shares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = _objective("linear", "logit", ("estimate", "estimate"), [1.0, 2.0, 3.0, 4.0])
    lower_logits, upper_logits, _ = _both_estimated_expected(_step(0.9, 0.25))
    first = (lower_logits[0], upper_logits[0])
    assert first == pytest.approx(
        (
            float(models_module._logit(_FIT_LOWER_SHARE)),
            float(models_module._logit(_FIT_UPPER_SHARE)),
        )
    )

    for batch in _scan_batches(monkeypatch, objective, _both_selected(), _step(0.9, 0.25)):
        assert tuple(batch[0, 2:]) == pytest.approx(first)
