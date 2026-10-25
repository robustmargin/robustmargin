"""The step-plateau guard of the estimated-asymptote path (``_fit_generic``).

When the data are a step between adjacent tested levels, the log-likelihood has no finite maximum
in ``s``: it rises monotonically as ``s -> 0`` toward a closed-form supremum. Such a fit is
``NOT_CONVERGED`` with its own reason, whatever the optimizer happened to report on the platform.

``_step_supremum_log_likelihood`` is the supremum of the *full* binomial log-likelihood (the
``lchoose`` term included, the same basis as ``Fit.log_likelihood``) on the ``s -> 0`` boundary.
A monotone curve as ``s -> 0`` is a step at one transition, so:

* levels are ordered by severity; for a free level ``j`` sitting on the transition, the levels in
  ``[0, j)`` have ``p = u``, the levels in ``(j, m)`` have ``p = l``, and level ``j`` has ``p``
  equal to its own rate clipped to ``[l, u]``; splits with no free level are enumerated too, and
  a free level that is not at the transition is not reachable;
* a log-axis control (severity 0) is never the free level; it always takes ``u``; an end level may
  be free only when the asymptote of its empty side is fixed;
* an estimated asymptote is its side's pooled rate; with ``u`` fixed an estimated ``l`` is
  clipped to at most ``u``; with ``l`` fixed an estimated ``u`` is clipped to at least ``l``; with
  both estimated and ``l > u`` both become the pooled rate of all levels;
* the supremum is the maximum over those configurations; fewer than two levels give ``-inf``.

Whether an independent multi-start optimisation of the stable log-likelihood finds a maximum above
that supremum decides every "interior maximum" claim below; the designs carry the supremum they
were checked against as a literal.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from statsmodels.base.model import GenericLikelihoodModel

from robustmargin import Axis, Observations, Status, fit_dose_response
from robustmargin import models as models_module

_STEP_REASON = "a step between adjacent tested levels"
_AXIS = Axis(name="severity", unit="unit", scale="log")
_SEVERITY_7 = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def _obs(severity: list[float], successes: list[int], trials: list[int]) -> Observations:
    return Observations.from_counts(
        _AXIS,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )


def _lchoose_total(successes: list[int], trials: list[int]) -> float:
    return sum(
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        for k, n in zip(successes, trials, strict=True)
    )


def _supremum(
    severity: list[float],
    successes: list[int],
    trials: list[int],
    *,
    estimate_lower: bool,
    estimate_upper: bool,
    lower_value: float = 0.0,
    upper_value: float = 1.0,
) -> float:
    value, _ = models_module._step_supremum_log_likelihood(
        np.asarray(severity, dtype=np.float64),
        np.asarray(successes, dtype=np.float64),
        np.asarray(trials, dtype=np.float64),
        estimate_lower=estimate_lower,
        estimate_upper=estimate_upper,
        lower_value=lower_value,
        upper_value=upper_value,
        scale="log",
    )
    return float(value)


# --- the pure function, pinned on hand-computed designs -----------------------------------------


def test_supremum_splits_at_the_pooled_rates_when_both_asymptotes_are_estimated() -> None:
    successes, trials = [9, 9, 3, 3], [10, 10, 10, 10]

    value = _supremum(
        [1.0, 2.0, 3.0, 4.0], successes, trials, estimate_lower=True, estimate_upper=True
    )

    # Baseline side 18/20 at u = 0.9, far side 6/20 at l = 0.3.
    expected = (
        _lchoose_total(successes, trials)
        + 18 * math.log(0.9)
        + 2 * math.log(0.1)
        + 6 * math.log(0.3)
        + 14 * math.log(0.7)
    )
    assert value == pytest.approx(expected, abs=1e-9)


def test_supremum_is_independent_of_the_order_the_levels_are_given_in() -> None:
    successes, trials = [9, 9, 3, 3], [10, 10, 10, 10]
    ordered = _supremum(
        [1.0, 2.0, 3.0, 4.0], successes, trials, estimate_lower=True, estimate_upper=True
    )

    shuffled = _supremum(
        [3.0, 1.0, 4.0, 2.0], [3, 9, 3, 9], trials, estimate_lower=True, estimate_upper=True
    )

    assert shuffled == pytest.approx(ordered, abs=1e-12)


def test_supremum_keeps_fixed_asymptotes_at_their_values() -> None:
    successes, trials = [9, 9, 3, 3], [10, 10, 10, 10]

    value = _supremum(
        [1.0, 2.0, 3.0, 4.0],
        successes,
        trials,
        estimate_lower=False,
        estimate_upper=False,
        lower_value=0.4,
        upper_value=0.8,
    )

    expected = (
        _lchoose_total(successes, trials)
        + 18 * math.log(0.8)
        + 2 * math.log(0.2)
        + 6 * math.log(0.4)
        + 14 * math.log(0.6)
    )
    assert value == pytest.approx(expected, abs=1e-9)


def test_supremum_lets_one_level_sit_on_the_step_at_its_own_rate() -> None:
    successes, trials = [10, 5, 0], [10, 10, 10]

    value = _supremum([1.0, 2.0, 3.0], successes, trials, estimate_lower=True, estimate_upper=True)

    # u = 1 and l = 0 from the end levels; the middle level is free at p = 0.5.
    expected = _lchoose_total(successes, trials) + 10 * math.log(0.5)
    assert value == pytest.approx(expected, abs=1e-9)


def test_supremum_clips_the_free_level_rate_into_the_fixed_asymptote_range() -> None:
    successes, trials = [10, 9, 1, 0], [10, 10, 10, 10]

    value = _supremum(
        [1.0, 2.0, 3.0, 4.0],
        successes,
        trials,
        estimate_lower=False,
        estimate_upper=False,
        lower_value=0.2,
        upper_value=0.9,
    )

    # Splitting after level 1: u side 19/20 at 0.9, far side 1/20 at 0.2. Freeing level 1 clips
    # its rate 0.9 into [0.2, 0.9] (no change), so the plain split is the maximum.
    expected = (
        _lchoose_total(successes, trials)
        + 19 * math.log(0.9)
        + 1 * math.log(0.1)
        + 1 * math.log(0.2)
        + 19 * math.log(0.8)
    )
    assert value == pytest.approx(expected, abs=1e-9)


def test_supremum_lets_an_end_level_be_free_when_its_asymptote_is_fixed() -> None:
    successes, trials = [7, 2, 2], [10, 10, 10]

    value = _supremum(
        [1.0, 2.0, 3.0],
        successes,
        trials,
        estimate_lower=False,
        estimate_upper=False,
        lower_value=0.2,
        upper_value=0.9,
    )

    # The first level sits on the step at p = 0.7; the other two are at l = 0.2. Putting it on
    # the u side instead (p = 0.9) is worse.
    expected = (
        _lchoose_total(successes, trials)
        + 7 * math.log(0.7)
        + 3 * math.log(0.3)
        + 4 * math.log(0.2)
        + 16 * math.log(0.8)
    )
    assert value == pytest.approx(expected, abs=1e-9)


def test_supremum_forces_equal_asymptotes_when_the_far_side_rate_is_higher() -> None:
    successes, trials = [2, 8], [10, 10]

    value = _supremum([1.0, 2.0], successes, trials, estimate_lower=True, estimate_upper=True)

    # l <= u: the constrained maximum is the constant curve at the pooled rate 0.5.
    assert value == pytest.approx(_lchoose_total(successes, trials) + 20 * math.log(0.5), abs=1e-9)


def test_supremum_clips_an_estimated_lower_asymptote_to_a_fixed_upper_asymptote() -> None:
    successes, trials = [2, 9], [10, 10]

    value = _supremum(
        [1.0, 2.0],
        successes,
        trials,
        estimate_lower=True,
        estimate_upper=False,
        upper_value=0.5,
    )

    # Far-side pooled rate 0.9 exceeds the fixed u = 0.5, so l = u = 0.5 for both levels.
    expected = _lchoose_total(successes, trials) + 11 * math.log(0.5) + 9 * math.log(0.5)
    assert value == pytest.approx(expected, abs=1e-9)


def test_supremum_of_a_single_level_is_minus_infinity() -> None:
    value = _supremum([1.0], [5], [10], estimate_lower=True, estimate_upper=True)

    assert value == -math.inf


# --- the guard through fit_dose_response ---------------------------------------------------------


def _assert_step_plateau(fit: Any) -> None:
    assert fit.status is Status.NOT_CONVERGED
    assert fit.params is None
    assert fit.covariance is None
    assert fit.log_likelihood is None
    assert any(_STEP_REASON in w for w in fit.warnings), fit.warnings


@pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
def test_exact_step_with_estimated_asymptotes_is_reported_as_a_step_plateau(link: str) -> None:
    obs = _obs(_SEVERITY_7, [190, 190, 190, 190, 10, 10, 10], [200] * 7)

    fit = fit_dose_response(obs, model="binomial", link=link, upper="estimate", lower="estimate")

    _assert_step_plateau(fit)


@pytest.mark.parametrize("link", ["probit", "logit"])
def test_step_with_one_intermediate_level_is_reported_as_a_step_plateau(link: str) -> None:
    obs = _obs(_SEVERITY_7, [190, 190, 190, 100, 10, 10, 10], [200] * 7)

    fit = fit_dose_response(obs, model="binomial", link=link, upper="estimate", lower="estimate")

    _assert_step_plateau(fit)


def test_step_with_fixed_asymptotes_other_than_one_and_zero_is_reported_as_a_step_plateau() -> None:
    obs = _obs([1.0, 2.0, 3.0, 4.0], [90, 90, 30, 30], [100] * 4)

    fit = fit_dose_response(obs, model="binomial", link="cloglog", upper=0.9, lower=0.3)

    _assert_step_plateau(fit)


@pytest.mark.parametrize(
    ("link", "severity", "successes", "trials", "kwargs"),
    [
        pytest.param(
            "probit",
            [1.0, 2.0, 3.0, 4.0],
            [100, 100, 30, 30],
            [100] * 4,
            {"upper": 0.95, "lower": 0.25},
            id="fixed_probit",
        ),
        pytest.param(
            "cloglog",
            [1.0, 3.0513, 9.3107, 28.4101, 86.6888],
            [39, 37, 40, 28, 7],
            [50] * 5,
            {"upper": "estimate", "lower": "estimate"},
            id="estimated_cloglog",
        ),
        pytest.param(
            "cloglog",
            [1.0, 28.808, 829.8996],
            [1000, 146, 157],
            [1000] * 3,
            {"upper": 0.95, "lower": 0.1},
            id="fixed_cloglog_three_levels",
        ),
    ],
)
def test_a_step_with_a_free_intermediate_level_is_a_plateau_not_a_maximum(
    link: str,
    severity: list[float],
    successes: list[int],
    trials: list[int],
    kwargs: dict[str, Any],
) -> None:
    # Independent multi-start optimisation of the stable log-likelihood finds no more than
    # 1e-12 nats above the adjacent-level s -> 0 supremum on each of these: there is no interior
    # maximum.
    obs = _obs(severity, successes, trials)

    fit = fit_dose_response(obs, model="binomial", link=link, **kwargs)

    _assert_step_plateau(fit)


def _patch_newton(monkeypatch: pytest.MonkeyPatch, transform: Any) -> None:
    real_fit = GenericLikelihoodModel.fit

    def fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = real_fit(self, *args, **kwargs)
        return transform(result) if kwargs.get("method") == "newton" else result

    monkeypatch.setattr(GenericLikelihoodModel, "fit", fit)


def test_step_plateau_is_reported_even_when_newton_claims_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def converged(result: Any) -> Any:
        result.mle_retvals["converged"] = True
        return result

    _patch_newton(monkeypatch, converged)
    obs = _obs(_SEVERITY_7, [190, 190, 190, 190, 10, 10, 10], [200] * 7)

    fit = fit_dose_response(obs, model="binomial", link="logit", upper="estimate", lower="estimate")

    _assert_step_plateau(fit)


# --- negative controls: designs with an interior maximum stay OK ---------------------------------


def _form_b() -> Observations:
    trials = [141, 153, 142, 139, 154, 155]
    dead = [2, 30, 59, 82, 62, 85]
    return Observations.from_counts(
        Axis(name="concentration", unit="ug/L", scale="log"),
        severity=[0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        successes=[n - d for n, d in zip(trials, dead, strict=True)],
        trials=trials,
        outcome="survival",
        direction="decreasing",
    )


def test_smooth_design_with_estimated_upper_asymptote_stays_ok() -> None:
    # Independent multi-start maximum -25.3684; the s -> 0 supremum is -92.4295.
    fit = fit_dose_response(_form_b(), model="binomial", link="probit", upper="estimate", lower=0.0)

    assert fit.status is Status.OK
    assert fit.log_likelihood is not None
    assert fit.log_likelihood > -92.4295 + 1.0
    assert not any(_STEP_REASON in w for w in fit.warnings)


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_smooth_design_with_both_asymptotes_estimated_stays_ok(link: str) -> None:
    # Independent multi-start maxima -17.0925 (logit) and -16.9446 (probit); the s -> 0
    # supremum is -50.1517.
    obs = _obs(_SEVERITY_7, [190, 186, 170, 120, 60, 20, 10], [200] * 7)

    fit = fit_dose_response(obs, model="binomial", link=link, upper="estimate", lower="estimate")

    assert fit.status is Status.OK
    assert fit.log_likelihood is not None
    assert fit.log_likelihood > -50.1517 + 1.0


@pytest.mark.parametrize("link", ["probit", "cloglog"])
def test_smooth_design_with_fixed_asymptotes_stays_ok(link: str) -> None:
    # Independent multi-start maxima -19.4407 (probit) and -22.4165 (cloglog); the s -> 0
    # supremum is -66.6797.
    obs = _obs([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0], [190, 170, 140, 100, 70, 55, 50], [200] * 7)

    fit = fit_dose_response(obs, model="binomial", link=link, upper=0.95, lower=0.25)

    assert fit.status is Status.OK
    assert fit.log_likelihood is not None
    assert fit.log_likelihood > -66.6797 + 1.0


_BOUNDARY_REASONS = (
    "the estimated upper asymptote reached the parameter boundary",
    "the estimated upper asymptote reached the probability boundary",
)


@pytest.mark.parametrize(
    ("link", "severity", "successes", "trials", "kwargs", "supremum"),
    [
        pytest.param(
            "probit",
            [0.0, 1.0, 3.1173, 9.7179, 30.2939, 94.4367, 294.3922],
            [34, 28, 33, 25, 17, 16, 16],
            [50] * 7,
            {"upper": 0.9, "lower": "estimate"},
            -33.3068,
            id="probit_estimated_lower_with_control",
        ),
        pytest.param(
            "probit",
            [1.0, 2.1462, 4.6062, 9.886, 21.2174],
            [43, 9, 14, 10, 9],
            [50] * 5,
            {"upper": 0.95, "lower": 0.1},
            -21.2812,
            id="probit_fixed_asymptotes",
        ),
        pytest.param(
            "logit",
            [1.0, 2.3463, 5.5053, 12.9172, 30.308],
            [916, 842, 668, 486, 319],
            [1000] * 5,
            {"upper": "estimate", "lower": "estimate"},
            -59.5769,
            id="logit_both_estimated",
        ),
        pytest.param(
            "logit",
            [1.0, 2.7945, 7.8093, 21.8232, 60.9851, 170.4236],
            [869, 802, 738, 646, 533, 428],
            [1000] * 6,
            {"upper": "estimate", "lower": 0.1},
            -195.4088,
            id="logit_estimated_upper_fixed_lower",
        ),
    ],
)
def test_design_with_an_interior_maximum_above_the_step_supremum_stays_ok(
    link: str,
    severity: list[float],
    successes: list[int],
    trials: list[int],
    kwargs: dict[str, Any],
    supremum: float,
) -> None:
    # On the log axis an independent multi-start maximisation finds -26.4014, -19.8032, -17.9295
    # and -21.6375 (in the order above), each well above the supremum given.
    obs = Observations.from_counts(
        _AXIS,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )

    fit = fit_dose_response(obs, model="binomial", link=link, **kwargs)

    assert fit.status is Status.OK, fit.warnings
    assert fit.log_likelihood is not None
    assert fit.log_likelihood > supremum + 1.0
    assert not any(_STEP_REASON in w for w in fit.warnings)


def test_local_maximum_below_the_step_supremum_is_still_a_step_plateau() -> None:
    # The only finite-slope stationary point has log-likelihood 3.075 nats below the supremum
    # -45.26 (the last level, 0 of 50, sits at l = 0.1 on the step); the step is the better fit.
    obs = _obs(
        [1.0, 3.05, 9.3025, 28.3726, 86.5365, 263.9362],
        [39, 38, 42, 40, 38, 0],
        [50] * 6,
    )

    fit = fit_dose_response(obs, model="binomial", link="probit", upper=0.95, lower=0.1)

    _assert_step_plateau(fit)


def test_boundary_reason_takes_precedence_over_the_step_reason() -> None:
    obs = _obs(
        [1.0, 2.5746, 6.6286, 17.066, 43.9382, 113.1234],
        [5, 5, 5, 5, 2, 2],
        [5] * 6,
    )

    fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.1)

    assert fit.status is Status.NOT_CONVERGED
    text = " ".join(fit.warnings)
    assert any(reason in text for reason in _BOUNDARY_REASONS), text
    assert _STEP_REASON not in text


def test_step_reason_is_worded_as_an_observation() -> None:
    obs = _obs(_SEVERITY_7, [190, 190, 190, 190, 10, 10, 10], [200] * 7)

    fit = fit_dose_response(
        obs, model="binomial", link="probit", upper="estimate", lower="estimate"
    )

    assert any(
        "fit_dose_response: no fitted curve exceeded the log-likelihood of a step between "
        "adjacent tested levels (s -> 0) -- the data do not identify the slope" in w
        for w in fit.warnings
    )


# --- wrong-sign or flat data keeps its own reason ------------------------------------------------


@pytest.mark.parametrize(
    ("link", "scale", "severity", "successes", "trials"),
    [
        pytest.param(
            "probit",
            "log",
            [1.0, 3.453, 11.9232, 41.1709, 142.1631, 490.8894],
            [0, 2, 3, 1, 3, 1],
            [5] * 6,
            id="probit_noise_around_a_low_rate",
        ),
        pytest.param(
            "logit",
            "linear",
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10, 20, 30, 40, 45],
            [50] * 5,
            id="logit_rising",
        ),
        pytest.param(
            "probit",
            "log",
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [20, 22, 25, 30, 40],
            [50] * 5,
            id="probit_rising",
        ),
    ],
)
def test_wrong_sign_or_flat_data_keeps_its_reason_not_the_step_reason(
    link: str, scale: str, severity: list[float], successes: list[int], trials: list[int]
) -> None:
    obs = Observations.from_counts(
        Axis(name="severity", unit="unit", scale=scale),
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )

    fit = fit_dose_response(obs, model="binomial", link=link, upper="estimate", lower="estimate")

    assert fit.status is Status.NOT_CONVERGED
    assert fit.warnings[0] == models_module._wrong_sign_or_flat_warning(mirrored=False)
    assert not any(_STEP_REASON in w for w in fit.warnings), fit.warnings


def test_nearly_flat_data_with_a_tiny_real_step_is_flat_or_step_not_converged() -> None:
    # The supremum is a genuine tiny step (u about 0.607, l = 0.600, -10.8522) that no finite-s
    # curve exceeds (independent multi-start maximum -10.85222), so both reasons are true;
    # which one is reported depends on where the optimizer stops.
    obs = _obs([1.0, 2.0, 4.0, 8.0, 16.0], [30, 30, 31, 30, 30], [50] * 5)

    fit = fit_dose_response(obs, model="binomial", link="logit", upper="estimate", lower="estimate")

    assert fit.status is Status.NOT_CONVERGED
    assert (
        fit.warnings[0] == models_module._wrong_sign_or_flat_warning(mirrored=False)
        or _STEP_REASON in (fit.warnings[0])
    )


# --- a non-finite Newton result still gets the boundary check on the BFGS parameters -------------


def test_nonfinite_newton_result_with_bfgs_at_an_asymptote_boundary_gives_the_boundary_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On this design BFGS stops at a logit-scale upper asymptote of about 15.18 (> 15).
    def broken(result: Any) -> Any:
        return SimpleNamespace(
            params=np.full(np.asarray(result.params).shape, np.nan),
            llf=float("nan"),
            mle_retvals={"converged": True},
            cov_params=result.cov_params,
        )

    _patch_newton(monkeypatch, broken)
    obs = _obs([1.0, 2.5746, 6.6286, 17.066, 43.9382, 113.1234], [5, 5, 5, 5, 2, 2], [5] * 6)

    fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.1)

    assert fit.status is Status.NOT_CONVERGED
    text = " ".join(fit.warnings)
    assert any(reason in text for reason in _BOUNDARY_REASONS), text
    assert _STEP_REASON not in text


# --- a non-finite Newton result ------------------------------------------------------------------


@pytest.mark.parametrize("nonfinite", ["params", "llf"])
def test_nonfinite_newton_result_reports_the_estimate_is_not_finite_not_the_hessian(
    monkeypatch: pytest.MonkeyPatch, nonfinite: str
) -> None:
    def broken(result: Any) -> Any:
        params = np.asarray(result.params, dtype=np.float64).copy()
        llf = float(result.llf)
        if nonfinite == "params":
            params[:] = np.nan
        else:
            llf = float("nan")
        return SimpleNamespace(
            params=params,
            llf=llf,
            mle_retvals={"converged": True},
            cov_params=result.cov_params,
        )

    _patch_newton(monkeypatch, broken)

    fit = fit_dose_response(_form_b(), model="binomial", link="probit", upper="estimate", lower=0.0)

    assert fit.status is Status.NOT_CONVERGED
    assert fit.params is None
    text = " ".join(fit.warnings)
    assert "not finite" in text
    assert "Hessian" not in text
