"""The ``step`` and ``flat`` rules of the Gaussian fit, and the fit-must-beat-the-start-grid rule.

* **Step.** A fit is a ``step`` if and only if its residual sum of squares (RSS) is no better
  than the closed-form ``s -> 0`` supremum: the best RSS of a step curve, taken over every split
  between adjacent levels and over each level sitting on the step at its own mean. This file
  computes that supremum itself (``_step_rss``), as the within-level sum of squares plus the
  weighted isotonic (decreasing) fit of the group means, and calls nothing from robustmargin's
  internals. A counting rule ("fewer than N levels in the transition") is not the criterion, so a
  sharp transition that the data still identify is ``OK``. The rule also holds when no start
  converged.
* **Flat.** ``flat`` is never applied when both asymptotes are fixed (the curve is then pinned,
  not degenerate), and otherwise only when the optimum sits at the ``s -> infinity`` boundary. A
  shallow but identified sigmoid is ``OK``.
* **Start grid.** A fit whose RSS is worse than the best point of the start grid is
  ``NOT_CONVERGED``, not a number.

Seeds are pinned, and each case is written so the verdict is decided by the independent oracle,
not by the fit's own status.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import norm

import robustmargin.models as models
from robustmargin import (
    Axis,
    ContinuousObservations,
    Fit,
    Status,
    fit_dose_response,
)

_LOG = Axis(name="severity", unit="unit", scale="log")
_LEVELS = [0.0, *np.logspace(-1.0, 1.0, 7).tolist()]  # includes 1.0, so mu = 0 is a tested level
_ESTIMATE: dict[str, Any] = {"upper": "estimate", "lower": "estimate"}


# ---------------------------------------------------------------------------------------------
# Independent oracle: the s -> 0 supremum of the likelihood, as an infimum of the RSS.
# ---------------------------------------------------------------------------------------------


def _isotonic_decreasing_rss(means: list[float], weights: list[float]) -> float:
    """Weighted RSS of the decreasing isotonic fit to ``means`` (pool adjacent violators)."""
    blocks: list[list[float]] = []  # [mean, weight, sum(w m), sum(w m^2)]
    for m, w in zip(means, weights, strict=True):
        blocks.append([m, w, m * w, m * m * w])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            b = blocks.pop()
            a = blocks.pop()
            weight, total, sq = a[1] + b[1], a[2] + b[2], a[3] + b[3]
            blocks.append([total / weight, weight, total, sq])
    # Misfit of the pooled blocks about their means, from the raw sums.
    return float(sum(b[3] - b[2] ** 2 / b[1] for b in blocks))


def _step_rss(severity: list[float], value: list[float]) -> float:
    """The infimum of the RSS over step curves (``s -> 0``), decreasing outcome, both asymptotes
    free: every split between adjacent levels, and every interior level on the step with the
    curve through its own mean (its fitted value is free in ``[lower, upper]``, which the
    isotonic term enforces). The constant curve bounds it above."""
    sev = np.asarray(severity)
    y = np.asarray(value)
    levels = sorted(set(severity))
    groups = [y[sev == lv] for lv in levels]

    def ss(v: np.ndarray) -> float:
        return float(((v - v.mean()) ** 2).sum())

    best = ss(y)
    for k in range(1, len(levels)):
        left, right = np.concatenate(groups[:k]), np.concatenate(groups[k:])
        if left.mean() >= right.mean():
            best = min(best, ss(left) + ss(right))
    for k in range(1, len(levels) - 1):
        left, right = np.concatenate(groups[:k]), np.concatenate(groups[k + 1 :])
        triple = _isotonic_decreasing_rss(
            [float(left.mean()), float(groups[k].mean()), float(right.mean())],
            [float(left.size), float(groups[k].size), float(right.size)],
        )
        best = min(best, ss(left) + ss(groups[k]) + ss(right) + triple)
    return best


def _smooth_rss(severity: list[float], value: list[float], link: str) -> float:
    """Best RSS of a smooth curve with ``s >= 0.02``, both asymptotes free, by multi-start least
    squares. A lower bound on what any fit with that ``s`` could reach is not needed: the aim is
    to show a smooth curve does not beat the step."""
    x = np.asarray(severity)
    y = np.asarray(value)
    cdf = norm.cdf if link == "probit" else (lambda z: 1.0 / (1.0 + np.exp(-z)))
    pos = x > 0
    lx = np.log(np.where(pos, x, 1.0))

    def resid(p: np.ndarray) -> np.ndarray:
        mu, log_s, u, low = p
        f = np.where(pos, cdf((lx - mu) / math.exp(log_s)), 0.0)
        return np.asarray(y - (u - (u - low) * f))

    best = math.inf
    for mu0 in np.linspace(-2.0, 2.0, 9):
        for s0 in (0.05, 0.2, 0.6, 1.5):
            run = least_squares(
                resid,
                [mu0, math.log(s0), float(y.max()), float(y.min())],
                bounds=(
                    [-5.0, math.log(0.02), -np.inf, -np.inf],
                    [5.0, math.log(50.0), np.inf, np.inf],
                ),
            )
            best = min(best, float((run.fun**2).sum()))
    return best


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------


def _sigmoid_series(
    *,
    link: str,
    mu: float,
    s: float,
    sigma: float,
    reps: int,
    seed: int,
    upper: float = 10.0,
    lower: float = 2.0,
    levels: list[float] | None = None,
) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    severity = [x for x in (levels or _LEVELS) for _ in range(reps)]
    x = np.asarray(severity)
    with np.errstate(divide="ignore"):
        z = (np.log(x) - mu) / s
    f = norm.cdf(z) if link == "probit" else 1.0 / (1.0 + np.exp(-z))
    f = np.where(x == 0.0, 0.0, f)
    mean = upper - (upper - lower) * f
    return severity, (mean + rng.normal(0.0, sigma, size=x.size)).tolist()


def _obs(severity: list[float], value: list[float]) -> ContinuousObservations:
    return ContinuousObservations.from_observations(
        _LOG, severity=severity, value=value, outcome="response", direction="decreasing"
    )


def _fit(severity: list[float], value: list[float], link: str = "probit", **kw: Any) -> Fit:
    return fit_dose_response(
        _obs(severity, value), model="gaussian", link=link, **{**_ESTIMATE, **kw}
    )


def _assert_failed_with(fit: Fit, word: str) -> None:
    assert fit.status is Status.NOT_CONVERGED
    assert fit.params is None and fit.covariance is None and fit.log_likelihood is None
    assert fit.residual is None and fit.diagnostics is None
    assert any(word in w.lower() for w in fit.warnings), fit.warnings


# ---------------------------------------------------------------------------------------------
# The oracle itself, on cases with a known answer
# ---------------------------------------------------------------------------------------------


class TestTheStepOracle:
    def test_an_exact_two_level_step_has_zero_step_rss(self) -> None:
        assert _step_rss([1.0, 1.0, 2.0, 2.0], [5.0, 5.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_noise_inside_a_step_is_its_sum_of_squares_about_each_side(self) -> None:
        got = _step_rss([1.0, 1.0, 2.0, 2.0], [5.0, 7.0, 1.0, 3.0])

        assert got == pytest.approx(2.0 + 2.0)

    def test_a_level_on_the_step_may_sit_between_the_asymptotes(self) -> None:
        severity = [1.0, 2.0, 3.0]
        value = [10.0, 6.0, 2.0]

        assert _step_rss(severity, value) == pytest.approx(0.0)

    def test_a_monotone_staircase_is_not_improved_by_a_split_it_cannot_use(self) -> None:
        severity = [1.0, 2.0, 3.0, 4.0]
        value = [10.0, 7.0, 4.0, 1.0]

        # Best two-group split of 10, 7, 4, 1 pools {10, 7 | 4, 1}: 4.5 + 4.5; a level on the
        # step keeps one level exact and pools the rest: {10 | 7 | 4, 1} costs 4.5.
        assert _step_rss(severity, value) == pytest.approx(4.5)


# ---------------------------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------------------------

# (seed, reps): near-step series, probit, mu = 0 at a tested level, s in 0.31 .. 0.375, noise 0.8
# on a gap of 8. The replicate count is the one for which the oracle (below) says no smooth curve
# beats the step on that seed's draw.
_STEP_CASES = ((58, 3), (3, 3), (35, 4))
_STEP_S = (0.31, 0.34, 0.375)


class TestStepRule:
    @pytest.mark.parametrize("s", _STEP_S)
    @pytest.mark.parametrize(("seed", "reps"), _STEP_CASES)
    def test_a_near_step_through_a_tested_level_is_a_step_when_no_curve_beats_the_step(
        self, seed: int, reps: int, s: float
    ) -> None:
        severity, value = _sigmoid_series(
            link="probit", mu=0.0, s=s, sigma=0.8, reps=reps, seed=seed
        )
        step = _step_rss(severity, value)
        assert _smooth_rss(severity, value, "probit") >= step * (1.0 - 1e-6), (
            "oracle precondition: no smooth curve beats the step on this draw"
        )

        fit = _fit(severity, value)

        _assert_failed_with(fit, "step")

    def test_probit_s_0_2_with_tiny_noise_is_ok_and_recovers_s(self) -> None:
        severity, value = _sigmoid_series(
            link="probit", mu=0.0, s=0.2, sigma=1.0e-4, reps=4, seed=7
        )
        assert _smooth_rss(severity, value, "probit") < _step_rss(severity, value) * 0.5, (
            "oracle precondition: a smooth curve is far better than any step on this draw"
        )

        fit = _fit(severity, value)

        assert fit.status is Status.OK and fit.params is not None
        assert fit.params["s"].value == pytest.approx(0.2, rel=0.05)
        assert fit.residual is not None
        assert fit.residual.rss < _step_rss(severity, value)

    def test_an_ok_fit_always_has_an_rss_below_the_step_supremum(self) -> None:
        severity, value = _sigmoid_series(
            link="logit", mu=math.log(0.8), s=0.5, sigma=0.3, reps=4, seed=20261108
        )

        fit = _fit(severity, value, link="logit")

        assert fit.status is Status.OK and fit.residual is not None
        assert fit.residual.rss < _step_rss(severity, value)

    def test_a_boundary_run_on_step_data_is_a_step_not_an_optimizer_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No start converged, and the data are a step: the reason is ``step``. It used to be
        reported as ``the optimizer did not converge``."""
        # Levels 0 .. 6.4 (as above): ten at 10 up to the middle level, the middle level between,
        # the rest at 2, with small deterministic wobble. Zero-spread-free so the step RSS is > 0.
        wobble = [0.05, -0.05, 0.0]
        severity = [x for x in _LEVELS for _ in range(3)]
        base = [10.0 if x < 1.0 else (6.0 if x == 1.0 else 2.0) for x in severity]
        value = [b + wobble[i % 3] for i, b in enumerate(base)]
        step = _step_rss(severity, value)
        assert step > 0.0

        def raising(*args: object, **kwargs: object) -> object:
            raise ValueError("no solution")

        monkeypatch.setattr(models, "least_squares", raising)

        fit = _fit(severity, value, link="logit")

        _assert_failed_with(fit, "step")
        assert not any("optimizer did not converge" in w for w in fit.warnings)

    def test_a_smooth_series_with_no_converging_start_is_still_an_optimizer_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        severity, value = _sigmoid_series(
            link="logit", mu=math.log(0.8), s=0.5, sigma=0.3, reps=4, seed=20261108
        )
        assert _smooth_rss(severity, value, "logit") < _step_rss(severity, value)

        def raising(*args: object, **kwargs: object) -> object:
            raise ValueError("no solution")

        monkeypatch.setattr(models, "least_squares", raising)

        fit = _fit(severity, value, link="logit")

        _assert_failed_with(fit, "optimizer")


# ---------------------------------------------------------------------------------------------
# Flat
# ---------------------------------------------------------------------------------------------


class TestFlatRule:
    def test_a_shallow_but_identified_logistic_with_free_asymptotes_is_ok(self) -> None:
        # s = 5 over a log span of ln(100) = 4.6: the curve covers a narrow band of F, but a
        # noise of 1e-3 identifies it.
        severity, value = _sigmoid_series(
            link="logit",
            mu=0.0,
            s=5.0,
            sigma=1.0e-3,
            reps=4,
            seed=11,
            levels=np.logspace(-1.0, 1.0, 8).tolist(),
        )

        fit = _fit(severity, value, link="logit")

        assert fit.status is Status.OK and fit.params is not None
        assert fit.params["s"].value == pytest.approx(5.0, rel=0.1)

    def test_fixed_asymptotes_with_the_same_shallow_curve_are_ok(self) -> None:
        severity, value = _sigmoid_series(
            link="logit",
            mu=0.0,
            s=5.0,
            sigma=1.0e-3,
            reps=4,
            seed=11,
            upper=1.0,
            lower=0.0,
            levels=np.logspace(-1.0, 1.0, 8).tolist(),
        )

        fit = _fit(severity, value, link="logit", upper=1.0, lower=0.0)

        assert fit.status is Status.OK and fit.params is not None
        assert fit.params["s"].value == pytest.approx(5.0, rel=0.1)

    def test_flat_is_never_the_reason_when_both_asymptotes_are_fixed(self) -> None:
        severity, value = _sigmoid_series(
            link="logit", mu=0.0, s=0.4, sigma=0.4, reps=3, seed=5, upper=1.0, lower=0.0
        )

        fit = _fit(severity, value, link="logit", upper=1.0, lower=0.0)

        assert not any("flat" in w.lower() for w in fit.warnings), fit.warnings


# ---------------------------------------------------------------------------------------------
# The fit must beat the start grid
# ---------------------------------------------------------------------------------------------


class TestFitMustBeatTheStartGrid:
    @staticmethod
    def _series() -> tuple[list[float], list[float]]:
        return _sigmoid_series(
            link="logit", mu=math.log(0.8), s=0.5, sigma=0.3, reps=4, seed=20261108
        )

    def test_an_unpatched_fit_on_the_series_is_ok(self) -> None:
        severity, value = self._series()

        assert _fit(severity, value, link="logit").status is Status.OK

    def test_an_optimum_worse_than_the_best_grid_point_is_not_converged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The optimiser reports success at a point whose cost is far above the grid's best:
        RSS must be <= SSW + min(grid cost) + tol, so this is ``NOT_CONVERGED``, with a reason and
        no number."""
        severity, value = self._series()
        real = models.least_squares

        def worse(*args: Any, **kwargs: Any) -> object:
            run = real(*args, **kwargs)
            return SimpleNamespace(x=run.x + 3.0, cost=run.cost * 1.0e6 + 1.0e3, status=run.status)

        monkeypatch.setattr(models, "least_squares", worse)

        fit = _fit(severity, value, link="logit")

        assert fit.status is Status.NOT_CONVERGED
        assert fit.params is None and fit.covariance is None and fit.log_likelihood is None
        assert fit.residual is None and fit.diagnostics is None
        assert fit.warnings, "a failed fit must say why"
        assert any("grid" in w.lower() for w in fit.warnings), fit.warnings


# ---------------------------------------------------------------------------------------------
# The unbounded-asymptote boundary: exponential-tail and straight-line limits
# ---------------------------------------------------------------------------------------------

_LIN = Axis(name="severity", unit="unit", scale="linear")
_T = np.arange(7.0)
_SD = 0.05
_N = 20
_LAMBDAS = np.concatenate([np.linspace(1.0e-3, 5.0, 1000), np.linspace(5.0, 30.0, 200)])


def _cdf(link: str, z: np.ndarray) -> np.ndarray:
    if link == "probit":
        return np.asarray(norm.cdf(z))
    if link == "logit":
        return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -700.0, 700.0))))
    return np.asarray(-np.expm1(-np.exp(np.clip(z, -700.0, 50.0))))  # cloglog


def _tail_rss(
    t: np.ndarray, y: np.ndarray, w: np.ndarray, fixed: float, free_is_lower: bool
) -> float:
    """Independent oracle: the infimum of the weighted lack-of-fit over the exponential-tail
    family ``fixed -/+ A exp(+/- lam t)``, ``A >= 0``, ``lam in (0, inf)``, and the line
    ``a -/+ c t`` (``lam -> 0``) with ``c >= 0`` and a free intercept, plus the constant.
    A 1-D scan over ``lam`` with the closed-form non-negative ``A``."""
    sign = 1.0 if free_is_lower else -1.0  # lower free: y = fixed - A e^{lam (t-tmax)}

    def cost(lam: float) -> float:
        b = np.exp(lam * (t - t.max())) if free_is_lower else np.exp(-lam * t)
        r = sign * (fixed - y)
        amp = max(0.0, float(np.sum(w * b * r) / np.sum(w * b * b)))
        return float(np.sum(w * (r - amp * b) ** 2))

    best = float(np.sum(w * (y - fixed) ** 2))
    costs = [cost(float(lam)) for lam in _LAMBDAS]
    k = int(np.argmin(costs))
    best = min(best, costs[k])
    lo, hi = float(_LAMBDAS[max(k - 1, 0)]), float(_LAMBDAS[min(k + 1, _LAMBDAS.size - 1)])
    best = min(best, float(minimize_scalar(cost, bounds=(lo, hi), method="bounded").fun))
    # The line: y = alpha - sign * c * t with c >= 0, alpha free.
    tbar = float(np.sum(w * t) / np.sum(w))
    ybar = float(np.sum(w * y) / np.sum(w))
    slope = float(np.sum(w * (t - tbar) * (y - ybar)) / np.sum(w * (t - tbar) ** 2))
    if -sign * slope >= 0.0:
        fitted = ybar + slope * (t - tbar)
    else:
        fitted = np.full_like(y, ybar)
    best = min(best, float(np.sum(w * (y - fitted) ** 2)))
    return best


def _unconstrained_rss(
    t: np.ndarray, y: np.ndarray, w: np.ndarray, link: str, fixed: float, free_is_lower: bool
) -> float:
    """RSS (weighted, about the cell means) of a deliberately unconstrained multi-start scipy fit
    of ``upper - (upper - lower) F((t - mu)/s)`` with one asymptote free."""

    def resid(p: np.ndarray) -> np.ndarray:
        mu, log_s, free = p
        f = _cdf(link, (t - mu) / math.exp(log_s))
        upper, lower = (fixed, free) if free_is_lower else (free, fixed)
        return np.asarray(np.sqrt(w) * (y - (upper - (upper - lower) * f)))

    best = math.inf
    for mu0 in (-4.0, 0.0, 3.0, 6.0, 10.0, 30.0):
        for s0 in (0.3, 1.0, 3.0, 10.0):
            for free0 in (-5.0, 0.0, 0.4, 5.0):
                try:
                    run = least_squares(
                        resid,
                        [mu0, math.log(s0), free0],
                        bounds=([-1.0e4, -6.0, -1.0e6], [1.0e4, 12.0, 1.0e6]),
                        xtol=1e-15,
                        ftol=1e-15,
                        gtol=1e-15,
                        max_nfev=2000,
                    )
                except ValueError:
                    continue
                best = min(best, float((run.fun**2).sum()))
    return best


def _tail_series(free_is_lower: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if free_is_lower:
        mean = 1.0 - 0.6 * np.exp(0.8 * (_T - 6.0))
    else:
        mean = 0.0 + 0.8 * np.exp(-0.8 * _T)
    return _T, mean + rng.normal(0.0, 0.003, _T.size)


def _tail_fit(t: np.ndarray, mean: np.ndarray, link: str, free_is_lower: bool) -> Fit:
    obs = ContinuousObservations.from_summaries(
        _LIN,
        severity=t.tolist(),
        mean=mean.tolist(),
        sd=[_SD] * t.size,
        n=[_N] * t.size,
        outcome="score",
        direction="decreasing",
    )
    kw: dict[str, Any] = (
        {"upper": 1.0, "lower": "estimate"}
        if free_is_lower
        else {"upper": "estimate", "lower": 0.0}
    )
    return fit_dose_response(obs, model="gaussian", link=link, **kw)


_LINKS = ("logit", "cloglog", "probit")
_SIDES = (True, False)  # lower asymptote free (upper fixed), and the mirror


class TestUnboundedAsymptoteBoundary:
    @pytest.mark.parametrize("free_is_lower", _SIDES)
    @pytest.mark.parametrize("link", _LINKS)
    def test_an_exponential_tail_toward_a_free_asymptote_is_refused_as_unbounded(
        self, link: str, free_is_lower: bool
    ) -> None:
        t, mean = _tail_series(free_is_lower, seed=3)

        fit = _tail_fit(t, mean, link, free_is_lower)

        _assert_failed_with(fit, "unbounded asymptote")
        assert fit.params is None

    @pytest.mark.parametrize(
        ("link", "free_is_lower"),
        # Cloglog reaches the exponential tail only on the lower-asymptote side; on the mirror its
        # survival decays doubly exponentially, so a scipy fit stays above that supremum.
        [("logit", True), ("logit", False), ("cloglog", True)],
    )
    def test_the_unconstrained_fit_attains_the_independent_tail_supremum(
        self, link: str, free_is_lower: bool
    ) -> None:
        """Shows the data sit on the boundary: no curve of the family beats the exponential-tail
        supremum, and a free scipy fit gets to it (it cannot go below it)."""
        t, mean = _tail_series(free_is_lower, seed=3)
        w = np.full(t.size, float(_N))
        fixed = 1.0 if free_is_lower else 0.0

        sup = _tail_rss(t, mean, w, fixed, free_is_lower)
        got = _unconstrained_rss(t, mean, w, link, fixed, free_is_lower)

        assert got >= sup * (1.0 - 1.0e-3)
        assert got == pytest.approx(sup, rel=0.02, abs=1.0e-6)

    @pytest.mark.parametrize("link", _LINKS)
    def test_the_rule_does_not_depend_on_s_hat_exceeding_the_span(self, link: str) -> None:
        """The refusal is by RSS against the boundary, whatever the optimiser's s. The observed
        failure had s-hat = 0.58 below the span of 6 and was still a boundary fit."""
        t, mean = _tail_series(True, seed=3)

        fit = _tail_fit(t, mean, link, True)

        assert fit.status is Status.NOT_CONVERGED
        assert fit.params is None
        assert any("unbounded asymptote" in w.lower() for w in fit.warnings), fit.warnings


class TestIdentifiedCurvesStayOk:
    def test_a_sharp_identified_probit_curve_is_ok(self) -> None:
        severity, value = _sigmoid_series(
            link="probit", mu=0.0, s=0.2, sigma=1.0e-4, reps=4, seed=7
        )

        assert _fit(severity, value).status is Status.OK

    def test_a_shallow_identified_logit_curve_with_both_asymptotes_free_is_ok(self) -> None:
        severity, value = _sigmoid_series(
            link="logit",
            mu=0.0,
            s=5.0,
            sigma=1.0e-3,
            reps=4,
            seed=11,
            levels=np.logspace(-1.0, 1.0, 8).tolist(),
        )

        assert _fit(severity, value, link="logit").status is Status.OK


class TestSlackIsRelativeToLackOfFit:
    def test_a_large_df_fit_that_clearly_beats_the_step_supremum_is_ok(self) -> None:
        """About 1e5 observations with a small sd: the within-level sum of squares dwarfs the
        lack of fit, so a slack relative to the whole RSS would swallow a fit that beats the step
        by a wide chi-square margin."""
        levels = np.logspace(-1.0, 1.0, 8)
        n = 12500
        sd = 0.5
        mu, s = 0.0, 0.5
        z = (np.log(levels) - mu) / s
        mean = 10.0 - 8.0 * norm.cdf(z)
        # Cell means carry no sampling noise here: the lack of fit of the truth is exactly zero,
        # while the best step misfits by an amount far above the absolute floor.
        obs = ContinuousObservations.from_summaries(
            _LOG,
            severity=levels.tolist(),
            mean=mean.tolist(),
            sd=[sd] * levels.size,
            n=[n] * levels.size,
            outcome="response",
            direction="decreasing",
        )
        ssw = float(levels.size * (n - 1) * sd**2)
        step_lack = _step_rss(levels.tolist(), mean.tolist())  # one point per level: lack only
        # Equal weights n: the step's lack of fit in RSS units is n * step_lack, i.e. in chi-square
        # units (sigma = sd) it is far above any plausible slack.
        assert n * step_lack / sd**2 > 1.0e3

        fit = fit_dose_response(obs, model="gaussian", link="probit", **_ESTIMATE)

        assert fit.status is Status.OK, fit.warnings
        assert fit.residual is not None
        assert fit.residual.rss - ssw < 0.01 * n * step_lack


class TestCloglogAtExtremeZ:
    @pytest.mark.parametrize("z", [710.0, 800.0, 1.0e4])
    def test_the_cloglog_density_slope_is_finite_beyond_the_exp_overflow(self, z: float) -> None:
        got = models._f_pdf_slope(np.array([z, -z]), "cloglog")

        assert np.all(np.isfinite(got)), got
        assert got[0] == pytest.approx(0.0, abs=1e-300)


# ---------------------------------------------------------------------------------------------
# Offset and scale invariance of a logit fit
# ---------------------------------------------------------------------------------------------


def _unit_range_series() -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(5)
    t = np.repeat(np.linspace(0.0, 1.0, 7), 4)
    base = 1.0 - 1.0 / (1.0 + np.exp(-(t - 0.5) / 0.12))
    return (t * 5.0).tolist(), (base + rng.normal(0.0, 0.01, t.size)).tolist()


def _linear_fit(severity: list[float], value: list[float], link: str = "logit") -> Fit:
    obs = ContinuousObservations.from_observations(
        _LIN, severity=severity, value=value, outcome="r", direction="decreasing"
    )
    return fit_dose_response(obs, model="gaussian", link=link, **_ESTIMATE)


class TestOffsetAndScaleInvariance:
    def test_the_unit_range_logit_design_is_ok(self) -> None:
        severity, value = _unit_range_series()

        fit = _linear_fit(severity, value)

        assert fit.status is Status.OK, fit.warnings

    @pytest.mark.parametrize(("c", "d"), [(1.0, 1.0e5), (1.0, 1.0e7), (1.0e-3, 0.0)])
    def test_an_offset_or_rescaling_of_the_response_gives_the_same_fit(
        self, c: float, d: float
    ) -> None:
        """A shift of 1e7 over a range of 1 used to raise OverflowError."""
        severity, value = _unit_range_series()
        base = _linear_fit(severity, value)

        moved = _linear_fit(severity, [c * v + d for v in value])

        assert base.status is Status.OK and base.params is not None
        assert moved.status is base.status, moved.warnings
        assert moved.params is not None
        assert moved.params["mu"].value == pytest.approx(base.params["mu"].value, rel=1e-5)
        assert moved.params["s"].value == pytest.approx(base.params["s"].value, rel=1e-5)
        for name in ("upper", "lower"):
            assert moved.params[name].value == pytest.approx(
                c * base.params[name].value + d, rel=1e-6
            )


# ---------------------------------------------------------------------------------------------
# A tail limit whose rate lies beyond a coarse lambda grid
# ---------------------------------------------------------------------------------------------

_TAIL_SPAN = 10.0
_TAIL_FRACTIONS = np.array([0.0, 0.5, 0.991, 0.994, 0.997, 0.9985, 1.0])


def _small_end_gap_series(rate: float) -> tuple[np.ndarray, np.ndarray]:
    """``5 - 3 exp(rate (x - 1))`` on a unit span, scaled to ``_TAIL_SPAN``: the last four gaps
    are under 1% of the span, so the tail minimum's rate (``rate``/span) exceeds ``e^6 / span``."""
    return _TAIL_FRACTIONS * _TAIL_SPAN, 5.0 - 3.0 * np.exp(rate * (_TAIL_FRACTIONS - 1.0))


def _wide_tail_rss(t: np.ndarray, y: np.ndarray, w: np.ndarray, lams: np.ndarray) -> float:
    """Lack of fit of ``5 - A exp(lam (t - tmax))`` (upper fixed at 5), ``A >= 0``, minimised over
    ``lams`` and then polished in the best bracket."""

    def cost(lam: float) -> float:
        b = np.exp(lam * (t - t.max()))
        r = 5.0 - y
        amp = max(0.0, float(np.sum(w * b * r) / np.sum(w * b * b)))
        return float(np.sum(w * (r - amp * b) ** 2))

    costs = [cost(float(lam)) for lam in lams]
    k = int(np.argmin(costs))
    lo, hi = float(lams[max(k - 1, 0)]), float(lams[min(k + 1, lams.size - 1)])
    return min(costs[k], float(minimize_scalar(cost, bounds=(lo, hi), method="bounded").fun))


class TestTailLimitWithASmallEndGap:
    @pytest.mark.parametrize("link", ["logit", "cloglog"])
    def test_data_on_a_fast_tail_are_refused_as_unbounded_not_as_an_optimizer_failure(
        self, link: str
    ) -> None:
        rate = 700.0  # per unit span: e^6 = 403 < 700, so beyond a grid ending at e^6 / span
        t, mean = _small_end_gap_series(rate)
        w = np.full(t.size, 10.0)
        coarse = np.linspace(1.0e-3, math.exp(6.0) / _TAIL_SPAN, 400)
        wide = np.exp(np.linspace(math.log(1.0e-3), math.log(1.0e4), 4000))
        assert _wide_tail_rss(t, mean, w, wide) < 1.0e-3 * _wide_tail_rss(t, mean, w, coarse), (
            "oracle precondition: only a wide lambda scan finds this tail"
        )
        obs = ContinuousObservations.from_summaries(
            _LIN,
            severity=t.tolist(),
            mean=mean.tolist(),
            sd=[0.05] * t.size,
            n=[10] * t.size,
            outcome="r",
            direction="decreasing",
        )

        fit = fit_dose_response(obs, model="gaussian", link=link, upper=5.0, lower="estimate")

        _assert_failed_with(fit, "unbounded asymptote")
        assert not any("optimizer" in w.lower() for w in fit.warnings), fit.warnings


# ---------------------------------------------------------------------------------------------
# No false step refusal
# ---------------------------------------------------------------------------------------------

_SHARP = (
    ("logit", 1087),
    ("logit", 1078),
    ("logit", 1003),
    ("cloglog", 1056),
    ("cloglog", 1077),
    ("cloglog", 1125),
    ("probit", 1013),
    ("probit", 1136),
)


def _sharp_series(link: str, case: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(case)
    mu = float(rng.uniform(0.25, 0.75))  # then log s, log sigma, and the noise, in this order
    s = math.exp(rng.uniform(-4.5, -2.5))
    sigma = math.exp(rng.uniform(-4.5, -3.0))
    t = np.repeat(np.linspace(0.0, 1.0, 7), 3)
    return t, 1.0 - _cdf(link, (t - mu) / s) + rng.normal(0.0, sigma, t.size)


def _multi_start_rss(t: np.ndarray, y: np.ndarray, link: str) -> float:
    def resid(p: np.ndarray) -> np.ndarray:
        mu, log_s, u, low = p
        scale = math.exp(float(np.clip(log_s, -30.0, 30.0)))
        return np.asarray(y - (u - (u - low) * _cdf(link, (t - mu) / scale)))

    best = math.inf
    for mu0 in np.linspace(t.min(), t.max(), 9):
        for s0 in (0.01, 0.03, 0.1, 0.3):
            try:
                run = least_squares(
                    resid, [mu0, math.log(s0), y.max(), y.min()], method="lm", max_nfev=3000
                )
            except ValueError:
                continue
            best = min(best, float((run.fun**2).sum()))
    return best


class TestNoFalseStepRefusal:
    @pytest.mark.parametrize(("link", "case"), _SHARP)
    def test_an_identified_sharp_curve_that_beats_the_step_supremum_is_ok(
        self, link: str, case: int
    ) -> None:
        t, y = _sharp_series(link, case)
        step = _step_rss(t.tolist(), y.tolist())
        assert _multi_start_rss(t, y, link) < 0.97 * step, (
            "oracle precondition: an independent multi-start beats the step supremum"
        )

        fit = _linear_fit(t.tolist(), y.tolist(), link)

        assert fit.status is Status.OK, fit.warnings
        assert fit.residual is not None
        assert fit.residual.rss < step


# ---------------------------------------------------------------------------------------------
# Honest labelling when no start converged
# ---------------------------------------------------------------------------------------------


def _raising(*args: object, **kwargs: object) -> object:
    raise ValueError("no solution")


class TestLabellingWithNoConvergedStart:
    def test_a_step_with_no_converged_start_says_both(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wobble = [0.05, -0.05, 0.0]
        severity = [x for x in _LEVELS for _ in range(3)]
        base = [10.0 if x < 1.0 else (6.0 if x == 1.0 else 2.0) for x in severity]
        value = [b + wobble[i % 3] for i, b in enumerate(base)]
        monkeypatch.setattr(models, "least_squares", _raising)

        fit = _fit(severity, value, link="logit")

        _assert_failed_with(fit, "step")
        assert any("no start converged" in w.lower() for w in fit.warnings), fit.warnings

    def test_an_exact_line_with_no_converged_start_names_flat_and_says_no_start_converged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        severity = [x for x in _LEVELS if x > 0.0 for _ in range(3)]
        spread = [0.2, -0.2, 0.0]
        value = [6.0 - 1.5 * math.log(x) + spread[i % 3] for i, x in enumerate(severity)]
        monkeypatch.setattr(models, "least_squares", _raising)

        fit = _fit(severity, value, link="logit")

        _assert_failed_with(fit, "flat")
        assert any("no start converged" in w.lower() for w in fit.warnings), fit.warnings
