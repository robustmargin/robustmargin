"""Confidence intervals for a solved threshold: profile likelihood (the primary method) and the
delta method (the cross-check).

Both operate on the transformed axis ``theta`` -- ``log(severity)`` on a log axis, raw severity
on a linear axis -- and map back to the natural severity scale before returning, per
:func:`robustmargin.threshold.threshold`'s contract. :func:`solve_value` is the shared closed-form
step: ``theta_hat = mu + s * z*``, where ``z* = F^-1((upper - target) / (upper - lower))``.

**Profile likelihood.** With both asymptotes fixed, the offset trick applies
exactly: a no-intercept ``GLM`` on the failure rate, covariate ``t - theta``, offset ``z*``
(constant, since fixed asymptotes make ``z*`` fixed too). With an estimated asymptote the offset
trick does not apply, because ``z* = F^-1((u - P*) / (u - l))`` moves with the candidate ``(u,
l)`` the constrained optimiser is trying at each step (and ``P*`` itself moves with ``u`` for
``baseline_fraction``): the substitution ``mu = theta - s * z*(u, l)`` is folded directly into
the log-likelihood instead, handed to ``statsmodels.base.model.GenericLikelihoodModel`` with the
same ``bfgs`` -> ``newton`` sequence the estimated-asymptote fit itself uses
(a likelihood for a statsmodels optimiser, never a hand-rolled one). The first run starts from the
fit's own parameters. Because the constrained surface can have several basins, and because the
constrained maximum can sit at the reachability wall (an estimated asymptote tending to the target,
where the logit parametrisation cannot represent it and the score diverges), the inner fit is also
restarted from alternative points chosen by a deterministic scan of log-likelihood values (no
optimiser): over ``log(s)`` from eight orders of magnitude below the span of the tested levels to
three above, over ``z*`` and, when both asymptotes are estimated, over one more free coordinate,
in a second chart whose coordinates are ``(log s, z*)`` and that coordinate. In that chart the
derived asymptote is the solution of the constraint ``F(z*) (u - l) = u - P*``, which is linear in
``(u, l)`` given ``z*``, with ``F`` and its complement computed separately, so an asymptote can
reach the target to within floating point. Up to four scan points are taken, and where an
estimated asymptote's feasible interval has a finite end, up to four more starts lie on that box
face. Where no such chart applies, the first chart is scanned over ``log(s)`` crossed with each
estimated asymptote. Alternatives are skipped when the first run's deviance is
already below the chi-square target, since its sign is then settled, except at the estimate,
where the value itself is used. The best alternative replaces the
first run only if its log-likelihood is larger by more than the Newton tolerance; it is then
polished, so the profile log-likelihood at each ``theta`` can only be raised by the alternatives.
That does not make a bound monotone in the number of starts: a different value at one ``theta``
can move where the search first sees a crossing, so a bound can move in either direction. Once a
stride brackets a crossing, the deviance is evaluated at interior points of the bracket and the
root is solved for in the first sub-interval, counted from the estimate, that changes sign.
A candidate ``(u, l)`` that makes ``P*`` unreachable, or an inner fit that fails
outright, opens that side of the search rather than returning a number (a failed fit never carries
a number).

Both ``ell_hat`` and ``ell_p(theta)`` are computed with
``robustmargin.models._full_binomial_log_likelihood`` at natural-scale parameters -- the inner GLM
or ``GenericLikelihoodModel`` fit is used only as an optimiser for the *other* parameters at a
fixed ``theta``, never trusted for the log-likelihood value itself, so the two numbers a root is
found between are always computed the same way (GLM deviance, ``Fit.log_likelihood`` and the
log-axis control row otherwise do not agree term for term).

**Delta method.** ``theta = mu + s * z*``, so ``d(theta)/d(mu) = 1``, ``d(theta)/d(s) = z*``,
and -- when an asymptote is estimated -- ``d(theta)/d(u) = s * (1 / f(z*)) * d(q)/d(u)``
(mirrored for ``l``), where ``q = (u - P*) / (u - l)`` and ``f`` is the link's density. The
variance of ``theta`` is computed on the transformed axis directly with ``fit.covariance``
(observed information), then converted to a *log-scale* standard error even
on a linear axis (``sigma_log = sigma_theta / value``) before exponentiating around ``value`` --
this is what keeps the interval positive and equivariant under ``x -> c * x`` regardless of
axis.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import expit, log_ndtr, ndtr
from scipy.special import logit as _logit
from scipy.stats import chi2, norm
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from robustmargin.models import (
    _NEWTON_DECREMENT_TOLERANCE,
    _STATSMODELS_LINKS,
    Fit,
    _cells_to_arrays,
    _f_log_cdf,
    _f_log_survival,
    _full_binomial_log_likelihood,
    _newton_decrement_half,
    _scan_log_s_grid,
    _share_logits,
    _transform,
)
from robustmargin.models import _top_local_maxima as _top_maxima
from robustmargin.types import Censoring, Definition, _check_finite_number

__all__ = ["IntervalResult", "delta_interval", "profile_interval", "solve_value"]

# The profile search brackets outward from theta_hat to this limit before
# giving up and reporting that side open. Log axis: additive in log(x); linear axis:
# multiplicative in x (an additive log(100) on a linear axis would be unit-dependent).
_SEARCH_MULTIPLE = 100.0

# `exp` of anything beyond this overflows a float64 (`math.exp(709.8)` is the last finite
# value). A delta half-width at or near it means the interval is unbounded in practice, so
# it is reported as no bounds rather than as a number that happens to fit in a float.
_MAX_LOG_HALF_WIDTH = 700.0
_MAX_BRACKET_STEPS = 200
# Bisections of a stride whose end point the inner fit fails at, so the search can still bracket a
# root lying before the failure.
_MAX_BACKOFF_HALVINGS = 30
_UNREACHABLE_LOGLIKELIHOOD = -1.0e12
# The scan that seeds the estimated-asymptote profile's alternative starts. ``log(s)`` runs from
# ``log(span) - _SCAN_LOG_S_BELOW`` to ``log(span) + _SCAN_LOG_S_ABOVE`` in steps of
# ``_SCAN_LOG_S_STEP``, with ``span`` the range of the finite tested levels: a transition squeezed
# between ``theta`` and a tested level has ``s`` orders of magnitude below the span, a flat curve a
# few above. The step is fine because a basin of the likelihood in ``log(s)`` can be narrower than
# one half. ``z*`` takes the grid values where they are feasible: steps of ``_SCAN_Z_STAR_STEP``
# over ``[-_SCAN_Z_STAR_RANGE, _SCAN_Z_STAR_RANGE]`` and the far values ``_SCAN_Z_STAR_FAR``,
# because a basin can be narrower than a unit step in ``z*``. The finite ends of the feasible
# ``z*`` interval are boundaries of the asymptote box (a derived lower asymptote at 0 or upper
# asymptote at 1), where the likelihood is evaluated on the edge of its domain, so a scan point
# for each is entered ``_SCAN_END_INSET`` inside it. The reachability walls, where an asymptote
# reaches the target, are at ``z* = +-inf`` and are never entered. A free extra coordinate takes
# the shares ``_SCAN_SHARES`` of its range and the fit's own value, kept ``_FIT_SHARE_EDGE`` from
# the ends of the range. Where ``z*`` does not vary with the asymptotes, each estimated asymptote
# coordinate takes the shares ``_SCAN_ASYMPTOTE_SHARES``, which include values within
# ``_FIT_SHARE_EDGE`` of both ends. The scan is reduced to its profile in ``log(s)`` (the best
# value over the other coordinates at each ``log(s)``), and the optimiser starts from the scan
# points that attain the ``_MAX_ALTERNATIVE_STARTS`` highest local maxima of that profile.
_SCAN_Z_STAR_STEP = 0.2
_SCAN_Z_STAR_RANGE = 8.0
_SCAN_Z_STAR_FAR = (10.0, 20.0, 40.0)
_SCAN_END_INSET = 0.05
_SCAN_ASYMPTOTE_SHARES = (
    1.0e-6,
    0.01,
    0.1,
    0.25,
    0.4,
    0.5,
    0.6,
    0.75,
    0.9,
    0.99,
    1.0 - 1.0e-6,
)
_MAX_ALTERNATIVE_STARTS = 4
# Interior points at which ``g`` is evaluated once a stride has bracketed a sign change on the
# estimated-asymptote path, so the root solved for is the first crossing seen from the estimate.
_ESTIMATED_BRACKET_SUBDIVISIONS = 8


@dataclass(frozen=True, kw_only=True)
class IntervalResult:
    """The result of :func:`profile_interval` or :func:`delta_interval`: severity-scale
    endpoints, already mapped back from the transformed axis.

    Attributes
    ----------
    lo, hi
        The interval's endpoints on the natural severity scale. ``None`` exactly on the side
        that ``censoring`` reports open.
    censoring
        :data:`~robustmargin.Censoring.NONE`, :data:`~robustmargin.Censoring.OPEN_UPPER` or
        :data:`~robustmargin.Censoring.OPEN_LOWER`.
    warnings
        Free-text notes (for example, an inner fit that failed while stepping outward).
    """

    lo: float | None
    hi: float | None
    censoring: Censoring
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))


# ------------------------------------------------------------------------------------------
# Shared: the link's inverse CDF and density, z*, and the closed-form solve.
# ------------------------------------------------------------------------------------------


def _link_inverse_cdf(q: float, link: str) -> float:
    if link == "probit":
        return float(norm.ppf(q))
    if link == "logit":
        return float(_logit(q))
    return float(math.log(-math.log(1.0 - q)))  # cloglog: F(z) = 1 - exp(-exp(z))


def _link_density(z: float, link: str) -> float:
    if link == "probit":
        return float(norm.pdf(z))
    if link == "logit":
        p = float(expit(z))
        return p * (1.0 - p)
    return float(math.exp(z - math.exp(z)))  # cloglog: F'(z) = exp(z - exp(z))


def _z_star(link: str, *, upper: float, lower: float, target: float) -> float | None:
    """``z* = F^-1((upper - target) / (upper - lower))``, or ``None`` when ``target`` is not
    strictly inside ``(lower, upper)`` (unreachable given these asymptotes)."""
    if not (lower < target < upper):
        return None
    q = (upper - target) / (upper - lower)
    return _link_inverse_cdf(q, link)


def _target_for(definition: Definition, *, upper: float, lower: float) -> float:
    """The target performance, as a function of a *candidate* ``(upper, lower)`` -- used by
    the estimated-asymptote profile path, where the candidate asymptotes move at every inner
    evaluation. ``absolute`` does not depend on them at all.

    ``baseline_multiple(k)`` belongs to an increasing outcome, which is solved on its mirror image
    (performance ``1 - P``, asymptotes ``1 - lower`` and ``1 - upper``). ``upper`` and ``lower``
    here are those mirrored asymptotes, so the baseline ``P(0) = lower`` of the outcome as
    reported is ``1 - upper`` and the mirrored target is ``1 - k * (1 - upper)``.
    """
    if definition.kind == "absolute":
        return definition.value
    if definition.kind == "baseline_fraction":
        return definition.value * upper
    if definition.kind == "baseline_multiple":
        return 1.0 - definition.value * (1.0 - upper)
    return upper - definition.value * (upper - lower)  # relative


def _dq_du_dl(definition: Definition, *, upper: float, lower: float) -> tuple[float, float]:
    """``d(q)/d(upper)`` and ``d(q)/d(lower)`` for ``q = (upper - P*) / (upper - lower)``, per
    definition kind (this module's docstring). For ``baseline_multiple(k)`` the asymptotes are the
    mirrored ones (:func:`_target_for`), where ``upper - P* = (k - 1) * (1 - upper)``."""
    spread = upper - lower
    if definition.kind == "absolute":
        a = definition.value
        return (a - lower) / spread**2, (upper - a) / spread**2
    if definition.kind == "baseline_fraction":
        p = definition.value
        return -(1.0 - p) * lower / spread**2, (1.0 - p) * upper / spread**2
    if definition.kind == "baseline_multiple":
        k = definition.value
        return -(k - 1.0) * (1.0 - lower) / spread**2, (k - 1.0) * (1.0 - upper) / spread**2
    return 0.0, 0.0  # relative: q = p, independent of (upper, lower)


def _to_natural(theta: float, *, scale: str) -> float:
    return math.exp(theta) if scale == "log" else theta


def solve_value(fit: Fit, *, target: float) -> tuple[float, float]:
    """The closed-form threshold, from the fit's own point estimates.

    Parameters
    ----------
    fit
        A ``status is Status.OK`` fit.
    target
        The already-resolved target performance ``P*``. Must satisfy
        ``fit.params["lower"].value < target < fit.params["upper"].value`` -- callers check
        reachability first (:func:`robustmargin.censoring.classify`'s ``UNREACHABLE`` step).

    Returns
    -------
    tuple[float, float]
        ``(theta_hat, value)``: ``theta_hat`` on the transformed axis, ``value`` on the natural
        severity scale.
    """
    assert fit.params is not None
    mu = fit.params["mu"].value
    s = fit.params["s"].value
    upper = fit.params["upper"].value
    lower = fit.params["lower"].value
    z_star = _z_star(fit.link, upper=upper, lower=lower, target=target)
    assert z_star is not None, "solve_value: target must be reachable; the caller must check first"
    theta_hat = mu + s * z_star
    return theta_hat, _to_natural(theta_hat, scale=fit.axis.scale)


def _theta_variance(fit: Fit, *, definition: Definition, target: float) -> float:
    """``Var(theta)`` on the *transformed* axis (``log(severity)`` on a log axis, raw severity on
    a linear one) -- the delta-method gradient assembly this module's docstring describes,
    factored out of :func:`delta_interval` so :mod:`robustmargin.ratio` can reuse the exact same
    computation for its Fieller and log-delta variance inputs instead of reconstructing the
    gradient a second time, which would risk the two drifting apart. Not converted to a
    natural-scale or log-scale standard error here -- callers each do that conversion themselves,
    since Fieller and a single threshold's own delta interval need it on different scales
    (:mod:`robustmargin.ratio`'s module docstring).

    Private: not part of the public API. Callable across modules (``robustmargin.ratio`` imports it
    directly) without being exported from either module's ``__all__``.

    Parameters
    ----------
    fit
        A ``status is Status.OK`` fit.
    definition
        The :class:`~robustmargin.Definition` the threshold solves for (needed for ``d(q)/d(u)``
        and ``d(q)/d(l)`` when an asymptote is estimated).
    target
        The already-resolved target performance.

    Returns
    -------
    float
        ``gradient @ covariance @ gradient``, not yet clamped to ``>= 0`` -- callers do that
        (mirrors this module's own ``max(var_theta, 0.0)`` usage below).
    """
    assert fit.params is not None and fit.covariance is not None
    upper = fit.params["upper"].value
    lower = fit.params["lower"].value
    z_star = _z_star(fit.link, upper=upper, lower=lower, target=target)
    assert z_star is not None, "the caller must have already checked reachability"

    names = fit.covariance.names
    gradient_by_name = {"mu": 1.0, "s": z_star}
    if "upper" in names or "lower" in names:
        density = _link_density(z_star, fit.link)
        dq_du, dq_dl = _dq_du_dl(definition, upper=upper, lower=lower)
        s = fit.params["s"].value
        if "upper" in names:
            gradient_by_name["upper"] = s * dq_du / density
        if "lower" in names:
            gradient_by_name["lower"] = s * dq_dl / density

    gradient = np.array([gradient_by_name[name] for name in names], dtype=np.float64)
    matrix = np.array([[float(v) for v in row] for row in fit.covariance.matrix], dtype=np.float64)
    return float(gradient @ matrix @ gradient)


def _llf_hat(fit: Fit) -> float:
    assert fit.params is not None
    severity, successes, trials = _cells_to_arrays(fit.cells)
    return _full_binomial_log_likelihood(
        severity,
        successes,
        trials,
        mu=fit.params["mu"].value,
        s=fit.params["s"].value,
        upper=fit.params["upper"].value,
        lower=fit.params["lower"].value,
        scale=fit.axis.scale,
        link=fit.link,
    )


def _search_limits(fit: Fit) -> tuple[float, float]:
    """``(lower_limit, upper_limit)`` on theta's own scale: additive ``log(_SEARCH_MULTIPLE)``
    beyond the tested (nonzero, on a log axis) range for a log axis; multiplicative by
    ``_SEARCH_MULTIPLE`` for a linear one."""
    severities = [c.severity for c in fit.cells]
    if fit.axis.scale == "log":
        nonzero = [s for s in severities if s > 0.0]
        transformed = [math.log(s) for s in nonzero]
        return min(transformed) - math.log(_SEARCH_MULTIPLE), max(transformed) + math.log(
            _SEARCH_MULTIPLE
        )
    positive = [s for s in severities if s > 0.0]
    x_min_pos = min(positive) if positive else 1.0e-6
    return x_min_pos / _SEARCH_MULTIPLE, max(severities) * _SEARCH_MULTIPLE


# ------------------------------------------------------------------------------------------
# Profile likelihood: root bracketing, shared by the fixed and estimated-asymptote paths.
# ------------------------------------------------------------------------------------------


class _InnerFitFailed(Exception):
    """The inner constrained fit failed at ``theta`` inside a root solve."""

    def __init__(self, theta: float) -> None:
        super().__init__(theta)
        self.theta = theta


def _bracket_root(
    g: Callable[[float], float | None], theta_hat: float, limit: float, direction: float
) -> tuple[str, float] | tuple[str, float, float]:
    """Step outward from ``theta_hat`` toward ``limit`` (``direction`` is ``+1.0`` or ``-1.0``)
    looking for ``g``'s sign change (``g(theta_hat) < 0``, by the caller's own precondition).

    Returns ``("bracket", a, b)`` with ``g(a) < 0 <= g(b)``, ``("open_failed", last_theta)`` if
    the inner fit failed and no sign change was found before it, or ``("open_limit", limit)`` if
    the search limit was reached with no sign change.

    When the inner fit fails at the end of a stride, the stride is shortened: its interval
    between the last good ``theta`` and the failed one is bisected, at most
    ``_MAX_BACKOFF_HALVINGS`` times. A midpoint where the fit fails becomes the new failed end, a
    midpoint with a sign change closes the bracket, and any other midpoint becomes the new good
    end. If the halvings run out, ``last_theta`` is the last good ``theta``.

    The search never evaluates past an observed failure, and the endpoint is the first crossing
    it observes walking out from ``theta_hat``. A failure region that is never observed, or two
    crossings inside one stride, can still be stepped over, as with the plain stride rule.
    """
    g0 = g(theta_hat)
    if g0 is None:
        return ("open_failed", theta_hat)
    span = abs(limit - theta_hat)
    if span <= 0.0:
        return ("open_limit", limit)
    step = max(span * 1.0e-4, 1.0e-8)
    prev_theta, prev_g = theta_hat, g0
    theta = theta_hat
    for _ in range(_MAX_BRACKET_STEPS):
        step *= 1.7
        theta = theta_hat + direction * step
        reached_limit = (direction > 0.0 and theta >= limit) or (direction < 0.0 and theta <= limit)
        if reached_limit:
            theta = limit
        g_theta = g(theta)
        if g_theta is None:
            good, failed = prev_theta, theta
            for _ in range(_MAX_BACKOFF_HALVINGS):
                mid = 0.5 * (good + failed)
                g_mid = g(mid)
                if g_mid is None:
                    failed = mid
                elif prev_g < 0.0 <= g_mid:
                    return ("bracket", good, mid)
                else:
                    good, prev_g = mid, g_mid
            return ("open_failed", good)
        if prev_g < 0.0 <= g_theta:
            return ("bracket", prev_theta, theta)
        if reached_limit:
            return ("open_limit", theta)
        prev_theta, prev_g = theta, g_theta
    return ("open_limit", theta)


def _first_sign_change(
    g: Callable[[float], float], a: float, b: float, subdivisions: int
) -> tuple[float, float]:
    """The first sub-interval of ``[a, b]`` (counted from ``a``) whose end-points change sign, for
    ``g(a) < 0 <= g(b)`` and ``subdivisions`` equally spaced interior points. The interior points
    are evaluated in order from ``a`` and the search stops at the first sign change, so a point
    beyond it is never evaluated. ``g`` raises where the inner fit fails, and that propagates only
    if the failing point is reached before a sign change. The whole bracket is returned when no
    sub-interval changes sign, which cannot happen for exact values of ``g``."""
    previous = a
    for k in range(1, subdivisions + 1):
        theta = a + (b - a) * k / (subdivisions + 1)
        if g(theta) >= 0.0:
            return previous, theta
        previous = theta
    return previous, b


def _solve_profile_bracket(
    g: Callable[[float], float | None],
    *,
    theta_hat: float,
    lower_limit: float,
    upper_limit: float,
    scale: str,
    subdivisions: int = 0,
) -> IntervalResult:
    """Search each side of ``theta_hat`` for the sign change of ``g`` and map the roots back.

    The upper side steps from ``theta_hat`` up to ``upper_limit`` and the lower side steps down to
    ``lower_limit``. The limits are fixed multiples of the *tested* range, so a fit whose crossing
    lies far outside that range has ``theta_hat`` at or beyond one of them. A side whose limit is
    at or on the wrong side of ``theta_hat`` has nothing to search: stepping "outward" toward it
    would walk back across ``theta_hat`` and find the *other* side's root, reported as this side's
    bound. That side is open instead, ``g`` is not evaluated for it, and the warning names both
    ``theta_hat`` and the limit. Every ``theta`` that is evaluated therefore lies between
    ``theta_hat`` and its own side's limit, and every root found lies on its own side of
    ``theta_hat``.

    With ``subdivisions > 0``, ``g`` is evaluated at up to that many equally spaced interior points
    of the bracket once a stride has closed it, in order from ``theta_hat`` and stopping at the
    first sign change, and the root is solved for in the first sub-interval, counted from
    ``theta_hat``, whose end-points change sign. A failed inner fit at a point after that sign
    change is never reached. A root solver on the whole bracket
    returns some root of it, not necessarily the one nearest the estimate, which matters when the
    profile deviance is not monotone. The default of ``0`` solves on the stride's bracket.
    """
    collected_warnings: list[str] = []

    def _side(direction: float, limit: float, label: str) -> float | None:
        beyond_limit = limit <= theta_hat if direction > 0.0 else limit >= theta_hat
        if beyond_limit:
            collected_warnings.append(
                f"intervals.profile_interval: the estimate (theta_hat={theta_hat!r}) is at or "
                f"beyond the {label} search limit ({limit!r}), so there is nothing to search "
                f"on the {label} side; {label} side left open"
            )
            return None
        outcome = _bracket_root(g, theta_hat, limit, direction)
        if outcome[0] == "open_failed":
            collected_warnings.append(
                "intervals.profile_interval: the inner fit failed while stepping toward the "
                f"{label} search limit (theta={outcome[1]!r}); {label} side left open"
            )
            return None
        if outcome[0] == "open_limit":
            collected_warnings.append(
                "intervals.profile_interval: the profile deviance did not cross the chi-square "
                f"target before the {label} search limit (theta={outcome[1]!r}); {label} side "
                "left open"
            )
            return None
        if len(outcome) != 3:  # pragma: no cover - narrowed by the checks above
            raise AssertionError(f"unexpected bracket outcome {outcome!r}")
        _, a, b = outcome

        def _g_or_raise(theta: float) -> float:
            value = g(theta)
            if value is None:
                raise _InnerFitFailed(theta)
            return value

        try:
            if subdivisions > 0:
                a, b = _first_sign_change(_g_or_raise, a, b, subdivisions)
            root = brentq(_g_or_raise, a, b, xtol=1.0e-10, rtol=1.0e-12, maxiter=200)
        except _InnerFitFailed as failure:
            collected_warnings.append(
                f"intervals.profile_interval: the inner fit failed inside the {label} bracket "
                f"[{a!r}, {b!r}] while solving for the root (theta={failure.theta!r}); "
                f"{label} side left open"
            )
            return None
        return _to_natural(root, scale=scale)

    hi = _side(1.0, upper_limit, "upper")
    lo = _side(-1.0, lower_limit, "lower")

    if hi is None and lo is None:
        # Neither side crossed the chi-square target: the design does not constrain the
        # threshold. v1 has no `Censoring` member for "open on both sides", so the estimate is
        # reported with no bounds and `censoring=NONE`, with the reason already collected in
        # `warnings`. Reporting a bound at the search limit would be inventing one.
        return IntervalResult(
            lo=None, hi=None, censoring=Censoring.NONE, warnings=tuple(collected_warnings)
        )
    if hi is None:
        return IntervalResult(
            lo=lo, hi=None, censoring=Censoring.OPEN_UPPER, warnings=tuple(collected_warnings)
        )
    if lo is None:
        return IntervalResult(
            lo=None, hi=hi, censoring=Censoring.OPEN_LOWER, warnings=tuple(collected_warnings)
        )
    return IntervalResult(
        lo=lo, hi=hi, censoring=Censoring.NONE, warnings=tuple(collected_warnings)
    )


# ------------------------------------------------------------------------------------------
# Profile likelihood: fixed-asymptote path (the no-intercept offset-trick GLM).
# ------------------------------------------------------------------------------------------


def _fixed_profile_llf(
    theta: float,
    *,
    nonzero_severity: NDArray[np.float64],
    nonzero_successes: NDArray[np.float64],
    nonzero_trials: NDArray[np.float64],
    all_severity: NDArray[np.float64],
    all_successes: NDArray[np.float64],
    all_trials: NDArray[np.float64],
    upper: float,
    lower: float,
    link: str,
    scale: str,
    z_star: float,
    direct: Callable[[], float | None],
) -> float | None:
    """``ell_p(theta)`` for the fixed-asymptote path: fit the no-intercept offset GLM at this
    ``theta`` (an *optimiser*, per this module's docstring), then evaluate the full likelihood
    at the resulting ``(mu, s)``. Returns ``None`` (an inner-fit failure) when its fitted slope
    is non-positive -- a wrong-sign slope means the profile value is the flat-curve boundary, so
    that side never closes there.

    The GLM's clipped iteratively reweighted least squares can stop short of the maximum, or
    report non-convergence where a maximum exists. When it does not converge, or its slope
    fails the Newton-decrement check on the stable log-likelihood (``lambda^2 / 2`` above the
    shared tolerance), ``direct`` recomputes ``ell_p(theta)`` by direct constrained maximisation
    of the same model.
    """
    covariate = (_transform(nonzero_severity, scale=scale) - theta).reshape(-1, 1)
    failures = nonzero_trials - nonzero_successes
    endog = failures / nonzero_trials
    offset = np.full(nonzero_severity.shape, z_star)

    try:
        with warnings.catch_warnings():
            # On grouped binomial data statsmodels' PerfectSeparationWarning fires on any
            # exact fit, so it carries no information here and maps to no status.
            # Separation is decided by the pre-fit overlap check, before this function runs.
            warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
            model = GLM(
                endog,
                covariate,
                offset=offset,
                family=families.Binomial(link=_STATSMODELS_LINKS[link]),
                var_weights=nonzero_trials,
            )
            result = model.fit()
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not bool(result.converged):
        return direct()
    beta = float(result.params[0])
    if not beta > 0.0:
        return None

    # The decrement is parametrisation-invariant; `log(beta)` keeps its finite-difference step
    # relative, whatever the units of the severity axis make `beta` (the inverse slope scale).
    def _log_beta_loglike(x: NDArray[np.float64]) -> float:
        slope_scale = math.exp(-float(x[0]))
        return _full_binomial_log_likelihood(
            all_severity,
            all_successes,
            all_trials,
            mu=theta - slope_scale * z_star,
            s=slope_scale,
            upper=upper,
            lower=lower,
            scale=scale,
            link=link,
        )

    s = 1.0 / beta
    mu = theta - s * z_star
    glm_llf = _full_binomial_log_likelihood(
        all_severity,
        all_successes,
        all_trials,
        mu=mu,
        s=s,
        upper=upper,
        lower=lower,
        scale=scale,
        link=link,
    )
    if _newton_decrement_half(_log_beta_loglike, np.array([math.log(beta)])) <= (
        _NEWTON_DECREMENT_TOLERANCE
    ):
        return glm_llf
    # The direct value is a maximum over the same set, so the larger of the two is the better
    # estimate of the profile; a failed direct maximisation leaves the converged GLM value.
    direct_llf = direct()
    if direct_llf is None or not math.isfinite(glm_llf):
        return direct_llf if direct_llf is not None else glm_llf
    return max(glm_llf, direct_llf)


def _profile_interval_fixed(
    fit: Fit, *, definition: Definition, target: float, level: float, theta_hat: float
) -> IntervalResult:
    assert fit.params is not None
    upper = fit.params["upper"].value
    lower = fit.params["lower"].value
    z_star = _z_star(fit.link, upper=upper, lower=lower, target=target)
    assert z_star is not None, "the caller must have already checked reachability"

    all_severity, all_successes, all_trials = _cells_to_arrays(fit.cells)
    nonzero_cells = tuple(
        c for c in fit.cells if not (fit.axis.scale == "log" and c.severity == 0.0)
    )
    nz_severity, nz_successes, nz_trials = _cells_to_arrays(nonzero_cells)

    llf_hat = _llf_hat(fit)
    chi2_target = float(chi2.ppf(level, 1))

    replaced = False

    def g(theta: float) -> float | None:
        def direct() -> float | None:
            nonlocal replaced
            replaced = True
            return _estimated_profile_llf(theta, fit=fit, definition=definition)

        llf_p = _fixed_profile_llf(
            theta,
            nonzero_severity=nz_severity,
            nonzero_successes=nz_successes,
            nonzero_trials=nz_trials,
            all_severity=all_severity,
            all_successes=all_successes,
            all_trials=all_trials,
            upper=upper,
            lower=lower,
            link=fit.link,
            scale=fit.axis.scale,
            z_star=z_star,
            direct=direct,
        )
        if llf_p is None:
            return None
        return 2.0 * (llf_hat - llf_p) - chi2_target

    g0 = g(theta_hat)
    if g0 is None or not g0 < 0.0:
        raise ValueError(
            "intervals.profile_interval: the profile deviance at theta_hat did not come out "
            f"below the chi-square target (g(theta_hat)={g0!r}); the inner optimiser may have "
            "beaten the reported MLE"
        )
    if not (upper == 1.0 and lower == 0.0) and (g0 + chi2_target) / 2.0 < (
        -_NEWTON_DECREMENT_TOLERANCE
    ):
        raise ValueError(
            "intervals.profile_interval: the constrained fit at theta_hat reached a log-likelihood "
            f"{-(g0 + chi2_target) / 2.0!r} above the reported maximum, so the fit is not at its "
            "MLE; refit before computing an interval"
        )

    lower_limit, upper_limit = _search_limits(fit)
    result = _solve_profile_bracket(
        g,
        theta_hat=theta_hat,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        scale=fit.axis.scale,
    )
    if not replaced:
        return result
    note = (
        "intervals.profile_interval: the profile inner fit was replaced by direct "
        "constrained maximisation of the same model at some theta, because the GLM did not "
        "reach the maximum there"
    )
    return replace(result, warnings=(*result.warnings, note))


# ------------------------------------------------------------------------------------------
# Profile likelihood: estimated-asymptote path (mu-substituted GenericLikelihoodModel).
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _MuSubstitutedLogLikelihood:
    """``GenericLikelihoodModel``'s ``loglike=`` callable for a fixed candidate ``theta``: the
    free parameters are ``log(s)`` plus whichever asymptote logits the original fit estimated
    (same parametrisation as ``models._unpack_optimizer_params``, with ``mu`` removed -- it is
    substituted at every evaluation from the *candidate* asymptotes, since ``z*`` moves with
    them here (this module's docstring)). Returns a very negative constant, never raises, when a
    candidate makes the target unreachable or yields a non-finite ``mu``, so the optimiser is
    steered away from that region rather than failing outright (mirrors
    ``models._GenericLogLikelihood``'s own robustness).
    """

    theta: float
    definition: Definition
    severity: NDArray[np.float64]
    successes: NDArray[np.float64]
    trials: NDArray[np.float64]
    scale: str
    link: str
    estimate_lower: bool
    estimate_upper: bool
    lower_value: float
    upper_value: float

    def asymptotes(self, coordinates: NDArray[np.float64]) -> tuple[float, float]:
        """The ``(lower, upper)`` asymptotes at the asymptote coordinates (``params[1:]``): the
        fixed values, and for each estimated one its logit-parametrised value. May be non-finite
        or not ordered."""
        idx = 0
        if self.estimate_lower:
            a = float(coordinates[idx])
            idx += 1
            lower = float(expit(a)) if self.estimate_upper else float(self.upper_value * expit(a))
        else:
            lower = self.lower_value
        if self.estimate_upper:
            b = float(coordinates[idx])
            upper = float(lower + (1.0 - lower) * expit(b))
        else:
            upper = self.upper_value
        return lower, upper

    def __call__(self, params: NDArray[np.float64]) -> float:
        with np.errstate(over="ignore"):
            s = float(np.exp(params[0]))
        if not (math.isfinite(s) and s > 0.0):
            return _UNREACHABLE_LOGLIKELIHOOD
        lower, upper = self.asymptotes(params[1:])

        if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
            return _UNREACHABLE_LOGLIKELIHOOD
        target = _target_for(self.definition, upper=upper, lower=lower)
        z_star = _z_star(self.link, upper=upper, lower=lower, target=target)
        if z_star is None or not math.isfinite(z_star):
            return _UNREACHABLE_LOGLIKELIHOOD
        mu = self.theta - s * z_star
        if not math.isfinite(mu):
            return _UNREACHABLE_LOGLIKELIHOOD
        return _full_binomial_log_likelihood(
            self.severity,
            self.successes,
            self.trials,
            mu=mu,
            s=s,
            upper=upper,
            lower=lower,
            scale=self.scale,
            link=self.link,
        )


def _optimizer_start_from_fit(fit: Fit) -> NDArray[np.float64]:
    """A starting point for the constrained (mu-substituted) optimiser, from the *fit's own*
    converged natural-scale parameters -- a good start, since the profile is centred on it, but
    only a start (as for the outer fit, a starting point is never itself reported)."""
    assert fit.params is not None
    s = fit.params["s"].value
    lower_param = fit.params["lower"]
    upper_param = fit.params["upper"]
    eps = 1.0e-6
    start = [math.log(s)]
    if not lower_param.fixed:
        if not upper_param.fixed:
            a0 = float(_logit(np.clip(lower_param.value, eps, 1.0 - eps)))
        else:
            a0 = float(_logit(np.clip(lower_param.value / upper_param.value, eps, 1.0 - eps)))
        start.append(a0)
    if not upper_param.fixed:
        denominator = max(1.0 - lower_param.value, 1.0e-6)
        fraction = float(
            np.clip((upper_param.value - lower_param.value) / denominator, eps, 1.0 - eps)
        )
        start.append(float(_logit(fraction)))
    return np.array(start, dtype=np.float64)


def _cdf_and_survival(z: float, link: str) -> tuple[float, float]:
    """``(F(z), 1 - F(z))`` for one finite ``z``, each from its own formula so neither is a
    difference from one (``scipy.special.ndtr`` for probit, the logistic function and its
    reflection for logit, ``-expm1(-exp(z))`` and ``exp(-exp(z))`` for cloglog), without the
    per-call overhead of the ``scipy.stats`` distribution objects. The values are those of
    ``robustmargin.models._f_cdf`` and ``_f_survival``."""
    if link == "probit":
        return float(ndtr(z)), float(ndtr(-z))
    if link == "logit":
        return float(expit(z)), float(expit(-z))
    with np.errstate(over="ignore", under="ignore"):
        inner = float(np.exp(z))
        return float(-np.expm1(-inner)), float(np.exp(-inner))


def _log_cdf_and_survival(
    z: NDArray[np.float64], link: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(log F(z), log(1 - F(z)))`` elementwise, as ``robustmargin.models._f_log_cdf`` and
    ``_f_log_survival`` give them, with ``scipy.special.log_ndtr`` for probit in place of the
    ``scipy.stats`` distribution methods (same function, less per-call overhead)."""
    if link == "probit":
        return np.asarray(log_ndtr(z), dtype=np.float64), np.asarray(log_ndtr(-z), dtype=np.float64)
    return _f_log_cdf(z, link), _f_log_survival(z, link)


def _ratio_or_inf(numerator: float, denominator: float) -> float:
    """``numerator / denominator`` for a bound on a positive length: ``+inf`` for a positive
    numerator over an exact zero, and ``0.0`` (no room) for a non-positive numerator."""
    if numerator <= 0.0:
        return 0.0
    return math.inf if denominator == 0.0 else numerator / denominator


def _zstar_chart_applies(
    definition: Definition,
    *,
    estimate_lower: bool,
    estimate_upper: bool,
    lower_value: float,
    upper_value: float,
) -> bool:
    """Whether ``z*`` varies with the free asymptotes, so a chart in ``z*`` is a different chart.

    It does not for ``relative`` (``F(z*) = 1 - p``), with both asymptotes fixed, or for
    ``baseline_fraction`` with the lower asymptote fixed at zero (``F(z*) = 1 - p``).
    """
    if definition.kind == "relative" or not (estimate_lower or estimate_upper):
        return False
    return not (
        definition.kind == "baseline_fraction" and not estimate_lower and lower_value == 0.0
    )


def _zstar_chart_asymptotes(
    z_star: float,
    extra: NDArray[np.float64],
    *,
    definition: Definition,
    link: str,
    estimate_lower: bool,
    estimate_upper: bool,
    lower_value: float,
    upper_value: float,
) -> tuple[float, float] | None:
    """The ``(lower, upper)`` asymptotes that put the curve through the target at ``z*``, or
    ``None`` when none is feasible (``0 <= lower < upper <= 1`` fails). Never raises.

    With ``F = F(z*)`` and ``S = 1 - F`` each computed from its own formula, the curve
    ``u - (u - l) F`` equals the target exactly when, per definition (``a`` the level, ``f`` the
    fraction, ``k`` the multiple; ``baseline_multiple`` is in the mirrored frame):

    - ``absolute(a)``: ``S u + F l = a``. One estimated: ``l = (a - S u) / F`` or
      ``u = (a - F l) / S``. Both: ``u = a + t F``, ``l = a - t S`` with
      ``t = t_max expit(c)``, ``t_max = min((1 - a) / F, a / S)`` (a division by exact zero is
      ``+inf``) and ``extra = [c]``.
    - ``baseline_fraction(f)``: ``F l = (f - S) u``. ``l = u (f - S) / F``, or
      ``u = F l / (f - S)`` with only ``u`` estimated; with both, ``u = expit(extra[0])``.
    - ``baseline_multiple(k)``: ``F l = (k - S) u - (k - 1)``. ``l = ((k - S) u - (k - 1)) / F``,
      or ``u = (F l + (k - 1)) / ((k - 1) + F)`` with only ``u`` estimated; with both,
      ``l = expit(extra[0])`` and ``u`` is derived, which keeps the whole range of ``z*`` feasible.

    A derived ``u`` is clipped at 1: where ``F`` is of the order of ``1e-17`` the quotient can
    round to one ulp above it.
    """
    if not math.isfinite(z_star):
        return None
    f, s = _cdf_and_survival(z_star, link)
    if not (math.isfinite(f) and math.isfinite(s)):
        return None
    lower = lower_value
    upper = upper_value
    v = definition.value
    if definition.kind == "absolute":
        if estimate_lower and estimate_upper:
            t_max = min(_ratio_or_inf(1.0 - v, f), _ratio_or_inf(v, s))
            t = t_max * float(expit(extra[0]))
            if not (0.0 < t < math.inf):
                return None
            # The construction keeps both inside [0, 1]; rounding can overshoot by an ulp.
            upper = min(v + t * f, 1.0)
            lower = max(v - t * s, 0.0)
        elif estimate_lower:
            if f <= 0.0:
                return None
            lower = (v - s * upper) / f
        else:
            if s <= 0.0:
                return None
            upper = (v - f * lower) / s
    elif definition.kind == "baseline_fraction":
        if estimate_upper and estimate_lower:
            upper = float(expit(extra[0]))
        if estimate_lower:
            if f <= 0.0:
                return None
            lower = upper * (v - s) / f
        else:
            if not v - s > 0.0:
                return None
            upper = f * lower / (v - s)
    elif definition.kind == "baseline_multiple":
        if estimate_upper and estimate_lower:
            lower = float(expit(extra[0]))
            upper = min((f * lower + (v - 1.0)) / ((v - 1.0) + f), 1.0)
        elif estimate_lower:
            if f <= 0.0:
                return None
            lower = ((v - s) * upper - (v - 1.0)) / f
        else:
            upper = min((f * lower + (v - 1.0)) / ((v - 1.0) + f), 1.0)
    else:
        return None
    if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 <= lower < upper <= 1.0):
        return None
    return lower, upper


def _zstar_feasible_interval(
    definition: Definition,
    *,
    link: str,
    estimate_lower: bool,
    estimate_upper: bool,
    lower_value: float,
    upper_value: float,
) -> tuple[float, float] | None:
    """The ``z*`` values at which :func:`_zstar_chart_asymptotes` is feasible, as ``(lo, hi)``
    (infinite where unbounded), or ``None`` when there are none. Closed form in ``F(z*)``:

    - ``absolute(a)``: lower only, ``F >= 1 - a / u`` (needs ``a < u``); upper only,
      ``F <= (1 - a) / (1 - l)`` (needs ``l < a < 1``); both, every ``z*`` (needs ``0 < a < 1``).
    - ``baseline_fraction(f)``: lower estimated, ``F >= 1 - f``; upper only,
      ``F >= (1 - f) / (1 - l)``.
    - ``baseline_multiple(k)``: lower only, ``F >= (k - 1)(1 - u) / u`` with ``u`` the fixed upper
      asymptote; upper only, and both, every ``z*``.

    The bound on ``F`` becomes a bound on ``z*`` through the link's inverse CDF.
    """
    v = definition.value
    q_lo, q_hi = 0.0, 1.0
    if definition.kind == "absolute":
        if estimate_lower and estimate_upper:
            if not 0.0 < v < 1.0:
                return None
        elif estimate_lower:
            if not 0.0 < v < upper_value:
                return None
            q_lo = 1.0 - v / upper_value
        else:
            if not lower_value < v < 1.0:
                return None
            q_hi = (1.0 - v) / (1.0 - lower_value)
    elif definition.kind == "baseline_fraction":
        if estimate_lower:
            q_lo = 1.0 - v
        else:
            if not lower_value < 1.0:
                return None
            q_lo = (1.0 - v) / (1.0 - lower_value)
    elif definition.kind == "baseline_multiple":
        if estimate_lower and not estimate_upper:
            u = upper_value
            if not 0.0 < u < 1.0:
                return None
            q_lo = (v - 1.0) * (1.0 - u) / u
    else:
        return None
    if not q_lo < q_hi:
        return None

    def to_z(q: float) -> float:
        if q <= 0.0:
            return -math.inf
        if q >= 1.0:
            return math.inf
        return _link_inverse_cdf(q, link)

    return to_z(q_lo), to_z(q_hi)


@dataclass(frozen=True)
class _PreparedCells:
    """Cell arrays for :meth:`_ZStarLogLikelihood.batch_log_likelihood`: the transformed levels
    and counts of the cells that carry a covariate, and the totals of the log-axis zero-severity
    control (whose curve value is the upper asymptote)."""

    levels: NDArray[np.float64]
    successes: NDArray[np.float64]
    failures: NDArray[np.float64]
    control_successes: float
    control_failures: float


@dataclass(frozen=True)
class _ZStarLogLikelihood:
    """``_MuSubstitutedLogLikelihood`` in the chart ``(log s, z*, *extra)``: the asymptotes are
    derived from ``z*`` by :func:`_zstar_chart_asymptotes` and ``mu = theta - s z*``. Unlike the
    logit chart it can represent an estimated asymptote that has reached the target. Its value is
    ``_full_binomial_log_likelihood`` at the natural-scale parameters, the same function as the
    other chart. Returns the unreachable constant, never raises, where no feasible asymptote
    exists at ``z*``."""

    theta: float
    definition: Definition
    severity: NDArray[np.float64]
    successes: NDArray[np.float64]
    trials: NDArray[np.float64]
    scale: str
    link: str
    estimate_lower: bool
    estimate_upper: bool
    lower_value: float
    upper_value: float

    @cached_property
    def _cells(self) -> _PreparedCells:
        """The cell arrays the batch evaluation uses, built once per chart."""
        if self.scale == "log":
            control = self.severity == 0.0
        else:
            control = np.zeros(self.severity.shape, dtype=bool)
        free = ~control
        failures = self.trials - self.successes
        return _PreparedCells(
            levels=_transform(self.severity[free], scale=self.scale),
            successes=self.successes[free],
            failures=failures[free],
            control_successes=float(self.successes[control].sum()),
            control_failures=float(failures[control].sum()),
        )

    def batch_log_likelihood(
        self,
        mu: NDArray[np.float64],
        s: NDArray[np.float64],
        upper: NDArray[np.float64],
        lower: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """The log-likelihood at many ``(mu, s, upper, lower)`` points at once, without the
        constant ``lchoose`` term. It is used only to rank the scan that seeds the optimiser's
        starts, never as a value that is reported or compared with a fit: that is
        :meth:`__call__`, which is ``robustmargin.models._full_binomial_log_likelihood`` itself.
        The curve and its log-space combination follow it, with the log-axis zero control entered
        at ``p = u``. Values may be ``-inf`` or NaN, as there."""
        cells = self._cells
        with np.errstate(all="ignore"):
            z = (cells.levels[None, :] - mu[:, None]) / s[:, None]
            log_f, log_s = _log_cdf_and_survival(z, self.link)
            log_range = np.log(upper - lower)[:, None]
            log_p = np.logaddexp(np.log(lower)[:, None], log_range + log_s)
            log_q = np.logaddexp(np.log(1.0 - upper)[:, None], log_range + log_f)
            total = np.where(cells.successes > 0.0, cells.successes * log_p, 0.0).sum(axis=1)
            total += np.where(cells.failures > 0.0, cells.failures * log_q, 0.0).sum(axis=1)
            if cells.control_successes > 0.0:
                total += cells.control_successes * np.log(upper)
            if cells.control_failures > 0.0:
                total += cells.control_failures * np.log(1.0 - upper)
        return np.asarray(total, dtype=np.float64)

    def __call__(self, params: NDArray[np.float64]) -> float:
        with np.errstate(over="ignore"):
            s = float(np.exp(params[0]))
        z_star = float(params[1])
        if not (math.isfinite(s) and s > 0.0):
            return _UNREACHABLE_LOGLIKELIHOOD
        asymptotes = _zstar_chart_asymptotes(
            z_star,
            params[2:],
            definition=self.definition,
            link=self.link,
            estimate_lower=self.estimate_lower,
            estimate_upper=self.estimate_upper,
            lower_value=self.lower_value,
            upper_value=self.upper_value,
        )
        if asymptotes is None:
            return _UNREACHABLE_LOGLIKELIHOOD
        lower, upper = asymptotes
        mu = self.theta - s * z_star
        if not math.isfinite(mu):
            return _UNREACHABLE_LOGLIKELIHOOD
        return _full_binomial_log_likelihood(
            self.severity,
            self.successes,
            self.trials,
            mu=mu,
            s=s,
            upper=upper,
            lower=lower,
            scale=self.scale,
            link=self.link,
        )


@dataclass(frozen=True)
class _FaceLogLikelihood:
    """The profile's inner problem on a box face of the asymptotes, in the chart
    ``(log s, *extra)``.

    At a finite end of the feasible ``z*`` interval the derived asymptote reaches its box face
    (a lower asymptote at 0 or an upper asymptote at 1), and beyond it no asymptote is feasible,
    so the likelihood has a cliff there that an optimiser started inside cannot cross. This chart
    pins that asymptote instead. The curve ``u - (u - l) F(z)`` passes through the target
    ``T(u, l)`` exactly when ``F (u - l) = u - T``, and with ``l = 0`` or ``u = 1`` this fixes
    ``F`` and so ``z*``, per definition (``a`` the level, ``f`` the fraction, ``k`` the multiple,
    the last in its mirrored frame where ``u - T = (k - 1)(1 - u)``):

    - ``absolute(a)``, lower only estimated: ``l = 0``, ``F = 1 - a / u``, ``u`` fixed.
    - ``absolute(a)``, upper only estimated: ``u = 1``, ``F = (1 - a) / (1 - l)``, ``l`` fixed.
    - ``baseline_fraction(f)``, lower estimated: ``l = 0``, ``F = 1 - f``, for any ``u``: ``u`` is
      fixed or, with both estimated, the free coordinate ``logit(u)``.
    - ``baseline_fraction(f)``, upper only estimated: ``u = 1``, ``F = (1 - f) / (1 - l)``.
    - ``baseline_multiple(k)``, lower only estimated: ``l = 0``, ``F = (k - 1)(1 - u) / u``.

    These are the finite ends of :func:`_zstar_feasible_interval`, which gives ``z*`` as the
    link's inverse CDF of ``F`` there; ``mu = theta - s z*``. Its value is
    ``_full_binomial_log_likelihood`` at that point, a feasible one, so it is attained by a model
    with an asymptote on the box face. Returns the unreachable constant, never raises, where the
    asymptotes are not ordered."""

    theta: float
    severity: NDArray[np.float64]
    successes: NDArray[np.float64]
    trials: NDArray[np.float64]
    scale: str
    link: str
    z_star: float
    lower_value: float
    upper_value: float
    free_upper: bool

    def asymptotes(self, coordinates: NDArray[np.float64]) -> tuple[float, float]:
        """The ``(lower, upper)`` asymptotes: the pinned face, the fixed value and, when
        ``free_upper``, ``upper = expit(coordinates[0])``."""
        upper = float(expit(coordinates[0])) if self.free_upper else self.upper_value
        return self.lower_value, upper

    def __call__(self, params: NDArray[np.float64]) -> float:
        with np.errstate(over="ignore"):
            s = float(np.exp(params[0]))
        if not (math.isfinite(s) and s > 0.0):
            return _UNREACHABLE_LOGLIKELIHOOD
        lower, upper = self.asymptotes(params[1:])
        if not (math.isfinite(upper) and 0.0 <= lower < upper <= 1.0):
            return _UNREACHABLE_LOGLIKELIHOOD
        mu = self.theta - s * self.z_star
        if not math.isfinite(mu):
            return _UNREACHABLE_LOGLIKELIHOOD
        return _full_binomial_log_likelihood(
            self.severity,
            self.successes,
            self.trials,
            mu=mu,
            s=s,
            upper=upper,
            lower=lower,
            scale=self.scale,
            link=self.link,
        )


def _tested_span(fit: Fit) -> float:
    """The range of the finite tested levels on the axis scale (a log-axis zero control has no
    finite level), or ``1.0`` when there are fewer than two distinct ones."""
    severity, _, _ = _cells_to_arrays(fit.cells)
    levels = severity[severity > 0.0] if fit.axis.scale == "log" else severity
    transformed = _transform(levels, scale=fit.axis.scale)
    if transformed.size < 2:
        return 1.0
    span = float(np.max(transformed) - np.min(transformed))
    return span if span > 0.0 else 1.0


def _scan_extra_values(
    fit: Fit, chart: _ZStarLogLikelihood, z_star: float, link: str
) -> list[tuple[float, ...]]:
    """The values of the free coordinate of the ``z*`` chart scanned at ``z_star``: none when no
    coordinate is free (one empty tuple), else the logits of :func:`_share_logits`. The coordinate
    is, per definition, the logit of the share ``t / t_max`` (``absolute``), of the upper
    asymptote (``baseline_fraction``) or of the lower asymptote (``baseline_multiple``)."""
    if not (chart.estimate_lower and chart.estimate_upper):
        return [()]
    assert fit.params is not None
    u_hat = fit.params["upper"].value
    l_hat = fit.params["lower"].value
    if chart.definition.kind == "absolute":
        f, s = _cdf_and_survival(z_star, link)
        a = chart.definition.value
        t_max = min(_ratio_or_inf(1.0 - a, f), _ratio_or_inf(a, s))
        if not 0.0 < t_max < math.inf:
            return []
        hat = (u_hat - l_hat) / t_max
    elif chart.definition.kind == "baseline_fraction":
        hat = u_hat
    else:
        hat = l_hat
    return [(logit,) for logit in _share_logits(hat)]


def _top_local_maxima(profile: NDArray[np.float64]) -> list[int]:
    """Indices into ``profile`` of its highest local maxima, best first, at most
    :data:`_MAX_ALTERNATIVE_STARTS` (the rule is :func:`robustmargin.models._top_local_maxima`)."""
    return _top_maxima(profile, _MAX_ALTERNATIVE_STARTS)


def _scan_picks(
    z_chart: _ZStarLogLikelihood,
    theta: float,
    log_s: NDArray[np.float64],
    z_values: list[float],
    lower_values: list[float],
    upper_values: list[float],
) -> tuple[list[int], NDArray[np.intp]]:
    """Scan ``log s`` against the combinations ``(z*, lower, upper)`` given as parallel lists,
    ranking by :meth:`_ZStarLogLikelihood.batch_log_likelihood`. Returns the indices into ``log_s``
    of the highest local maxima of the profile in ``log s`` (:func:`_top_local_maxima`) and, for
    every ``log s``, the index of the best combination."""
    z_arr = np.array(z_values)
    lower_arr = np.array(lower_values)
    upper_arr = np.array(upper_values)
    count = z_arr.size
    grid_s = np.repeat(log_s, count)
    grid_index = np.tile(np.arange(count), log_s.size)
    s_arr = np.exp(grid_s)
    mu_arr = theta - s_arr * z_arr[grid_index]
    values = z_chart.batch_log_likelihood(
        mu_arr, s_arr, upper_arr[grid_index], lower_arr[grid_index]
    )
    surface = np.where(np.isfinite(values), values, -np.inf).reshape(log_s.size, count)
    best_combo = np.argmax(surface, axis=1)
    return (
        _top_local_maxima(surface[np.arange(log_s.size), best_combo]),
        best_combo,
    )


def _face_charts(
    first_chart: _MuSubstitutedLogLikelihood, interval: tuple[float, float]
) -> list[_FaceLogLikelihood]:
    """The box-face problems of a chart case, one for each finite end of the feasible ``z*``
    interval (:class:`_FaceLogLikelihood`). An estimated lower asymptote is the one that reaches
    its face at 0; with only the upper estimated, the upper reaches 1."""
    charts: list[_FaceLogLikelihood] = []
    for z_end in interval:
        if not math.isfinite(z_end):
            continue
        if first_chart.estimate_lower:
            lower, upper = 0.0, first_chart.upper_value
            free_upper = first_chart.estimate_upper
        else:
            lower, upper = first_chart.lower_value, 1.0
            free_upper = False
        if not free_upper and not 0.0 <= lower < upper <= 1.0:
            continue
        charts.append(
            _FaceLogLikelihood(
                theta=first_chart.theta,
                severity=first_chart.severity,
                successes=first_chart.successes,
                trials=first_chart.trials,
                scale=first_chart.scale,
                link=first_chart.link,
                z_star=z_end,
                lower_value=lower,
                upper_value=upper,
                free_upper=free_upper,
            )
        )
    return charts


_Start = tuple[
    _MuSubstitutedLogLikelihood | _ZStarLogLikelihood | _FaceLogLikelihood, NDArray[np.float64]
]


def _alternative_starts(
    fit: Fit, first_chart: _MuSubstitutedLogLikelihood, first_start: NDArray[np.float64]
) -> list[_Start]:
    """The starts for the optimiser's alternative runs, chosen by deterministic scans of
    log-likelihood values (no optimiser). Each scan is over ``log s`` (steps of
    :data:`_SCAN_LOG_S_STEP`, from eight orders of magnitude below the tested range's span to
    three above) and one more set of coordinates, and is reduced to its profile in ``log s``, the
    best finite value at each ``log s``. The starts are the scan points attaining the highest local
    maxima of that profile (:func:`_top_local_maxima`): fewer maxima give fewer starts.

    Where :func:`_zstar_chart_applies`, the ``z*`` chart is scanned over ``z*`` (steps of
    :data:`_SCAN_Z_STAR_STEP` over ``+-``:data:`_SCAN_Z_STAR_RANGE` and the values of
    :data:`_SCAN_Z_STAR_FAR`, inside the feasible interval, and each finite end of that interval
    moved :data:`_SCAN_END_INSET` inside it) and the free coordinate (:func:`_share_logits`). Each
    finite end of the interval is also a box face, scanned in the :class:`_FaceLogLikelihood`
    chart over ``log s`` and its free coordinate. Otherwise, where ``z*`` does not move with the
    asymptotes, the first chart is scanned over ``log s`` crossed with every estimated asymptote
    coordinate at the logits of :data:`_SCAN_ASYMPTOTE_SHARES`.
    """
    theta = first_chart.theta
    log_s = _scan_log_s_grid(_tested_span(fit))
    z_chart = _ZStarLogLikelihood(
        theta=theta,
        definition=first_chart.definition,
        severity=first_chart.severity,
        successes=first_chart.successes,
        trials=first_chart.trials,
        scale=first_chart.scale,
        link=first_chart.link,
        estimate_lower=first_chart.estimate_lower,
        estimate_upper=first_chart.estimate_upper,
        lower_value=first_chart.lower_value,
        upper_value=first_chart.upper_value,
    )
    if not _zstar_chart_applies(
        first_chart.definition,
        estimate_lower=first_chart.estimate_lower,
        estimate_upper=first_chart.estimate_upper,
        lower_value=first_chart.lower_value,
        upper_value=first_chart.upper_value,
    ):
        return _asymptote_scan_starts(first_chart, z_chart, log_s)

    interval = _zstar_feasible_interval(
        first_chart.definition,
        link=first_chart.link,
        estimate_lower=first_chart.estimate_lower,
        estimate_upper=first_chart.estimate_upper,
        lower_value=first_chart.lower_value,
        upper_value=first_chart.upper_value,
    )
    if interval is None:
        return []
    starts: list[_Start] = []
    lo, hi = interval
    steps = round(_SCAN_Z_STAR_RANGE / _SCAN_Z_STAR_STEP)
    grid = [round(k * _SCAN_Z_STAR_STEP, 10) for k in range(-steps, steps + 1)]
    grid += [sign * z for z in _SCAN_Z_STAR_FAR for sign in (1.0, -1.0)]
    z_values = {z for z in grid if lo < z < hi}
    inset = _SCAN_END_INSET
    if math.isfinite(lo) and math.isfinite(hi):
        inset = min(inset, (hi - lo) / 4.0)
    if math.isfinite(lo):
        z_values.add(lo + inset)
    if math.isfinite(hi):
        z_values.add(hi - inset)
    combos: list[tuple[float, tuple[float, ...], float, float]] = []
    for z_star in sorted(z_values):
        for extra in _scan_extra_values(fit, z_chart, z_star, first_chart.link):
            asymptotes = _zstar_chart_asymptotes(
                z_star,
                np.array(extra, dtype=np.float64),
                definition=first_chart.definition,
                link=first_chart.link,
                estimate_lower=first_chart.estimate_lower,
                estimate_upper=first_chart.estimate_upper,
                lower_value=first_chart.lower_value,
                upper_value=first_chart.upper_value,
            )
            if asymptotes is not None:
                combos.append((z_star, extra, asymptotes[0], asymptotes[1]))
    if combos:
        picks, best_combo = _scan_picks(
            z_chart,
            theta,
            log_s,
            [c[0] for c in combos],
            [c[2] for c in combos],
            [c[3] for c in combos],
        )
        starts += [
            (z_chart, np.array([log_s[i], combos[best_combo[i]][0], *combos[best_combo[i]][1]]))
            for i in picks
        ]

    for face in _face_charts(first_chart, interval):
        assert fit.params is not None
        extras: list[tuple[float, ...]]
        if face.free_upper:
            extras = [(logit,) for logit in _share_logits(fit.params["upper"].value)]
        else:
            extras = [()]
        face_picks, face_best = _scan_picks(
            z_chart,
            theta,
            log_s,
            [face.z_star] * len(extras),
            [face.lower_value] * len(extras),
            [float(expit(e[0])) if face.free_upper else face.upper_value for e in extras],
        )
        starts += [(face, np.array([log_s[i], *extras[face_best[i]]])) for i in face_picks]
    return starts


def _asymptote_scan_starts(
    first_chart: _MuSubstitutedLogLikelihood,
    z_chart: _ZStarLogLikelihood,
    log_s: NDArray[np.float64],
) -> list[_Start]:
    """The scan where ``z*`` does not move with the asymptotes, in the first chart: ``log s``
    crossed with the logits of :data:`_SCAN_ASYMPTOTE_SHARES` for each estimated asymptote
    coordinate (one empty combination when none is estimated). Each start carries the asymptote
    coordinates of the best combination at its ``log s``."""
    n_free = int(first_chart.estimate_lower) + int(first_chart.estimate_upper)
    logits = [float(_logit(share)) for share in _SCAN_ASYMPTOTE_SHARES]
    coordinates = [np.array(c, dtype=np.float64) for c in itertools.product(logits, repeat=n_free)]
    kept: list[NDArray[np.float64]] = []
    z_values: list[float] = []
    lower_values: list[float] = []
    upper_values: list[float] = []
    for coords in coordinates:
        lower, upper = first_chart.asymptotes(coords)
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            continue
        target = _target_for(first_chart.definition, upper=upper, lower=lower)
        z_star = _z_star(first_chart.link, upper=upper, lower=lower, target=target)
        if z_star is None or not math.isfinite(z_star):
            continue
        kept.append(coords)
        z_values.append(z_star)
        lower_values.append(lower)
        upper_values.append(upper)
    if not kept:
        return []
    picks, best_combo = _scan_picks(
        z_chart, first_chart.theta, log_s, z_values, lower_values, upper_values
    )
    return [(first_chart, np.concatenate(([log_s[i]], kept[best_combo[i]]))) for i in picks]


def _bfgs_fit(model: GenericLikelihoodModel, start: NDArray[np.float64]) -> Any:
    """The BFGS run of the inner fit. Raises ``LinAlgError``/``ValueError`` as statsmodels does."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(start_params=start, method="bfgs", disp=False, maxiter=500, gtol=1e-8)


def _newton_polished_llf(
    model: GenericLikelihoodModel, params: NDArray[np.float64]
) -> float | None:
    """The Newton polish of a BFGS result, or ``None`` when it raises. It is an improvement, not a
    requirement: its own failure modes -- a singular Hessian at an outlying theta, or a downhill
    step -- must not discard the usable BFGS profile value underneath it. ``ell_p`` too low
    inflates ``2 * (llf_hat - ell_p)``, which moves ``g``'s crossing closer to ``theta_hat`` and
    ships a silently *too narrow* interval. Taking the maximum is the conservative direction, and
    mirrors the guard ``models.py`` already applies to the outer fit."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            polished = model.fit(
                start_params=params,
                method="newton",
                disp=False,
                maxiter=100,
                tol=1e-8,
            )
    except (np.linalg.LinAlgError, ValueError):
        return None
    return float(polished.llf)


def _failed_alternatives_warning(count: int) -> str:
    return (
        f"intervals.profile_interval: {count} alternative start(s) of the profile inner fit "
        "raised in the optimiser and were skipped; the profile value at that theta is the best "
        "of the first run and the alternatives that completed"
    )


def _best_alternative_fit(
    fit: Fit,
    first_chart: _MuSubstitutedLogLikelihood,
    first_start: NDArray[np.float64],
    on_failed_alternative: Callable[[], None] | None = None,
) -> tuple[float, GenericLikelihoodModel, NDArray[np.float64]] | None:
    """BFGS from each of :func:`_alternative_starts`, in the ``z*`` chart, the box-face chart or
    the first chart as each start says. Returns the best finite ``(llf, model, params)``, or
    ``None`` when no start completed. ``on_failed_alternative`` is called once for each start
    whose run raises or finishes with a non-finite log-likelihood."""
    best: tuple[float, GenericLikelihoodModel, NDArray[np.float64]] | None = None
    for chart, alt_start in _alternative_starts(fit, first_chart, first_start):
        model = GenericLikelihoodModel(
            first_chart.successes,
            np.ones((first_chart.severity.size, alt_start.size)),
            loglike=chart,
        )
        try:
            result = _bfgs_fit(model, alt_start)
        except (np.linalg.LinAlgError, ValueError):
            if on_failed_alternative is not None:
                on_failed_alternative()
            continue
        alt_llf = float(result.llf)
        if not math.isfinite(alt_llf):
            if on_failed_alternative is not None:
                on_failed_alternative()
            continue
        if best is None or alt_llf > best[0]:
            best = (alt_llf, model, np.asarray(result.params, dtype=np.float64))
    return best


def _estimated_profile_llf(
    theta: float,
    *,
    fit: Fit,
    definition: Definition,
    on_failed_alternative: Callable[[], None] | None = None,
    llf_hat: float | None = None,
    chi2_target: float | None = None,
) -> float | None:
    """``ell_p(theta)`` for a profile in which an asymptote is estimated (this module's
    docstring): the first run from the fit's own parameters, then -- unless both asymptotes are
    fixed at ``(1, 0)``, where the problem is concave along the constraint -- alternative starts
    chosen by scans of log-likelihood values, run in the ``z*`` chart where it applies (and, at a
    finite end of its feasible interval, on that box face) and otherwise in the first chart. The
    best alternative is used only if it beats the first run by more than the Newton tolerance. An
    alternative that raises is skipped and reported through ``on_failed_alternative``, once per
    start; the first run's own failure returns ``None``.

    When ``llf_hat`` and ``chi2_target`` are both given and the first run's value already has
    ``2 * (llf_hat - ell) - chi2_target < 0``, the alternatives are not run and the first run's
    value is returned: the constrained maximum is at least that value, so the sign of the deviance
    minus the target is already settled. The returned value is then a lower bound on the
    constrained maximum and is the constrained maximum only where it matters for that sign. Leave
    both unset where the value itself is used, as at the estimate.
    """
    assert fit.params is not None
    severity, successes, trials = _cells_to_arrays(fit.cells)
    lower_param = fit.params["lower"]
    upper_param = fit.params["upper"]
    loglike_fn = _MuSubstitutedLogLikelihood(
        theta=theta,
        definition=definition,
        severity=severity,
        successes=successes,
        trials=trials,
        scale=fit.axis.scale,
        link=fit.link,
        estimate_lower=not lower_param.fixed,
        estimate_upper=not upper_param.fixed,
        lower_value=lower_param.value if lower_param.fixed else math.nan,
        upper_value=upper_param.value if upper_param.fixed else math.nan,
    )
    start = _optimizer_start_from_fit(fit)
    exog = np.ones((severity.size, start.size))
    model = GenericLikelihoodModel(successes, exog, loglike=loglike_fn)
    try:
        bfgs_result = _bfgs_fit(model, start)
    except (np.linalg.LinAlgError, ValueError):
        return None
    llf = float(bfgs_result.llf)
    polished_llf = _newton_polished_llf(model, np.asarray(bfgs_result.params, dtype=np.float64))
    if polished_llf is not None and math.isfinite(polished_llf):
        llf = max(llf, polished_llf)

    fixed_unit_range = (
        lower_param.fixed
        and upper_param.fixed
        and lower_param.value == 0.0
        and upper_param.value == 1.0
    )
    pilot_best = llf if math.isfinite(llf) else -math.inf
    sign_settled = (
        llf_hat is not None
        and chi2_target is not None
        and 2.0 * (llf_hat - pilot_best) - chi2_target < 0.0
    )
    if not fixed_unit_range and not sign_settled:
        best_alternative = _best_alternative_fit(fit, loglike_fn, start, on_failed_alternative)
        if best_alternative is not None and (
            best_alternative[0] > pilot_best + _NEWTON_DECREMENT_TOLERANCE
        ):
            llf, alt_model, alt_params = best_alternative
            alt_polished = _newton_polished_llf(alt_model, alt_params)
            if alt_polished is not None and math.isfinite(alt_polished):
                llf = max(llf, alt_polished)
    if not math.isfinite(llf) or llf <= _UNREACHABLE_LOGLIKELIHOOD / 2.0:
        return None
    return llf


def _profile_interval_estimated(
    fit: Fit, *, definition: Definition, level: float, theta_hat: float
) -> IntervalResult:
    llf_hat = _llf_hat(fit)
    chi2_target = float(chi2.ppf(level, 1))

    failed_alternatives = 0

    def count_failed_alternative() -> None:
        nonlocal failed_alternatives
        failed_alternatives += 1

    def g(theta: float) -> float | None:
        """The deviance minus the target. Away from the estimate its sign is exact and, where it is
        negative, its value is an upper bound; at the estimate and where it is not negative it is
        exact."""
        sign_only = theta != theta_hat
        llf_p = _estimated_profile_llf(
            theta,
            fit=fit,
            definition=definition,
            on_failed_alternative=count_failed_alternative,
            llf_hat=llf_hat if sign_only else None,
            chi2_target=chi2_target if sign_only else None,
        )
        if llf_p is None:
            return None
        return 2.0 * (llf_hat - llf_p) - chi2_target

    g0 = g(theta_hat)
    if g0 is None or not g0 < 0.0:
        raise ValueError(
            "intervals.profile_interval: the estimated-asymptote profile deviance at theta_hat "
            f"did not come out below the chi-square target (g(theta_hat)={g0!r}); the inner "
            "optimiser may have beaten the reported MLE"
        )
    if (g0 + chi2_target) / 2.0 < -_NEWTON_DECREMENT_TOLERANCE:
        raise ValueError(
            "intervals.profile_interval: the constrained fit at theta_hat reached a log-likelihood "
            f"{-(g0 + chi2_target) / 2.0!r} above the reported maximum, so the fit is not at its "
            "MLE; refit before computing an interval"
        )

    lower_limit, upper_limit = _search_limits(fit)
    result = _solve_profile_bracket(
        g,
        theta_hat=theta_hat,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        scale=fit.axis.scale,
        subdivisions=_ESTIMATED_BRACKET_SUBDIVISIONS,
    )
    if not failed_alternatives:
        return result
    return replace(
        result,
        warnings=(*result.warnings, _failed_alternatives_warning(failed_alternatives)),
    )


def profile_interval(
    fit: Fit, *, definition: Definition, target: float, level: float, theta_hat: float
) -> IntervalResult:
    """The profile-likelihood interval for a solved threshold (the primary method).

    Parameters
    ----------
    fit
        A ``status is Status.OK`` fit.
    definition
        The :class:`~robustmargin.Definition` the threshold solves for -- only used by the
        estimated-asymptote path, where the candidate asymptotes at each inner evaluation move
        both ``z*`` and (for ``baseline_fraction``) the target itself.
    target
        The already-resolved target performance at the fit's own point estimates.
    level
        The confidence level, strictly between 0 and 1.
    theta_hat
        The point estimate on the transformed axis (:func:`solve_value`'s first return value).

    Returns
    -------
    IntervalResult
    """
    _check_finite_number(level, label="profile_interval level")
    assert fit.params is not None
    # Dispatch on the asymptote *values*, not on their `fixed` flags. The offset trick models
    # the raw failure rate, `P(failure) = F(z)`, which is an identity only at `u = 1, l = 0`.
    # For any other fixed pair the modelled quantity is `(u - P) / (u - l)`, so the inner GLM
    # maximises the wrong objective: `ell_p` is then evaluated with the right likelihood at the
    # wrong parameters, understating it and shipping a too-narrow interval.
    # This mirrors `models.py`'s own GLM special case, which already tests the values.
    if (
        fit.params["upper"].fixed
        and fit.params["lower"].fixed
        and fit.params["upper"].value == 1.0
        and fit.params["lower"].value == 0.0
    ):
        return _profile_interval_fixed(
            fit, definition=definition, target=target, level=level, theta_hat=theta_hat
        )
    return _profile_interval_estimated(fit, definition=definition, level=level, theta_hat=theta_hat)


# ------------------------------------------------------------------------------------------
# Delta method (cross-check).
# ------------------------------------------------------------------------------------------


def delta_interval(
    fit: Fit, *, definition: Definition, target: float, level: float, theta_hat: float, value: float
) -> IntervalResult:
    """The delta-method interval for a solved threshold (the cross-check).

    Computed on the log scale and exponentiated -- even on a linear axis, where
    ``sigma_log = sigma_theta / value`` -- which is what keeps the interval positive and
    equivariant under ``x -> c * x`` regardless of axis (this module's docstring).

    Parameters
    ----------
    fit
        A ``status is Status.OK`` fit.
    definition
        The :class:`~robustmargin.Definition` the threshold solves for (needed for ``d(q)/d(u)``
        and ``d(q)/d(l)`` when an asymptote is estimated).
    target
        The already-resolved target performance.
    level
        The confidence level, strictly between 0 and 1.
    theta_hat, value
        :func:`solve_value`'s two return values.

    Returns
    -------
    IntervalResult
        ``censoring`` is always :data:`~robustmargin.Censoring.NONE`: the delta method has no
        open-sided outcome.
    """
    var_theta = _theta_variance(fit, definition=definition, target=target)
    sigma_theta = math.sqrt(max(var_theta, 0.0))

    sigma_log = sigma_theta if fit.axis.scale == "log" else sigma_theta / value
    z_score = float(norm.ppf(0.5 + level / 2.0))
    # `math.exp` raises OverflowError rather than returning inf, and a flat curve on a coarse
    # grid makes `sigma_log` large enough to reach it. An interval whose upper end overflows is
    # not a bounded interval, so it takes the same route as an unbounded profile: the estimate
    # with no bounds, and the reason in `warnings`.
    half_width = z_score * sigma_log
    if not math.isfinite(half_width) or half_width > _MAX_LOG_HALF_WIDTH:
        return IntervalResult(
            lo=None,
            hi=None,
            censoring=Censoring.NONE,
            warnings=(
                "intervals.delta_interval: the delta standard error on the log scale "
                f"({sigma_log!r}) is too large for a finite interval; the design does not "
                "constrain the threshold, so no bounds are reported",
            ),
        )
    lo = value * math.exp(-half_width)
    hi = value * math.exp(half_width)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return IntervalResult(
            lo=None,
            hi=None,
            censoring=Censoring.NONE,
            warnings=(
                "intervals.delta_interval: the interval endpoints are not finite; the design "
                "does not constrain the threshold, so no bounds are reported",
            ),
        )
    return IntervalResult(lo=lo, hi=hi, censoring=Censoring.NONE, warnings=())
