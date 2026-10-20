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
(a likelihood for a statsmodels optimiser, never a hand-rolled one). A candidate ``(u, l)`` that
makes ``P*`` unreachable, or an inner fit that fails outright, opens that side of the search rather
than returning a number (a failed fit never carries a number).

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

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import expit
from scipy.special import logit as _logit
from scipy.stats import chi2, norm
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from robustmargin.models import (
    _STATSMODELS_LINKS,
    Fit,
    _cells_to_arrays,
    _full_binomial_log_likelihood,
    _transform,
)
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
_UNREACHABLE_LOGLIKELIHOOD = -1.0e12


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
    """
    if definition.kind == "absolute":
        return definition.value
    if definition.kind == "baseline_fraction":
        return definition.value * upper
    return upper - definition.value * (upper - lower)  # relative


def _dq_du_dl(definition: Definition, *, upper: float, lower: float) -> tuple[float, float]:
    """``d(q)/d(upper)`` and ``d(q)/d(lower)`` for ``q = (upper - P*) / (upper - lower)``, per
    definition kind (this module's docstring)."""
    spread = upper - lower
    if definition.kind == "absolute":
        a = definition.value
        return (a - lower) / spread**2, (upper - a) / spread**2
    if definition.kind == "baseline_fraction":
        p = definition.value
        return -(1.0 - p) * lower / spread**2, (1.0 - p) * upper / spread**2
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


def _bracket_root(
    g: Callable[[float], float | None], theta_hat: float, limit: float, direction: float
) -> tuple[str, float] | tuple[str, float, float]:
    """Step outward from ``theta_hat`` toward ``limit`` (``direction`` is ``+1.0`` or ``-1.0``)
    looking for ``g``'s sign change (``g(theta_hat) < 0``, by the caller's own precondition).

    Returns ``("bracket", a, b)`` with ``g(a) < 0 <= g(b)``, ``("open_failed", last_theta)`` if
    the inner fit failed at some ``theta`` before a sign change was found, or
    ``("open_limit", limit)`` if the search limit was reached with no sign change.
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
            return ("open_failed", prev_theta)
        if prev_g < 0.0 <= g_theta:
            return ("bracket", prev_theta, theta)
        if reached_limit:
            return ("open_limit", theta)
        prev_theta, prev_g = theta, g_theta
    return ("open_limit", theta)


def _solve_profile_bracket(
    g: Callable[[float], float | None],
    *,
    theta_hat: float,
    lower_limit: float,
    upper_limit: float,
    scale: str,
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
        root = brentq(g, a, b, xtol=1.0e-10, rtol=1.0e-12, maxiter=200)
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
) -> float | None:
    """``ell_p(theta)`` for the fixed-asymptote path: fit the no-intercept offset GLM at this
    ``theta`` (an *optimiser*, per this module's docstring), then evaluate the full likelihood
    at the resulting ``(mu, s)``. Returns ``None`` (an inner-fit failure) when the GLM does not
    converge or its fitted slope is non-positive -- a wrong-sign slope means the profile value
    is the flat-curve boundary, so that side never closes there.
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
        return None
    beta = float(result.params[0])
    if not beta > 0.0:
        return None

    s = 1.0 / beta
    mu = theta - s * z_star
    return _full_binomial_log_likelihood(
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


def _profile_interval_fixed(
    fit: Fit, *, target: float, level: float, theta_hat: float
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

    def g(theta: float) -> float | None:
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

    lower_limit, upper_limit = _search_limits(fit)
    return _solve_profile_bracket(
        g,
        theta_hat=theta_hat,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        scale=fit.axis.scale,
    )


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

    def __call__(self, params: NDArray[np.float64]) -> float:
        s = float(np.exp(params[0]))
        idx = 1
        if self.estimate_lower:
            a = float(params[idx])
            idx += 1
            lower = float(expit(a)) if self.estimate_upper else float(self.upper_value * expit(a))
        else:
            lower = self.lower_value
        if self.estimate_upper:
            b = float(params[idx])
            upper = float(lower + (1.0 - lower) * expit(b))
        else:
            upper = self.upper_value

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


def _estimated_profile_llf(theta: float, *, fit: Fit, definition: Definition) -> float | None:
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bfgs_result = model.fit(
                start_params=start, method="bfgs", disp=False, maxiter=500, gtol=1e-8
            )
    except (np.linalg.LinAlgError, ValueError):
        return None
    llf = float(bfgs_result.llf)

    # The Newton polish is an improvement, not a requirement. Its own failure modes -- a
    # singular Hessian at an outlying theta, or a downhill step -- must not discard the usable
    # BFGS profile value underneath it: `ell_p` too low inflates `2 * (llf_hat - ell_p)`, which
    # moves `g`'s crossing closer to `theta_hat` and ships a silently *too narrow* interval.
    # Taking the maximum is the conservative direction, and mirrors the guard `models.py`
    # already applies to the outer fit.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            polished = model.fit(
                start_params=np.asarray(bfgs_result.params, dtype=np.float64),
                method="newton",
                disp=False,
                maxiter=100,
                tol=1e-8,
            )
    except (np.linalg.LinAlgError, ValueError):
        polished = None
    if polished is not None:
        polished_llf = float(polished.llf)
        if math.isfinite(polished_llf):
            llf = max(llf, polished_llf)
    if not math.isfinite(llf) or llf <= _UNREACHABLE_LOGLIKELIHOOD / 2.0:
        return None
    return llf


def _profile_interval_estimated(
    fit: Fit, *, definition: Definition, level: float, theta_hat: float
) -> IntervalResult:
    llf_hat = _llf_hat(fit)
    chi2_target = float(chi2.ppf(level, 1))

    def g(theta: float) -> float | None:
        llf_p = _estimated_profile_llf(theta, fit=fit, definition=definition)
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

    lower_limit, upper_limit = _search_limits(fit)
    return _solve_profile_bracket(
        g,
        theta_hat=theta_hat,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        scale=fit.axis.scale,
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
        return _profile_interval_fixed(fit, target=target, level=level, theta_hat=theta_hat)
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
