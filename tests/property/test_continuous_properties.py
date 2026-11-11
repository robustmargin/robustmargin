"""Properties of the Gaussian dose-response family (continuous outcomes), checked with hypothesis.

Each property is stated on the public API only, except the profile oracle, which is this file's
own brute-force minimiser of the residual sum of squares and calls nothing from robustmargin's
internals.

* **Aggregation does not matter.** ``from_observations`` and ``from_summaries`` describe the
  same data, so the fits and thresholds agree. They are compared to relative ``1e-9``, not bit
  for bit: a summary's within-cell sum of squares is rebuilt from the sample SD, which is a
  rounding apart from the sum the rows pooled to.
* **Negation is exact.** An increasing series is the negation of a decreasing one, and negation
  is exact in floating point, so the two fits and their thresholds are compared with ``==``. The
  mirror of ``(upper, lower)`` is ``(-lower, -upper)``, and ``absolute(a)`` mirrors to
  ``absolute(-a)``.
* **Response-scale equivariance.** ``y -> a*y + b`` (``a > 0``) leaves a ``relative`` threshold
  where it was and moves an ``absolute`` target with the data. ``baseline_fraction`` is
  equivariant under pure rescaling only, since it is a ratio to the zero-severity level.
* **Monotone in the criterion.** ``relative(p)`` moves out with ``p``, and ``absolute(a)`` moves
  with ``a`` in the direction the outcome runs.
* **No number on a failure, and no exact claim.** A result that is not ``OK`` carries no value,
  bound or grid, and a Gaussian result is never ``RIGHT``, ``LEFT`` or ``FAILS_AT_BASELINE``.
* **Intervals nest.** ``lo <= value <= hi`` where all three exist, and a higher level never
  narrows a side or closes one that the lower level left open.
* **The profile bound is not too narrow.** At each closed profile bound ``theta_b`` the F
  statistic ``(RSS_theta - RSS) / (RSS / df)`` is at least the ``F(1, df)`` cut-off, where
  ``RSS_theta`` is minimised by this file's own dense grid plus ``scipy.optimize`` over every
  parameter free under the constraint ``f(theta_b) = target``.

Seeds are drawn by hypothesis and the data generated from a seeded ``numpy`` generator, so a
failing example replays.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import f as f_dist
from scipy.stats import norm

from robustmargin import (
    Axis,
    Censoring,
    ContinuousObservations,
    Definition,
    Fit,
    Status,
    Threshold,
    fit_dose_response,
    threshold,
)

_AXIS = Axis(name="severity", unit="unit", scale="log")
_LEVELS = [0.0, *np.logspace(-1.0, 1.0, 7).tolist()]
_LINKS = ("logit", "probit", "cloglog")
_SETTINGS = settings(
    max_examples=8,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------


def _cdf(z: np.ndarray, link: str) -> np.ndarray:
    if link == "logit":
        return np.asarray(1.0 / (1.0 + np.exp(-z)))
    if link == "probit":
        return np.asarray(norm.cdf(z))
    return np.asarray(1.0 - np.exp(-np.exp(z)))


def _quantile(q: float, link: str) -> float:
    if link == "logit":
        return math.log(q / (1.0 - q))
    if link == "probit":
        return float(norm.ppf(q))
    return math.log(-math.log(1.0 - q))


def _curve(x: np.ndarray, mu: float, s: float, u: float, low: float, link: str) -> np.ndarray:
    """Decreasing LL.4-family curve on a log axis; the zero-severity control sits at ``u``."""
    with np.errstate(divide="ignore"):
        z = (np.log(x) - mu) / s
    f = np.where(x == 0.0, 0.0, _cdf(z, link))
    return np.asarray(u - (u - low) * f)


@st.composite
def _designs(draw: st.DrawFn) -> dict[str, Any]:
    """A well-identified decreasing series: the signal is at least thirty times the noise."""
    u = draw(st.floats(6.0, 20.0))
    gap = draw(st.floats(4.0, 12.0))
    design: dict[str, Any] = {
        "seed": draw(st.integers(0, 2**31 - 1)),
        "mu": draw(st.floats(-0.3, 0.3)),
        "s": draw(st.floats(0.3, 0.8)),
        "u": u,
        "l": u - gap,
        "sigma": gap * draw(st.floats(0.005, 0.03)),
        "reps": draw(st.integers(2, 4)),
        "link": draw(st.sampled_from(_LINKS)),
    }
    # A transition sharper than the grid can resolve is a step, and a step is a legitimate
    # failure of the fit. Drop such a draw on the TRUE curve, never on what the fit reports: at
    # least two non-control levels must sit where the true curve is informative, i.e. the
    # distance to the nearer asymptote, gap * min(F, 1 - F), is at least 3 sigma / sqrt(reps).
    x = np.asarray([v for v in _LEVELS if v > 0.0])
    f = _cdf((np.log(x) - design["mu"]) / design["s"], design["link"])
    informative = gap * np.minimum(f, 1.0 - f) >= 3.0 * design["sigma"] / math.sqrt(design["reps"])
    assume(int(informative.sum()) >= 2)
    return design


def _rows(design: dict[str, Any]) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(design["seed"])
    severity = [x for x in _LEVELS for _ in range(design["reps"])]
    mean = _curve(
        np.asarray(severity), design["mu"], design["s"], design["u"], design["l"], design["link"]
    )
    return severity, (mean + rng.normal(0.0, design["sigma"], size=mean.size)).tolist()


def _obs(
    severity: list[float], value: list[float], direction: str = "decreasing"
) -> ContinuousObservations:
    return ContinuousObservations.from_observations(
        _AXIS, severity=severity, value=value, outcome="response", direction=direction
    )


def _fit(obs: ContinuousObservations, link: str, **asymptotes: Any) -> Fit:
    kw = {"upper": "estimate", "lower": "estimate", **asymptotes}
    return fit_dose_response(obs, model="gaussian", link=link, **kw)


def _threshold(fit: Fit, definition: Definition, method: str, level: float = 0.95) -> Threshold:
    return threshold(fit, definition=definition, interval_method=method, level=level)


def _numbers(t: Threshold) -> tuple[Any, ...]:
    return (t.status, t.censoring, t.value, t.lo, t.hi)


def _close(a: float | None, b: float | None, rtol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=rtol, abs_tol=1e-12)


# ---------------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------------


@_SETTINGS
@given(design=_designs())
def test_values_and_their_summaries_give_the_same_fit_and_thresholds(
    design: dict[str, Any],
) -> None:
    severity, value = _rows(design)
    rows = _obs(severity, value)
    cells = rows.cells()
    summaries = ContinuousObservations.from_summaries(
        _AXIS,
        severity=[c.severity for c in cells],
        mean=[c.mean for c in cells],
        sd=[c.sd for c in cells],
        n=[c.n for c in cells],
        outcome="response",
        direction="decreasing",
    )

    fit_a, fit_b = _fit(rows, design["link"]), _fit(summaries, design["link"])

    assert fit_a.status is Status.OK and fit_b.status is Status.OK
    assert fit_a.params is not None and fit_b.params is not None
    for name, parameter in fit_a.params.items():
        assert _close(parameter.value, fit_b.params[name].value, 1e-9), name
    assert fit_a.residual is not None and fit_b.residual is not None
    assert fit_a.residual.df == fit_b.residual.df
    assert _close(fit_a.residual.rss, fit_b.residual.rss, 1e-9)
    for method in ("delta", "profile"):
        ta = _threshold(fit_a, Definition.relative(0.5), method)
        tb = _threshold(fit_b, Definition.relative(0.5), method)
        assert ta.status is tb.status and ta.censoring is tb.censoring
        assert _close(ta.value, tb.value, 1e-9)
        assert _close(ta.lo, tb.lo, 1e-7) and _close(ta.hi, tb.hi, 1e-7)


@_SETTINGS
@given(design=_designs(), data=st.data())
def test_row_order_does_not_change_the_fit(design: dict[str, Any], data: st.DataObject) -> None:
    severity, value = _rows(design)
    order = data.draw(st.permutations(range(len(severity))))

    fit_a = _fit(_obs(severity, value), design["link"])
    fit_b = _fit(_obs([severity[i] for i in order], [value[i] for i in order]), design["link"])

    assert fit_a.status is Status.OK and fit_b.status is Status.OK
    assert fit_a.params is not None and fit_b.params is not None
    for name, parameter in fit_a.params.items():
        assert _close(parameter.value, fit_b.params[name].value, 1e-8), name


# ---------------------------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------------------------


@_SETTINGS
@given(design=_designs(), fix_upper=st.booleans())
def test_an_increasing_series_is_bit_identical_to_its_negated_decreasing_series(
    design: dict[str, Any], fix_upper: bool
) -> None:
    severity, value = _rows(design)
    link = design["link"]
    dec_kw: dict[str, Any] = {"upper": design["u"]} if fix_upper else {}
    inc_kw: dict[str, Any] = {"lower": -design["u"]} if fix_upper else {}

    dec = _fit(_obs(severity, value), link, **dec_kw)
    inc = _fit(_obs(severity, [-v for v in value], "increasing"), link, **inc_kw)

    assert dec.status is Status.OK and inc.status is Status.OK
    assert dec.params is not None and inc.params is not None
    assert inc.params["mu"].value == dec.params["mu"].value
    assert inc.params["s"].value == dec.params["s"].value
    assert inc.params["lower"].value == -dec.params["upper"].value
    assert inc.params["upper"].value == -dec.params["lower"].value
    assert inc.params["lower"].fixed is dec.params["upper"].fixed
    assert dec.residual is not None and inc.residual is not None
    assert inc.residual == dec.residual

    mid = design["l"] + 0.5 * (design["u"] - design["l"])
    pairs = [
        (Definition.relative(0.3), Definition.relative(0.3)),
        (Definition.absolute(-mid), Definition.absolute(mid)),
    ]
    for method in ("delta", "profile"):
        for inc_def, dec_def in pairs:
            ti, td = _threshold(inc, inc_def, method), _threshold(dec, dec_def, method)
            assert _numbers(ti) == _numbers(td), (method, inc_def)


# ---------------------------------------------------------------------------------------------
# Response-scale equivariance
# ---------------------------------------------------------------------------------------------


@_SETTINGS
@given(design=_designs(), a=st.floats(0.1, 10.0), b=st.floats(-20.0, 20.0))
def test_an_affine_change_of_response_units_leaves_a_relative_threshold_where_it_was(
    design: dict[str, Any], a: float, b: float
) -> None:
    severity, value = _rows(design)
    link = design["link"]
    fit_a = _fit(_obs(severity, value), link)
    fit_b = _fit(_obs(severity, [a * v + b for v in value]), link)

    assert fit_a.status is Status.OK and fit_b.status is Status.OK
    assert fit_a.params is not None and fit_b.params is not None
    assert _close(fit_a.params["mu"].value, fit_b.params["mu"].value, 1e-5)
    assert _close(fit_a.params["s"].value, fit_b.params["s"].value, 1e-5)
    assert _close(a * fit_a.params["upper"].value + b, fit_b.params["upper"].value, 1e-5)
    assert _close(a * fit_a.params["lower"].value + b, fit_b.params["lower"].value, 1e-5)
    assert fit_a.residual is not None and fit_b.residual is not None
    assert _close(a * fit_a.residual.sigma, fit_b.residual.sigma, 1e-5)

    target = design["l"] + 0.5 * (design["u"] - design["l"])
    for method in ("delta", "profile"):
        for def_a, def_b in [
            (Definition.relative(0.4), Definition.relative(0.4)),
            (Definition.absolute(target), Definition.absolute(a * target + b)),
        ]:
            ta, tb = _threshold(fit_a, def_a, method), _threshold(fit_b, def_b, method)
            assert ta.status is tb.status is Status.OK
            assert _close(ta.value, tb.value, 1e-5)
            assert _close(ta.lo, tb.lo, 1e-4) and _close(ta.hi, tb.hi, 1e-4)


@_SETTINGS
@given(
    design=_designs(),
    log_c=st.floats(math.log(1.0e-3), math.log(1.0e3)),
    d=st.floats(-1.0e7, 1.0e7),
)
def test_an_offset_and_rescaling_of_the_response_maps_the_fit_through_c_u_plus_d(
    design: dict[str, Any], log_c: float, d: float
) -> None:
    c = math.exp(log_c)
    severity, value = _rows(design)
    link = design["link"]
    fit_a = _fit(_obs(severity, value), link)
    fit_b = _fit(_obs(severity, [c * v + d for v in value]), link)

    assert fit_a.status is Status.OK
    assert fit_b.status is fit_a.status, fit_b.warnings
    assert fit_a.params is not None and fit_b.params is not None
    assert _close(fit_a.params["mu"].value, fit_b.params["mu"].value, 1e-4)
    assert _close(fit_a.params["s"].value, fit_b.params["s"].value, 1e-4)
    for name in ("upper", "lower"):
        assert _close(c * fit_a.params[name].value + d, fit_b.params[name].value, 1e-5)


@_SETTINGS
@given(design=_designs(), a=st.floats(0.1, 10.0))
def test_a_pure_rescaling_leaves_a_baseline_fraction_threshold_where_it_was(
    design: dict[str, Any], a: float
) -> None:
    severity, value = _rows(design)
    link = design["link"]
    fit_a = _fit(_obs(severity, value), link)
    fit_b = _fit(_obs(severity, [a * v for v in value]), link)

    ta = _threshold(fit_a, Definition.baseline_fraction(0.6), "delta")
    tb = _threshold(fit_b, Definition.baseline_fraction(0.6), "delta")

    assert ta.status is tb.status
    assert _close(ta.value, tb.value, 1e-5)
    assert _close(ta.lo, tb.lo, 1e-5) and _close(ta.hi, tb.hi, 1e-5)


# ---------------------------------------------------------------------------------------------
# Monotone in the criterion
# ---------------------------------------------------------------------------------------------


@_SETTINGS
@given(design=_designs(), increasing=st.booleans())
def test_the_threshold_moves_monotonically_with_the_criterion(
    design: dict[str, Any], increasing: bool
) -> None:
    severity, value = _rows(design)
    sign = -1.0 if increasing else 1.0
    fit = _fit(
        _obs(severity, [sign * v for v in value], "increasing" if increasing else "decreasing"),
        design["link"],
    )
    assert fit.status is Status.OK

    relative = [
        _threshold(fit, Definition.relative(p), "delta").value for p in (0.2, 0.4, 0.6, 0.8)
    ]
    assert all(v is not None for v in relative)
    assert relative == sorted(relative)  # type: ignore[type-var]
    assert len(set(relative)) == 4

    span = design["u"] - design["l"]
    levels = [design["l"] + f * span for f in (0.25, 0.5, 0.75)]
    absolute = [
        _threshold(fit, Definition.absolute(sign * a), "delta").value for a in sorted(levels)
    ]
    assert all(v is not None for v in absolute)
    # Targets are +a (decreasing series) or -a (negated, increasing series), a ascending. A
    # falling series reaches a higher level sooner; a rising series reaches a lower level sooner.
    # In both cases the ascending-a targets are reached at descending severity.
    expected = sorted(absolute, reverse=True)  # type: ignore[type-var]
    assert absolute == expected


# ---------------------------------------------------------------------------------------------
# Failure and censoring
# ---------------------------------------------------------------------------------------------

_ARBITRARY_VALUES = st.lists(st.floats(-10.0, 10.0), min_size=16, max_size=16)


@_SETTINGS
@given(values=_ARBITRARY_VALUES, link=st.sampled_from(_LINKS), increasing=st.booleans())
def test_no_number_accompanies_a_failure_and_no_exact_claim_is_ever_made(
    values: list[float], link: str, increasing: bool
) -> None:
    severity = [x for x in _LEVELS for _ in range(2)]
    fit = _fit(_obs(severity, values, "increasing" if increasing else "decreasing"), link)

    if fit.status is not Status.OK:
        assert fit.params is None and fit.covariance is None and fit.log_likelihood is None
        assert fit.diagnostics is None
    for definition in (Definition.relative(0.5), Definition.absolute(0.0)):
        for method in ("delta", "profile"):
            t = _threshold(fit, definition, method)
            assert t.status not in (Status.FAILS_AT_BASELINE, Status.SEPARATION)
            assert t.censoring not in (Censoring.RIGHT, Censoring.LEFT)
            if fit.status is not Status.OK:
                assert t.status is Status.NOT_CONVERGED
            if t.status is not Status.OK:
                assert (t.value, t.lo, t.hi, t.grid) == (None, None, None, None)
            elif t.censoring is Censoring.NONE:
                assert t.value is not None and t.lo is not None and t.hi is not None
            else:
                assert t.value is not None
            if t.value is not None and t.lo is not None and t.hi is not None:
                assert t.lo <= t.value <= t.hi


@_SETTINGS
@given(design=_designs())
def test_a_higher_level_never_narrows_an_interval_or_closes_an_open_side(
    design: dict[str, Any],
) -> None:
    severity, value = _rows(design)
    fit = _fit(_obs(severity, value), design["link"])
    assert fit.status is Status.OK

    for method in ("delta", "profile"):
        narrow = _threshold(fit, Definition.relative(0.5), method, level=0.90)
        wide = _threshold(fit, Definition.relative(0.5), method, level=0.99)
        assert narrow.status is Status.OK and wide.status is Status.OK
        assert _close(narrow.value, wide.value, 1e-9)
        if narrow.lo is None:
            assert wide.lo is None
        elif wide.lo is not None:
            assert wide.lo <= narrow.lo * (1.0 + 1e-9)
        if narrow.hi is None:
            assert wide.hi is None
        elif wide.hi is not None:
            assert wide.hi >= narrow.hi * (1.0 - 1e-9)


# ---------------------------------------------------------------------------------------------
# The profile against an independent oracle
# ---------------------------------------------------------------------------------------------


def _rss_given_chart(
    lx: np.ndarray,
    y: np.ndarray,
    zero: np.ndarray,
    *,
    theta: float,
    log_s: float,
    q: float,
    link: str,
    absolute: float | None,
) -> float:
    """Residual sum of squares at a point of the constraint surface ``f(theta) = target``.

    ``q = (u - target) / (u - l)`` and ``mu = theta - s * F^-1(q)``. With ``s`` and ``q`` given,
    ``(u, l)`` enters linearly: free for ``relative`` (where ``q`` is the criterion itself) and on
    the line ``u (1 - q) + q l = a`` for ``absolute(a)``. ``u > l`` is kept by its closure: when
    the unconstrained least squares leaves it, the infimum is the constant fit.
    """
    s = math.exp(log_s)
    mu = theta - s * _quantile(q, link)
    with np.errstate(over="ignore", divide="ignore"):
        cdf = np.where(zero, 0.0, _cdf((lx - mu) / s, link))
    if absolute is None:
        design = np.column_stack([1.0 - cdf, cdf])
        (u, low), *_ = np.linalg.lstsq(design, y, rcond=None)
        if not u > low:
            return float(np.sum((y - y.mean()) ** 2))
        return float(np.sum((y - design @ np.array([u, low])) ** 2))
    a = absolute
    base = a * (1.0 - cdf) / (1.0 - q)
    slope = cdf - q * (1.0 - cdf) / (1.0 - q)
    denominator = float(slope @ slope)
    low = float((slope @ (y - base)) / denominator) if denominator > 0.0 else a
    if not low < a:
        return float(np.sum((y - a) ** 2))
    return float(np.sum((y - base - low * slope) ** 2))


def _oracle_min_rss(
    lx: np.ndarray,
    y: np.ndarray,
    zero: np.ndarray,
    *,
    theta: float,
    link: str,
    relative: float | None,
    absolute: float | None,
) -> float:
    """The constrained minimum of the RSS by a dense grid then a local polish."""

    def at(log_s: float, q: float) -> float:
        return _rss_given_chart(
            lx, y, zero, theta=theta, log_s=log_s, q=q, link=link, absolute=absolute
        )

    log_s_grid = np.linspace(math.log(0.01), math.log(30.0), 260)
    if relative is not None:
        values = np.array([at(g, relative) for g in log_s_grid])
        k = int(np.argmin(values))
        lo, hi = log_s_grid[max(k - 1, 0)], log_s_grid[min(k + 1, len(log_s_grid) - 1)]
        polished = minimize_scalar(
            lambda g: at(g, relative), bounds=(lo, hi), method="bounded", options={"xatol": 1e-10}
        )
        return float(min(values[k], polished.fun))

    q_grid = np.concatenate([[1e-4, 1e-3], np.linspace(0.01, 0.99, 99), [0.999, 0.9999]])
    table = np.array([[at(g, q) for q in q_grid] for g in log_s_grid])
    best = float(table.min())
    flat = np.argsort(table, axis=None)[:4]
    for index in flat:
        i, j = np.unravel_index(index, table.shape)
        start = np.array([log_s_grid[i], q_grid[j]])

        def penalised(p: np.ndarray) -> float:
            if not 1e-6 < p[1] < 1.0 - 1e-6:
                return float("inf")
            return at(float(p[0]), float(p[1]))

        result = minimize(penalised, start, method="Nelder-Mead", options={"xatol": 1e-9})
        best = min(best, float(result.fun))
    return best


@settings(
    max_examples=6,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    design=_designs(),
    which=st.sampled_from(["relative", "absolute"]),
    fraction=st.floats(0.3, 0.7),
)
def test_each_closed_profile_bound_is_not_narrower_than_a_brute_force_minimisation_allows(
    design: dict[str, Any], which: str, fraction: float
) -> None:
    severity, value = _rows(design)
    link = design["link"]
    fit = _fit(_obs(severity, value), link)
    assert fit.status is Status.OK and fit.residual is not None

    if which == "relative":
        definition = Definition.relative(fraction)
        relative, absolute = fraction, None
    else:
        target = design["l"] + fraction * (design["u"] - design["l"])
        definition = Definition.absolute(target)
        relative, absolute = None, target
    t = _threshold(fit, definition, "profile")
    assert t.status is Status.OK
    bounds = [b for b in (t.lo, t.hi) if b is not None]
    assert bounds, "a well-identified series must close at least one side"

    x = np.asarray(severity)
    zero = x == 0.0
    lx = np.log(np.where(zero, 1.0, x))
    y = np.asarray(value)
    cutoff = float(f_dist.ppf(0.95, 1, fit.residual.df))
    for bound in bounds:
        rss_theta = _oracle_min_rss(
            lx, y, zero, theta=math.log(bound), link=link, relative=relative, absolute=absolute
        )
        statistic = (rss_theta - fit.residual.rss) / (fit.residual.rss / fit.residual.df)
        assert statistic >= cutoff * 0.98, (statistic, cutoff, bound)


def test_the_oracle_itself_is_zero_at_the_fitted_threshold_and_grows_away_from_it() -> None:
    """A check on this file's oracle, so a wrong oracle cannot pass a wrong bound."""
    design = {
        "seed": 11, "mu": 0.1, "s": 0.5, "u": 10.0, "l": 2.0, "sigma": 0.08, "reps": 3,
        "link": "logit",
    }  # fmt: skip
    severity, value = _rows(design)
    fit = _fit(_obs(severity, value), "logit")
    assert fit.status is Status.OK and fit.residual is not None and fit.params is not None
    x = np.asarray(severity)
    zero = x == 0.0
    lx = np.log(np.where(zero, 1.0, x))
    y = np.asarray(value)
    theta_hat = fit.params["mu"].value + fit.params["s"].value * _quantile(0.5, "logit")

    at_hat = _oracle_min_rss(
        lx, y, zero, theta=theta_hat, link="logit", relative=0.5, absolute=None
    )
    away = _oracle_min_rss(
        lx, y, zero, theta=theta_hat + 0.5, link="logit", relative=0.5, absolute=None
    )

    assert at_hat == pytest.approx(fit.residual.rss, rel=1e-4)
    assert away > at_hat * 1.5
