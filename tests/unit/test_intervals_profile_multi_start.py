"""The profile interval's inner fit must reach the constrained maximum.

The profile log-likelihood is the maximum of the binomial log-likelihood over the scale and any
free asymptotes with the curve pinned through ``(theta, target)``. A single local optimiser run
can settle in a worse basin, miss a transition squeezed between ``theta`` and a tested level (the
scale tending to zero), or, where an estimated asymptote tends to the target, meet a boundary the
optimiser's parametrisation cannot represent. Any of these gives a profile log-likelihood that is
too low, ``g`` too high, and a reported interval too narrow.

Three groups of tests.

* **Chart algebra** (``_zstar_chart_applies``, ``_zstar_chart_asymptotes``,
  ``_ZStarLogLikelihood``). Oracles: the defining identity "the curve at ``theta`` equals the
  target" evaluated with link functions written in this file; an exhaustive table of which
  direction of ``z* -> +/-inf`` is feasible, derived by hand per definition kind and estimated
  set; and agreement with the existing chart's log-likelihood at the same point, with the
  parameter conversion done here.
* **Mirror agreement and regression designs** (small designs on which the increasing and
  decreasing views of one dataset disagreed, plus the falsifying example of the direction-flip
  property).
* **Interval end-points against independent maxima.** Each oracle avoids the package's own
  profile code and optimiser: an explicit feasible curve whose likelihood is an attained lower
  bound on the profile maximum; a dense one-dimensional grid in ``log s`` for fixed asymptotes
  (the constraint then fixes ``z*``, leaving one free coordinate); Nelder-Mead from many starts
  in a chart written here; and a separate fit with an asymptote fixed at the target (the wall
  supremum). The package's outer fit supplies the global log-likelihood maximum only.

Designs that cost minutes are marked ``slow`` (``pytest -m slow --no-cov``).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import binom, chi2, norm

from robustmargin import (
    Axis,
    Censoring,
    Definition,
    Fit,
    Observations,
    Status,
    Threshold,
    fit_dose_response,
    threshold,
)
from robustmargin import intervals as iv
from robustmargin.models import _NEWTON_DECREMENT_TOLERANCE, _full_binomial_log_likelihood

_LEVEL = 0.95
_CHI2 = float(chi2.ppf(_LEVEL, 1))
_LINKS = ("probit", "logit", "cloglog")


# ------------------------------------------------------------------------------------------
# Independent link functions (written here; not the package's).
# ------------------------------------------------------------------------------------------


def _cdf(z: float, link: str) -> float:
    if link == "probit":
        return float(norm.cdf(z))
    if link == "logit":
        return float(expit(z))
    return float(-np.expm1(-np.exp(z)))


def _sf(z: float, link: str) -> float:
    if link == "probit":
        return float(norm.sf(z))
    if link == "logit":
        return float(expit(-z))
    return float(np.exp(-np.exp(z)))


def _inv_cdf(q: float, link: str) -> float:
    if link == "probit":
        return float(norm.ppf(q))
    if link == "logit":
        return float(logit(q))
    return float(np.log(-np.log1p(-q)))


# ------------------------------------------------------------------------------------------
# Chart applicability.
# ------------------------------------------------------------------------------------------


def _applies(definition: Definition, *, est_l: bool, est_u: bool, lo: float, up: float) -> bool:
    return bool(
        iv._zstar_chart_applies(
            definition,
            estimate_lower=est_l,
            estimate_upper=est_u,
            lower_value=lo,
            upper_value=up,
        )
    )


def test_zstar_chart_does_not_apply_to_relative_definitions() -> None:
    """Hand-derived: ``relative(p)`` has a constant ``z*``, so a chart in ``z*`` is moot."""
    for est_l, est_u in ((True, False), (False, True), (True, True)):
        assert not _applies(
            Definition.relative(0.5),
            est_l=est_l,
            est_u=est_u,
            lo=math.nan if est_l else 0.1,
            up=math.nan if est_u else 0.9,
        )


def test_zstar_chart_does_not_apply_when_both_asymptotes_are_fixed() -> None:
    """Hand-derived: with both asymptotes fixed ``z*`` is a constant."""
    for definition in (
        Definition.absolute(0.5),
        Definition.baseline_fraction(0.6),
        Definition.baseline_multiple(2.0),
    ):
        assert not _applies(definition, est_l=False, est_u=False, lo=0.05, up=0.95)


def test_zstar_chart_does_not_apply_to_baseline_fraction_with_lower_fixed_at_zero() -> None:
    """Hand-derived: ``baseline_fraction`` with ``l = 0`` gives ``F(z*) = 1 - f``, constant."""
    assert not _applies(
        Definition.baseline_fraction(0.6), est_l=False, est_u=True, lo=0.0, up=math.nan
    )


@pytest.mark.parametrize(
    ("definition", "est_l", "est_u", "lo", "up"),
    [
        (Definition.absolute(0.5), True, False, math.nan, 0.95),
        (Definition.absolute(0.5), False, True, 0.0, math.nan),
        (Definition.absolute(0.5), True, True, math.nan, math.nan),
        (Definition.baseline_fraction(0.6), True, False, math.nan, 0.9),
        (Definition.baseline_fraction(0.6), False, True, 0.1, math.nan),
        (Definition.baseline_fraction(0.6), True, True, math.nan, math.nan),
        (Definition.baseline_multiple(2.0), True, False, math.nan, 0.9),
        (Definition.baseline_multiple(2.0), False, True, 0.1, math.nan),
        (Definition.baseline_multiple(2.0), True, True, math.nan, math.nan),
    ],
)
def test_zstar_chart_applies_when_an_asymptote_moves_and_z_star_is_not_constant(
    definition: Definition, est_l: bool, est_u: bool, lo: float, up: float
) -> None:
    assert _applies(definition, est_l=est_l, est_u=est_u, lo=lo, up=up)


# ------------------------------------------------------------------------------------------
# Chart algebra.
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Case:
    """One definition kind and estimated set. ``fixed`` is the fixed asymptote (the other is
    estimated), ``extra`` the free coordinate when it exists (with both estimated: ``logit(l)``
    for ``baseline_multiple``, where ``u`` is derived; ``logit(u)`` for ``baseline_fraction``; the
    logit of the share ``t / t_max`` for ``absolute``), ``interior`` feasible ``z*``
    values (hand-checked), ``infeasible`` interior ``z*`` values with no feasible asymptote,
    ``minus``/``plus`` the behaviour as ``z* -> -inf/+inf``: ``"lower"``/``"upper"`` means
    that asymptote tends to the target, ``None`` means infeasible."""

    name: str
    definition: Definition
    est_l: bool
    est_u: bool
    fixed_lower: float
    fixed_upper: float
    extra: tuple[float, ...]
    interior: tuple[float, ...]
    infeasible: tuple[float, ...]
    minus: str | None
    plus: str | None
    llf_point: tuple[float, float]  # (upper, lower) for the cross-chart comparison


_ABS = Definition.absolute(0.5)
_BF = Definition.baseline_fraction(0.6)
_BM = Definition.baseline_multiple(2.0)
_N = math.nan

_CASES = (
    _Case(
        "absolute-lower",
        _ABS,
        True,
        False,
        _N,
        0.95,
        (),
        (0.5, 1.0),
        (-1.0,),
        None,
        "lower",
        (0.95, 0.2),
    ),
    _Case(
        "absolute-upper",
        _ABS,
        False,
        True,
        0.05,
        _N,
        (),
        (-1.0, -0.5),
        (1.0,),
        "upper",
        None,
        (0.7, 0.05),
    ),
    _Case(
        "absolute-both",
        _ABS,
        True,
        True,
        _N,
        _N,
        (0.4,),
        (-1.0, 1.0),
        (),
        "upper",
        "lower",
        (0.8, 0.15),
    ),
    _Case(
        "fraction-lower",
        _BF,
        True,
        False,
        _N,
        0.9,
        (),
        (0.5, 1.0),
        (-1.5,),
        None,
        "lower",
        (0.9, 0.2),
    ),
    _Case(
        "fraction-upper",
        _BF,
        False,
        True,
        0.1,
        _N,
        (),
        (0.5, 1.0),
        (-1.5,),
        None,
        "lower",
        (0.8, 0.1),
    ),
    _Case(
        "fraction-both",
        _BF,
        True,
        True,
        _N,
        _N,
        (float(logit(0.85)),),
        (0.5, 1.0),
        (-1.5,),
        None,
        "lower",
        (0.85, 0.15),
    ),
    _Case(
        "multiple-lower", _BM, True, False, _N, 0.9, (), (-0.5, 1.0), (), None, "lower", (0.9, 0.3)
    ),
    _Case(
        "multiple-upper",
        _BM,
        False,
        True,
        0.1,
        _N,
        (),
        (-0.5, 1.0),
        (),
        "upper",
        "lower",
        (0.8, 0.1),
    ),
    _Case(
        "multiple-both",
        _BM,
        True,
        True,
        _N,
        _N,
        (float(logit(0.2)),),
        (-0.5, 1.0),
        (),
        "upper",
        "lower",
        (0.9, 0.2),
    ),
)


def _asymptotes(case: _Case, z_star: float, link: str) -> tuple[float, float] | None:
    result: tuple[float, float] | None = iv._zstar_chart_asymptotes(
        z_star,
        np.array(case.extra, dtype=np.float64),
        definition=case.definition,
        link=link,
        estimate_lower=case.est_l,
        estimate_upper=case.est_u,
        lower_value=case.fixed_lower,
        upper_value=case.fixed_upper,
    )
    return result


def _case_id(case: _Case) -> str:
    return case.name


@pytest.mark.parametrize("link", _LINKS)
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_zstar_chart_asymptotes_put_the_curve_through_the_target_at_interior_z_star(
    case: _Case, link: str
) -> None:
    """Oracle: ``u - (u - l) * F(z*)`` with ``F`` written in this file equals the definition's
    target, and the fixed asymptote is returned unchanged."""
    for z_star in case.interior:
        result = _asymptotes(case, z_star, link)
        assert result is not None, f"z*={z_star}"
        lower, upper = result
        assert 0.0 <= lower < upper <= 1.0
        target = iv._target_for(case.definition, upper=upper, lower=lower)
        curve = upper - (upper - lower) * _cdf(z_star, link)
        assert curve == pytest.approx(target, abs=1e-12, rel=1e-12)
        if not case.est_l:
            assert lower == case.fixed_lower
        if not case.est_u:
            assert upper == case.fixed_upper


@pytest.mark.parametrize("link", _LINKS)
@pytest.mark.parametrize("case", [c for c in _CASES if c.infeasible], ids=_case_id)
def test_zstar_chart_asymptotes_are_none_where_the_derived_asymptote_is_outside_its_range(
    case: _Case, link: str
) -> None:
    """Hand-derived: e.g. ``absolute`` with only the lower estimated gives
    ``l = (a - S u) / F``, negative whenever ``S > a / u``."""
    for z_star in case.infeasible:
        assert _asymptotes(case, z_star, link) is None, f"z*={z_star}"


@pytest.mark.parametrize("link", _LINKS)
@pytest.mark.parametrize("sign", [-1, 1], ids=["minus-inf", "plus-inf"])
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_zstar_chart_asymptotes_reach_the_wall_where_feasible_and_none_where_not(
    case: _Case, link: str, sign: int
) -> None:
    """Hand-derived table (``minus``/``plus`` on each case). At ``z* = +/-40`` ``F`` or ``S``
    underflows (probit), or is below 1e-17, so the derived asymptote must sit on the target
    (equality allowed: it is the limit), feasible and finite; or be ``None`` where that direction
    has no feasible asymptote."""
    expected = case.plus if sign > 0 else case.minus
    result = _asymptotes(case, 40.0 * sign, link)
    if expected is None:
        assert result is None
        return
    assert result is not None
    lower, upper = result
    assert math.isfinite(lower) and math.isfinite(upper)
    assert 0.0 <= lower <= upper <= 1.0
    target = iv._target_for(case.definition, upper=upper, lower=lower)
    assert lower <= target + 1e-12 <= upper + 2e-12
    wall_gap = (target - lower) if expected == "lower" else (upper - target)
    assert abs(wall_gap) < 1e-9


# --- the log-likelihood matches the existing chart ------------------------------------------

_SEVERITY = np.array([1.0, 3.0, 10.0, 30.0, 100.0])
_SUCCESSES = np.array([95.0, 80.0, 50.0, 20.0, 8.0])
_TRIALS = np.full(5, 100.0)
_THETA = math.log(12.0)
_S = 0.8


def _old_chart_params(case: _Case, upper: float, lower: float) -> NDArray[np.float64]:
    """``[log s, logits]`` of the existing chart for a given ``(upper, lower)``, converted here
    from its documented parametrisation (lower = expit(a), or upper * expit(a) with the upper
    fixed; upper = lower + (1 - lower) * expit(b))."""
    params = [math.log(_S)]
    if case.est_l:
        a = logit(lower) if case.est_u else logit(lower / case.fixed_upper)
        params.append(float(a))
    if case.est_u:
        params.append(float(logit((upper - lower) / (1.0 - lower))))
    return np.array(params, dtype=np.float64)


def _new_chart_params(case: _Case, link: str, upper: float, lower: float) -> NDArray[np.float64]:
    target = iv._target_for(case.definition, upper=upper, lower=lower)
    z_star = _inv_cdf((upper - target) / (upper - lower), link)
    params = [math.log(_S), z_star]
    if case.definition.kind == "absolute" and case.est_l and case.est_u:
        a = case.definition.value
        t = (upper - a) / _cdf(z_star, link)
        t_max = min((1.0 - a) / _cdf(z_star, link), a / _sf(z_star, link))
        params.append(float(logit(t / t_max)))
    elif case.definition.kind == "baseline_multiple" and case.est_l and case.est_u:
        params.append(float(logit(lower)))
    elif case.est_l and case.est_u:
        params.append(float(logit(upper)))
    return np.array(params, dtype=np.float64)


def _llf_args(case: _Case, link: str) -> dict[str, Any]:
    return {
        "theta": _THETA,
        "definition": case.definition,
        "severity": _SEVERITY,
        "successes": _SUCCESSES,
        "trials": _TRIALS,
        "scale": "log",
        "link": link,
        "estimate_lower": case.est_l,
        "estimate_upper": case.est_u,
        "lower_value": case.fixed_lower,
        "upper_value": case.fixed_upper,
    }


@pytest.mark.parametrize("link", _LINKS)
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_zstar_log_likelihood_equals_the_existing_chart_at_a_shared_interior_point(
    case: _Case, link: str
) -> None:
    """Oracle: the existing ``_MuSubstitutedLogLikelihood`` at the same ``(s, u, l)``; the
    parameter conversion between charts is computed in this file from the documented
    parametrisations, not by the package."""
    upper, lower = case.llf_point
    old = iv._MuSubstitutedLogLikelihood(**_llf_args(case, link))
    new = iv._ZStarLogLikelihood(**_llf_args(case, link))

    old_value = old(_old_chart_params(case, upper, lower))
    new_params = _new_chart_params(case, link, upper, lower)
    new_value = new(new_params)

    assert old_value > iv._UNREACHABLE_LOGLIKELIHOOD / 2
    assert new_value == pytest.approx(old_value, abs=1e-10, rel=1e-10)


@pytest.mark.parametrize("link", _LINKS)
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_zstar_chart_asymptotes_invert_the_existing_chart_point(case: _Case, link: str) -> None:
    """Round trip: the ``(l, u)`` recovered from ``(z*, extra)`` computed here equals the
    ``(l, u)`` the point was built from."""
    upper, lower = case.llf_point
    params = _new_chart_params(case, link, upper, lower)
    result = iv._zstar_chart_asymptotes(
        float(params[1]),
        params[2:],
        definition=case.definition,
        link=link,
        estimate_lower=case.est_l,
        estimate_upper=case.est_u,
        lower_value=case.fixed_lower,
        upper_value=case.fixed_upper,
    )
    assert result is not None
    assert result[0] == pytest.approx(lower, rel=1e-9)
    assert result[1] == pytest.approx(upper, rel=1e-9)


@pytest.mark.parametrize("case", [_CASES[0], _CASES[3]], ids=_case_id)
def test_zstar_log_likelihood_is_the_unreachable_constant_where_the_asymptote_is_infeasible(
    case: _Case,
) -> None:
    """``z*`` at which the derived asymptote leaves ``[0, 1]`` returns the shared constant,
    never raises and never a number."""
    new = iv._ZStarLogLikelihood(**_llf_args(case, "probit"))
    assert new(np.array([math.log(_S), -1.5])) == iv._UNREACHABLE_LOGLIKELIHOOD


# ------------------------------------------------------------------------------------------
# Regression designs.
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Design:
    scale: str
    link: Literal["probit", "logit", "cloglog"]
    upper: float | Literal["estimate"]  # decreasing frame
    lower: float | Literal["estimate"]
    severity: tuple[float, ...]
    successes: tuple[int, ...]  # decreasing successes
    trials: tuple[int, ...]


_DESIGNS = {
    "d18": _Design(
        "log",
        "probit",
        "estimate",
        "estimate",
        (0.0, 0.705, 7.714, 20.729, 26.179),
        (258, 245, 161, 107, 113),
        (300,) * 5,
    ),
    "d52": _Design(
        "linear", "logit", 0.95, 0.05, (54.966, 84.166, 95.205, 95.73), (6, 6, 6, 3), (10,) * 4
    ),
    "d67": _Design(
        "linear",
        "logit",
        0.95,
        0.05,
        (0.0, 2.661, 36.889, 41.251, 46.443),
        (8, 10, 8, 10, 8),
        (10,) * 5,
    ),
    "d87": _Design(
        "log",
        "cloglog",
        "estimate",
        "estimate",
        (0.0, 1.232, 52.687, 88.308),
        (99, 96, 16, 7),
        (100,) * 4,
    ),
    "d8": _Design(
        "log",
        "cloglog",
        0.95,
        0.05,
        (0.0, 0.836, 12.339, 25.345, 181.559),
        (97, 99, 98, 91, 27),
        (100,) * 5,
    ),
    "d101": _Design(
        "linear",
        "cloglog",
        0.95,
        0.05,
        (0.0, 5.276, 21.486, 43.019, 64.73, 68.175),
        (16, 19, 17, 9, 3, 2),
        (20,) * 6,
    ),
    # decreasing view (complement counts) of an increasing dataset with events 1, 6, 16, 19
    "d203": _Design(
        "log", "logit", 0.95, 0.05, (0.0, 3.465, 58.998, 195.939), (19, 14, 4, 1), (20,) * 4
    ),
    "d147": _Design(
        "log",
        "probit",
        "estimate",
        "estimate",
        (0.0, 1.201, 2.852, 3.236, 50.878),
        (8, 9, 6, 4, 1),
        (10,) * 5,
    ),
    "d73": _Design(
        "linear",
        "logit",
        "estimate",
        "estimate",
        (0.0, 8.742, 15.612, 30.255, 67.4, 91.752, 94.917),
        (88, 95, 87, 86, 56, 39, 24),
        (100,) * 7,
    ),
    "d153": _Design(
        "linear",
        "logit",
        "estimate",
        "estimate",
        (0.0, 24.142, 31.762, 74.49, 97.173),
        (80, 67, 61, 16, 15),
        (100,) * 5,
    ),
    "d292": _Design(
        "log",
        "cloglog",
        "estimate",
        "estimate",
        (0.0, 2.726, 18.595, 28.059, 45.341, 155.317, 186.255),
        (9, 9, 3, 3, 1, 3, 0),
        (10,) * 7,
    ),
}

# (decreasing definition, increasing definition)
_DEFINITIONS = {
    "absolute": (Definition.absolute(0.8), Definition.absolute(0.2)),
    "relative": (Definition.relative(0.5), Definition.relative(0.5)),
}


def _axis(scale: str) -> Axis:
    return Axis(name="severity", unit="unit", scale=scale)


def _mirror(value: float | Literal["estimate"]) -> float | Literal["estimate"]:
    return value if isinstance(value, str) else 1.0 - value


def _decreasing_fit(design: _Design) -> Fit:
    obs = Observations.from_counts(
        _axis(design.scale),
        severity=design.severity,
        successes=design.successes,
        trials=design.trials,
        outcome="outcome",
        direction="decreasing",
    )
    return fit_dose_response(
        obs,
        model="binomial",
        link=design.link,
        upper=design.upper,
        lower=design.lower,
    )


def _increasing_fit(design: _Design) -> Fit:
    obs = Observations.from_counts(
        _axis(design.scale),
        severity=design.severity,
        successes=[n - k for k, n in zip(design.successes, design.trials, strict=True)],
        trials=design.trials,
        outcome="outcome",
        direction="increasing",
    )
    return fit_dose_response(
        obs,
        model="binomial",
        link=design.link,
        upper=_mirror(design.lower),
        lower=_mirror(design.upper),
    )


@functools.cache
def _thresholds(design_name: str, definition_name: str) -> tuple[Threshold, Threshold]:
    """``(increasing, decreasing)`` profile thresholds of one dataset in both views."""
    design = _DESIGNS[design_name]
    dec_def, inc_def = _DEFINITIONS[definition_name]
    dec = threshold(
        _decreasing_fit(design), definition=dec_def, interval_method="profile", level=_LEVEL
    )
    inc = threshold(
        _increasing_fit(design), definition=inc_def, interval_method="profile", level=_LEVEL
    )
    return inc, dec


_SLOW = pytest.mark.slow
_SWEEP = [
    ("d52", "absolute"),
    ("d52", "relative"),
    ("d67", "absolute"),
    ("d67", "relative"),
    ("d18", "absolute"),
    pytest.param("d18", "relative", marks=_SLOW),
    pytest.param("d87", "absolute", marks=_SLOW),
    pytest.param("d87", "relative", marks=_SLOW),
    pytest.param("d147", "absolute", marks=_SLOW),
    pytest.param("d147", "relative", marks=_SLOW),
    pytest.param("d292", "absolute", marks=_SLOW),
    pytest.param("d292", "relative", marks=_SLOW),
    pytest.param("d73", "absolute", marks=_SLOW),
    pytest.param("d153", "relative", marks=_SLOW),
]


def _same_bound(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a == pytest.approx(b, rel=1e-6)


@pytest.mark.parametrize(("design", "definition"), _SWEEP)
def test_increasing_and_decreasing_profile_intervals_of_one_dataset_agree(
    design: str, definition: str
) -> None:
    """Oracle: the same dataset through the complement view. Binomial data have no preferred
    orientation, so status, censoring and both bounds (relative ``1e-6``) must be equal."""
    inc, dec = _thresholds(design, definition)

    assert inc.status is dec.status
    assert inc.censoring is dec.censoring
    assert _same_bound(inc.lo, dec.lo), (inc.lo, dec.lo)
    assert _same_bound(inc.hi, dec.hi), (inc.hi, dec.hi)


# Upper bounds returned by the code before the inner fit tried several starts, for the sides where
# an independent oracle shows the true interval extends beyond them: the deviance just outside each
# value is below the chi-square target (the d18 and d67 oracles are the tests below; for d147 the
# explicit-curve check is the first d147 test below).
# For these sides any correct ``hi`` is at or beyond the recorded value, or the side is open.
# Other sides are deliberately not recorded: a root-finder returns *some* root inside a bracket, so
# "bounds only move outward" does not hold in general and is not asserted.
_PRE_FIX_UPPER_INSIDE_TRUE: dict[tuple[str, str, str], float] = {
    ("d18", "absolute", "dec"): 7.662808069080362,
    ("d67", "absolute", "dec"): 1880.3674960909045,
    ("d147", "absolute", "dec"): 3.205508681891577,
    ("d147", "absolute", "inc"): 3.119576537349238,
    ("d147", "relative", "dec"): 10.262081396289556,
    ("d147", "relative", "inc"): 10.304835958904729,
}


@pytest.mark.parametrize(
    ("design", "definition", "view"),
    [
        ("d18", "absolute", "dec"),
        ("d67", "absolute", "dec"),
        pytest.param("d147", "absolute", "dec", marks=_SLOW),
        pytest.param("d147", "absolute", "inc", marks=_SLOW),
        pytest.param("d147", "relative", "dec", marks=_SLOW),
        pytest.param("d147", "relative", "inc", marks=_SLOW),
    ],
)
def test_upper_bound_is_at_or_beyond_a_pre_fix_value_the_oracle_shows_was_too_low(
    design: str, definition: str, view: str
) -> None:
    """Oracle: the recorded pre-fix value (``_PRE_FIX_UPPER_INSIDE_TRUE``), for sides where an
    independent maximum shows the deviance outside it is below the chi-square target. The
    reported ``hi`` is then beyond it, or the side is open."""
    inc, dec = _thresholds(design, definition)
    result = inc if view == "inc" else dec
    old_hi = _PRE_FIX_UPPER_INSIDE_TRUE[(design, definition, view)]

    if result.hi is not None:
        assert result.hi >= old_hi * (1 - 1e-6)


# --- independent oracles ---------------------------------------------------------------------


def _theta_axis(x: NDArray[np.float64], scale: str) -> NDArray[np.float64]:
    if scale == "log":
        with np.errstate(divide="ignore"):
            return np.log(x)
    return x


def _vec_cdf(z: NDArray[np.float64], link: str) -> NDArray[np.float64]:
    if link == "probit":
        return np.asarray(norm.cdf(z))
    if link == "logit":
        return np.asarray(expit(z))
    return np.asarray(-np.expm1(-np.exp(z)))


def _llf(design: _Design, *, mu: Any, s: Any, upper: Any, lower: Any) -> Any:
    """Full binomial log-likelihood of the decreasing data, broadcast over parameter arrays
    (cells on axis 0)."""
    x = np.asarray(design.severity, dtype=np.float64)
    k = np.asarray(design.successes, dtype=np.float64)
    n = np.asarray(design.trials, dtype=np.float64)
    shape = (-1,) + (1,) * np.ndim(mu)
    theta_x = _theta_axis(x, design.scale).reshape(shape)
    z = (theta_x - mu) / s
    p = upper - (upper - lower) * _vec_cdf(z, design.link)
    p = np.clip(p, 0.0, 1.0)
    return binom.logpmf(k.reshape(shape), n.reshape(shape), p).sum(axis=0)


def _fixed_asymptote_profile_llf(design: _Design, theta: float, target: float) -> float:
    """Constrained maximum over ``log s`` with both asymptotes fixed: ``mu = theta - s z*``.
    Dense grid, then bounded Brent refinement around the best three grid cells."""
    assert isinstance(design.upper, float) and isinstance(design.lower, float)
    z_star = _inv_cdf((design.upper - target) / (design.upper - design.lower), design.link)
    grid = np.linspace(-8.0, 10.0, 7201)

    def llf_at(log_s: Any) -> Any:
        s = np.exp(log_s)
        return _llf(design, mu=theta - s * z_star, s=s, upper=design.upper, lower=design.lower)

    values = llf_at(grid)
    best = np.argsort(values)[-3:]
    result = float(np.max(values))
    for index in best:
        lo, hi = grid[max(index - 1, 0)], grid[min(index + 1, grid.size - 1)]
        refined = minimize_scalar(
            lambda t: -float(llf_at(np.array(t))),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-10},
        )
        result = max(result, -float(refined.fun))
    return result


def test_fixed_asymptote_oracle_shows_the_pre_fix_upper_bound_is_not_a_crossing() -> None:
    """Oracle: brute-force dense ``log s`` grid with Brent refinement over the 1-D constrained
    problem (asymptotes fixed, so the inner problem has one free coordinate). At the bound the
    unmodified code reported (1880.37) the profile deviance is still below the chi-square
    target, so that bound was not a crossing."""
    design = _DESIGNS["d67"]
    llf_hat = _decreasing_fit(design).log_likelihood
    assert llf_hat is not None
    old_hi = _PRE_FIX_UPPER_INSIDE_TRUE[("d67", "absolute", "dec")]
    assert old_hi is not None

    g = 2.0 * (llf_hat - _fixed_asymptote_profile_llf(design, old_hi, 0.8)) - _CHI2

    assert g < -1.0e-3


def test_fixed_asymptote_profile_upper_bound_moves_past_the_pre_fix_non_crossing() -> None:
    """The reported ``hi`` is outward of 1880.37 or open. Where it is closed, the brute-force
    oracle's ``g`` is ~0 at the bound and negative just inside it."""
    design = _DESIGNS["d67"]
    _, dec = _thresholds("d67", "absolute")
    old_hi = _PRE_FIX_UPPER_INSIDE_TRUE[("d67", "absolute", "dec")]
    assert old_hi is not None

    if dec.hi is None:
        return
    assert dec.hi > old_hi
    assert dec.value is not None
    llf_hat = _decreasing_fit(design).log_likelihood
    assert llf_hat is not None

    def g(theta: float) -> float:
        return 2.0 * (llf_hat - _fixed_asymptote_profile_llf(design, theta, 0.8)) - _CHI2

    assert abs(g(dec.hi)) < 1.0e-3
    assert g(dec.hi - 1.0e-3 * (dec.hi - dec.value)) < 0.0


def _multi_start_profile_llf(design: _Design, theta: float, target: float, fit: Any) -> float:
    """Constrained maximum over ``(log s, l, u)`` by Nelder-Mead from a grid of starts: ``log s``
    at the fit's value plus ``{-2, -1, 0, 1, 2}``, asymptotes at the fit's. Every value returned
    is an attained feasible log-likelihood, so this is a lower bound on the true supremum."""
    s_hat = fit.params["s"].value
    l_hat, u_hat = fit.params["lower"].value, fit.params["upper"].value

    def unpack(p: NDArray[np.float64]) -> tuple[float, float, float]:
        s = math.exp(p[0])
        lower = float(expit(p[1]))
        upper = lower + (1.0 - lower) * float(expit(p[2]))
        return s, lower, upper

    def negative(p: NDArray[np.float64]) -> float:
        s, lower, upper = unpack(p)
        if not lower < target < upper:
            return 1.0e12
        z_star = _inv_cdf((upper - target) / (upper - lower), design.link)
        value = float(_llf(design, mu=theta - s * z_star, s=s, upper=upper, lower=lower))
        return -value if math.isfinite(value) else 1.0e12

    best = -math.inf
    for offset in (-2.0, -1.0, 0.0, 1.0, 2.0):
        start = np.array(
            [
                math.log(s_hat) + offset,
                logit(l_hat),
                logit((u_hat - l_hat) / (1.0 - l_hat)),
            ]
        )
        result = minimize(
            negative,
            start,
            method="Nelder-Mead",
            options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000, "maxfev": 4000},
        )
        best = max(best, -float(result.fun))
    return best


def test_multi_start_oracle_shows_the_pre_fix_estimated_asymptote_bound_is_not_a_crossing() -> None:
    """Oracle: scipy Nelder-Mead over ``(log s, l, u)`` from a grid of starts, written in this
    file. At the bound the unmodified code reported (7.6628), a better constrained maximum
    exists, so the deviance there is below the chi-square target (about -3 against a target of
    3.84)."""
    design = _DESIGNS["d18"]
    fit = _decreasing_fit(design)
    assert fit.log_likelihood is not None
    old_hi = _PRE_FIX_UPPER_INSIDE_TRUE[("d18", "absolute", "dec")]
    assert old_hi is not None

    llf_p = _multi_start_profile_llf(design, math.log(old_hi), 0.8, fit)
    g = 2.0 * (fit.log_likelihood - llf_p) - _CHI2

    assert g < -1.0


def test_estimated_asymptote_upper_bound_extends_to_the_next_design_cell() -> None:
    """Decreasing ``absolute(0.8)``: the true profile stays about 3 below the target up to the
    cell at 7.714 and only then jumps, so ``hi`` lies in ``[7.70, 7.714]``. The unmodified code
    returned 7.6628. Oracle: the multi-start check above, plus the cell location."""
    _, dec = _thresholds("d18", "absolute")

    assert dec.hi is not None
    assert 7.70 <= dec.hi <= 7.714 * (1 + 1e-6)


# --- the direction-flip property's falsifying example ---------------------------------------

_M2_SEVERITY = (0.50006, 1.5, 8.6244, 18.7087, 19.2634, 19.2895)
_M2_EVENTS = (28, 48, 89, 57, 67, 95)
_M2_TRIALS = (170, 159, 200, 97, 93, 110)


def _m2_observations(direction: str) -> Observations:
    successes = (
        _M2_EVENTS
        if direction == "increasing"
        else tuple(n - k for k, n in zip(_M2_EVENTS, _M2_TRIALS, strict=True))
    )
    return Observations.from_counts(
        _axis("linear"),
        severity=_M2_SEVERITY,
        successes=successes,
        trials=_M2_TRIALS,
        outcome="outcome",
        direction=direction,
    )


@functools.cache
def _m2_thresholds() -> tuple[Threshold, Threshold]:
    increasing = threshold(
        fit_dose_response(
            _m2_observations("increasing"),
            model="binomial",
            link="cloglog",
            upper="estimate",
            lower=0.0,
        ),
        definition=Definition.absolute(0.7),
        interval_method="profile",
        level=_LEVEL,
    )
    decreasing = threshold(
        fit_dose_response(
            _m2_observations("decreasing"),
            model="binomial",
            link="cloglog",
            upper=1.0,
            lower="estimate",
        ),
        definition=Definition.absolute(0.3),
        interval_method="profile",
        level=_LEVEL,
    )
    return increasing, decreasing


def test_wall_supremum_is_within_the_chi_square_target_of_the_maximum() -> None:
    """Oracle: ``fit_dose_response`` with the lower asymptote fixed at the target (0.3) gives
    the supremum of the profile as the estimated asymptote tends to the target, a different fit
    path from the profile code. It sits 3.03 below ``2 * llf_hat``, inside the 3.84 target, so
    the profile never crosses and the upper side is open."""
    full = fit_dose_response(
        _m2_observations("decreasing"),
        model="binomial",
        link="cloglog",
        upper=1.0,
        lower="estimate",
    )
    wall = fit_dose_response(
        _m2_observations("decreasing"),
        model="binomial",
        link="cloglog",
        upper=1.0,
        lower=0.3,
    )

    assert wall.status is Status.OK
    assert full.log_likelihood is not None and wall.log_likelihood is not None
    assert wall.log_likelihood == pytest.approx(-29.4075, abs=5e-4)
    assert 2.0 * (full.log_likelihood - wall.log_likelihood) < _CHI2


def test_profile_upper_side_is_open_when_the_estimated_asymptote_can_reach_the_target() -> None:
    """The decreasing view, ``lower`` estimated with ``absolute(0.3)``, reports the upper side
    open (``OPEN_UPPER``, ``hi is None``), not a spurious bound near 29.5."""
    _, dec = _m2_thresholds()

    assert dec.status is Status.OK
    assert dec.censoring is Censoring.OPEN_UPPER
    assert dec.hi is None
    assert dec.lo is not None


def test_increasing_view_mirrors_the_open_upper_side_of_the_decreasing_view() -> None:
    """The increasing fit on the events, ``upper`` estimated with ``absolute(0.7)``, mirrors the
    decreasing result: same censoring, ``hi`` open, ``lo`` equal to ``1e-6`` relative."""
    inc, dec = _m2_thresholds()

    assert inc.censoring is dec.censoring
    assert inc.hi is None
    assert inc.lo is not None and dec.lo is not None
    assert inc.lo == pytest.approx(dec.lo, rel=1e-6)


# --- a transition squeezed against a tested level --------------------------------------------
#
# Design d147 (log axis, probit, both asymptotes estimated, decreasing ``absolute(0.8)``). As the
# threshold approaches the tested level 3.236 from below, the best-fitting transition is squeezed
# between the threshold and that level (``log s -> -inf``), so the profile deviance stays negative
# right up to the level and ``hi`` is the level itself.

_D147_CELL = 3.236
_D147_THRESHOLD_TARGET = 0.8
_D147_CELL_TARGET = 0.4


def _explicit_curve_llf(design: _Design, theta_x: float) -> float:
    """Log-likelihood of one explicit feasible curve through ``(theta_x, 0.8)`` and
    ``(3.236, 0.4)``: upper asymptote ``0.8 + 1e-12``, lower 0.1, probit. Its scale and location
    follow from the two points, so no optimiser is involved. An attained likelihood of a feasible
    curve is a lower bound on the constrained maximum."""
    upper, lower = 0.8 + 1.0e-12, 0.1
    theta = math.log(theta_x)
    z_at_theta = _inv_cdf((upper - _D147_THRESHOLD_TARGET) / (upper - lower), "probit")
    z_at_cell = _inv_cdf((upper - _D147_CELL_TARGET) / (upper - lower), "probit")
    s = (math.log(_D147_CELL) - theta) / (z_at_cell - z_at_theta)
    mu = theta - s * z_at_theta
    return float(_llf(design, mu=mu, s=s, upper=upper, lower=lower))


@pytest.mark.parametrize("theta_x", [3.0, 3.2, 3.2359])
def test_explicit_feasible_curve_keeps_the_deviance_negative_up_to_the_cell(theta_x: float) -> None:
    """Oracle: the explicit curve above, written in this file. Its attained likelihood is a lower
    bound on the profile maximum, so a negative ``g`` from it proves the true ``g`` is negative.
    The deviance is -1.4133 (alongside the 3.84 target) for every threshold in [3.0, 3.2359]."""
    design = _DESIGNS["d147"]
    llf_hat = _decreasing_fit(design).log_likelihood
    assert llf_hat is not None

    g = 2.0 * (llf_hat - _explicit_curve_llf(design, theta_x)) - _CHI2

    assert g == pytest.approx(-1.4133, abs=1e-3)


@pytest.mark.slow
def test_squeezed_transition_upper_bound_reaches_the_tested_level() -> None:
    """``hi`` lies in ``[3.2359, 3.236 * (1 + 1e-6)]`` and the lower side is open. The explicit
    curve above (``test_explicit_feasible_curve_keeps_the_deviance_negative_up_to_the_cell``)
    shows ``g < 0`` up to 3.2359, so no smaller ``hi`` can be a crossing; the oracle's jump to
    ``g > 0`` happens only beyond the level itself (an independent stats review put ``g`` at
    +6.17 at 3.2361). The relative width ``1e-6`` is the root finder's tolerance on a steep
    step."""
    _, dec = _thresholds("d147", "absolute")

    assert dec.status is Status.OK
    assert dec.censoring is Censoring.OPEN_LOWER
    assert dec.lo is None
    assert dec.hi is not None
    assert 3.2359 <= dec.hi <= _D147_CELL * (1 + 1e-6)


@pytest.mark.slow
def test_squeezed_transition_upper_bound_agrees_between_increasing_and_decreasing_views() -> None:
    """The complement view (increasing, ``absolute(0.2)``) reports the same censoring and the
    same ``hi`` to ``1e-6`` relative."""
    inc, dec = _thresholds("d147", "absolute")

    assert inc.censoring is dec.censoring is Censoring.OPEN_LOWER
    assert inc.lo is None and dec.lo is None
    assert inc.hi is not None and dec.hi is not None
    assert inc.hi == pytest.approx(dec.hi, rel=1e-6)
    assert 3.2359 <= inc.hi <= _D147_CELL * (1 + 1e-6)


# --- fixed asymptotes: a one-dimensional oracle in log s ---------------------------------------
#
# With both asymptotes fixed the constraint fixes ``z*``, so the profile log-likelihood is a
# one-dimensional maximisation over ``log s`` (``mu = theta - s z*``). The oracle is a dense grid
# over ``log s`` from ``log(span) - ln(1e9)`` to ``log(span) + ln(1e4)`` (``span`` the range of
# the finite tested levels on the model axis), each of the best four grid cells then refined by a
# bounded scalar search. It reaches both a squeezed transition and a wide interior basin, which
# the narrower grid of ``_fixed_asymptote_profile_llf`` does not.


def _wide_fixed_asymptote_profile_llf(design: _Design, theta: float, target: float) -> float:
    assert isinstance(design.upper, float) and isinstance(design.lower, float)
    z_star = _inv_cdf((design.upper - target) / (design.upper - design.lower), design.link)
    axis = _theta_axis(np.asarray(design.severity, dtype=np.float64), design.scale)
    finite = axis[np.isfinite(axis)]
    span = float(finite.max() - finite.min())
    grid = np.linspace(math.log(span) - math.log(1.0e9), math.log(span) + math.log(1.0e4), 6001)

    def llf_at(log_s: Any) -> Any:
        s = np.exp(log_s)
        with np.errstate(all="ignore"):
            return _llf(design, mu=theta - s * z_star, s=s, upper=design.upper, lower=design.lower)

    values = llf_at(grid)
    result = float(np.max(values))
    for index in np.argsort(values)[-4:]:
        refined = minimize_scalar(
            lambda t: -float(llf_at(np.array(t))),
            bounds=(grid[max(index - 1, 0)], grid[min(index + 1, grid.size - 1)]),
            method="bounded",
            options={"xatol": 1e-10},
        )
        result = max(result, -float(refined.fun))
    return result


def _fixed_design_deviance(design_name: str, target: float, x: float) -> float:
    """``g(x) = 2 (llf_hat - llf_p(x)) - chi2`` from the wide oracle, at a threshold ``x`` on the
    natural axis."""
    design = _DESIGNS[design_name]
    llf_hat = _decreasing_fit(design).log_likelihood
    assert llf_hat is not None
    theta = math.log(x) if design.scale == "log" else x
    return 2.0 * (llf_hat - _wide_fixed_asymptote_profile_llf(design, theta, target)) - _CHI2


def test_fixed_asymptote_upper_bound_on_a_squeezed_transition_is_the_tested_level() -> None:
    """Design d8 (log axis, cloglog, asymptotes (0.95, 0.05), decreasing ``absolute(0.8)``).
    Oracle: the wide log-``s`` grid. The oracle's ``g`` is -2.195 all the way up to the top
    tested level 181.559 and jumps to +127 just beyond it, so the crossing is at that level. The
    window ``1e-5`` relative is where the oracle's sign has changed at both ends."""
    level = 181.559
    assert _fixed_design_deviance("d8", 0.8, level * (1 - 1e-5)) < -2.0
    assert _fixed_design_deviance("d8", 0.8, level * (1 + 1e-5)) > 0.0

    result = threshold(
        _decreasing_fit(_DESIGNS["d8"]),
        definition=Definition.absolute(0.8),
        interval_method="profile",
        level=_LEVEL,
    )

    assert result.status is Status.OK
    assert result.hi is not None
    assert level * (1 - 1e-5) <= result.hi <= level * (1 + 1e-5)


@pytest.mark.parametrize(
    ("design_name", "definition", "target", "view"),
    [
        ("d101", Definition.baseline_fraction(0.8), 0.8 * 0.95, "dec"),
        ("d203", Definition.baseline_multiple(2.0), 1.0 - 2.0 * (1.0 - 0.95), "inc"),
    ],
    ids=["d101-interior-basin", "d203-interior-basin"],
)
def test_fixed_asymptote_upper_bound_sits_where_the_oracle_deviance_crosses_zero(
    design_name: str, definition: Definition, target: float, view: str
) -> None:
    """Designs d101 (linear axis, ``baseline_fraction(0.8)``) and d203 (increasing,
    ``baseline_multiple(2)``, target in the decreasing frame ``1 - 2 (1 - u)``), asymptotes fixed
    at (0.95, 0.05). The oracle is the wide log-``s`` grid, whose ``g`` is negative just outside
    the old bounds (-0.21 at 35.3, -1.26 at 2.0). The oracle's ``g`` has slope about 6 to 26 per
    unit relative change at the crossing, so at ``1e-3`` relative from the reported ``hi`` it is
    ``+/-0.006`` or more, six orders above the oracle's own error; the reported bound must lie
    between a negative and a positive oracle value."""
    design = _DESIGNS[design_name]
    fit = _increasing_fit(design) if view == "inc" else _decreasing_fit(design)

    result = threshold(fit, definition=definition, interval_method="profile", level=_LEVEL)

    assert result.status is Status.OK
    assert result.hi is not None
    assert _fixed_design_deviance(design_name, target, result.hi * (1 - 1e-3)) < 0.0
    assert _fixed_design_deviance(design_name, target, result.hi * (1 + 1e-3)) > 0.0


# --- baseline_multiple with both asymptotes estimated ------------------------------------------
#
# Increasing outcomes on a log axis, severity 0, 1, 2, 4, 8, 16, 32. Under ``baseline_multiple(2)``
# the threshold is where the curve reaches twice the lower asymptote. As the threshold falls, the
# lower asymptote can shrink with the curve's value at the threshold (``l = u F / (k - 1 + F)``
# with ``F`` the link at the threshold), so the constrained maximum tends to the fit with the
# lower asymptote at 0 and the profile never crosses on the lower side.

_C2_SEVERITY = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
_C2_TRIALS = (200, 100, 100, 100, 100, 100, 100)
_C2_EVENTS = {
    "logit": (0, 1, 1, 2, 30, 80, 95),
    "probit": (0, 0, 2, 1, 40, 85, 97),
}


def _c2_observations(link: str) -> Observations:
    return Observations.from_counts(
        _axis("log"),
        severity=_C2_SEVERITY,
        successes=_C2_EVENTS[link],
        trials=_C2_TRIALS,
        outcome="outcome",
        direction="increasing",
    )


@functools.cache
def _c2_fit(link: str) -> Fit:
    return fit_dose_response(
        _c2_observations(link),
        model="binomial",
        link=link,  # type: ignore[arg-type]
        upper="estimate",
        lower="estimate",
    )


def _c2_profile_llf(link: str, theta: float, multiple: float = 2.0) -> float:
    """Oracle: constrained maximum by Nelder-Mead over ``(mu, log s, logit u)`` with the lower
    asymptote derived as ``l = u F / (k - 1 + F)``, ``F = F((theta - mu) / s)``. This is a chart
    written here, not the package's. 60 starts (27 on a grid of ``z*`` and ``log s``, 33 random
    with seed 7); the best attained value is a lower bound on the supremum."""
    fit = _c2_fit(link)
    assert fit.params is not None
    events = np.asarray(_C2_EVENTS[link], dtype=np.float64)
    trials = np.asarray(_C2_TRIALS, dtype=np.float64)
    with np.errstate(divide="ignore"):
        axis = np.log(np.asarray(_C2_SEVERITY, dtype=np.float64))

    def negative(p: NDArray[np.float64]) -> float:
        mu, log_s, a = float(p[0]), float(p[1]), float(p[2])
        s = math.exp(log_s)
        upper = float(expit(a))
        f_theta = float(_vec_cdf(np.array(((theta - mu) / s,)), link)[0])
        lower = upper * f_theta / (multiple - 1.0 + f_theta)
        with np.errstate(all="ignore"):
            curve = lower + (upper - lower) * _vec_cdf((axis - mu) / s, link)
            value = float(binom.logpmf(events, trials, np.clip(curve, 0.0, 1.0)).sum())
        return -value if math.isfinite(value) else 1.0e12

    s_hat = fit.params["s"].value
    u_hat = fit.params["upper"].value
    starts = [
        [theta - math.exp(math.log(s_hat) + shift) * z, math.log(s_hat) + shift, logit(u_hat)]
        for z in (-7.0, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
        for shift in (-1.5, 0.0, 1.5)
    ]
    rng = np.random.default_rng(7)
    starts += [
        [rng.uniform(-8, 8), rng.uniform(-5, 3), rng.uniform(-1, 5)]
        for _ in range(60 - len(starts))
    ]
    best = -math.inf
    for start in starts:
        result = minimize(
            negative,
            np.array(start),
            method="Nelder-Mead",
            options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 3000, "maxfev": 3000},
        )
        best = max(best, -float(result.fun))
    return best


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_baseline_multiple_with_both_asymptotes_estimated_reports_open_lower(link: str) -> None:
    """The lower side is open (``OPEN_LOWER``, ``lo is None``), not a closed bound at 0.0126
    (logit) or 0.230 (probit). The evidence is the oracle in
    ``test_oracle_deviance_stays_negative_below_the_spurious_lower_bound``."""
    fit = _c2_fit(link)

    result = threshold(
        fit, definition=Definition.baseline_multiple(2.0), interval_method="profile", level=_LEVEL
    )

    assert result.status is Status.OK
    assert result.censoring is Censoring.OPEN_LOWER
    assert result.lo is None
    assert result.value is not None


@pytest.mark.slow
@pytest.mark.parametrize(("link", "ceiling"), [("logit", -0.5), ("probit", -0.05)])
def test_oracle_deviance_stays_negative_below_the_spurious_lower_bound(
    link: str, ceiling: float
) -> None:
    """Oracle: 60-start Nelder-Mead in the chart ``(mu, log s, logit u)`` above. Every value it
    returns is attained, so ``g`` from it can only overstate the true ``g``; ``g < ceiling < 0``
    at 0.5, 0.2, 0.0126 and 0.005 proves the profile does not cross anywhere in between and
    below, including at the lower bounds the earlier code reported (0.01257 logit, 0.2299
    probit). Independent review values are -0.839 (logit) and -0.0758 (probit); the ceilings sit
    between those and zero."""
    llf_hat = _c2_fit(link).log_likelihood
    assert llf_hat is not None

    for x in (0.5, 0.2, 0.0126, 0.005):
        g = 2.0 * (llf_hat - _c2_profile_llf(link, math.log(x))) - _CHI2
        assert g < ceiling, (link, x, g)


@pytest.mark.parametrize(("link", "deviance"), [("logit", 3.0025), ("probit", 3.7656)])
def test_fit_with_lower_asymptote_fixed_at_zero_is_inside_the_chi_square_target(
    link: str, deviance: float
) -> None:
    """Closure cross-check: the fit with the lower asymptote fixed at 0 (a different fit path)
    is the supremum of the profile as the threshold tends to 0. Its deviance against the full
    fit, 3.0025 (logit) and 3.7656 (probit), is below the 3.841 target, so no crossing exists."""
    full = _c2_fit(link)
    wall = fit_dose_response(
        _c2_observations(link),
        model="binomial",
        link=link,  # type: ignore[arg-type]
        upper="estimate",
        lower=0.0,
    )

    assert full.log_likelihood is not None and wall.log_likelihood is not None
    measured = 2.0 * (full.log_likelihood - wall.log_likelihood)
    assert measured == pytest.approx(deviance, abs=5e-3)
    assert measured < _CHI2


# --- fixed asymptotes: two close basins ----------------------------------------------------------
#
# Both designs below (linear axis, cloglog, asymptotes (0.95, 0.05), decreasing) have two basins in
# ``log s`` that are 0.5 to 1.5 apart for the thresholds near the bound. A start rule that keeps
# only well-separated starts, or scans ``log s`` on a coarse grid, can miss the better basin and
# report a bound that is too narrow. The oracle is the wide log-``s`` grid above: with the
# asymptotes fixed, ``z*`` is fixed and the profile is exactly a one-dimensional maximisation.

_D101_TARGET = 0.8 * 0.95
_D101_VARIANT = replace(_DESIGNS["d101"], successes=(17, 20, 15, 7, 3, 2))


def _oracle_deviance(design: _Design, target: float, x: float, level: float) -> float:
    """``g(x) = 2 (llf_hat - llf_p(x)) - chi2(level)`` from the wide oracle at the natural-axis
    threshold ``x``."""
    llf_hat = _decreasing_fit(design).log_likelihood
    assert llf_hat is not None
    theta = math.log(x) if design.scale == "log" else x
    chi = float(chi2.ppf(level, 1))
    return 2.0 * (llf_hat - _wide_fixed_asymptote_profile_llf(design, theta, target)) - chi


def _assert_crossing_at(
    design: _Design, target: float, level: float, hi: float | None, expected: float
) -> None:
    """The reported ``hi`` is the oracle's crossing: ``g < 0`` at ``1e-3`` relative inside it and
    ``g > 0`` at ``1e-3`` relative outside it, and it is within ``2e-2`` relative of the
    crossing located independently (``expected``, rounded from the oracle)."""
    assert hi is not None
    assert hi == pytest.approx(expected, rel=2e-2)
    assert _oracle_deviance(design, target, hi * (1 - 1e-3), level) < 0.0
    assert _oracle_deviance(design, target, hi * (1 + 1e-3), level) > 0.0


_CLOSE_BASIN_CASES = [
    pytest.param(_D101_VARIANT, 0.95, 32.19, Definition.baseline_fraction(0.8), id="variant-bf"),
    pytest.param(_D101_VARIANT, 0.95, 32.19, Definition.absolute(0.76), id="variant-abs"),
    pytest.param(
        _DESIGNS["d101"], 0.995, 39.98, Definition.baseline_fraction(0.8), id="d101-0.995-bf"
    ),
    pytest.param(_DESIGNS["d101"], 0.995, 39.98, Definition.absolute(0.76), id="d101-0.995-abs"),
]


@pytest.mark.parametrize(("design", "level", "expected", "definition"), _CLOSE_BASIN_CASES)
def test_close_basin_decreasing_upper_bound_sits_where_the_oracle_deviance_crosses_zero(
    design: _Design, level: float, expected: float, definition: Definition
) -> None:
    """Oracle: the wide log-``s`` grid (fixed asymptotes, so one free coordinate). At the
    variant (level 0.95) its ``g`` is -0.146 at 32.0 and +0.051 at 32.25; for d101 at level
    0.995 the crossing is near 39.98. The oracle's ``g`` has slope about 0.8 per unit, so
    ``1e-3`` relative from the bound it is ``+/-0.02`` or more, far above the oracle's own
    error. The earlier values were 31.2917 and 39.8974."""
    result = threshold(
        _decreasing_fit(design), definition=definition, interval_method="profile", level=level
    )

    assert result.status is Status.OK
    _assert_crossing_at(design, _D101_TARGET, level, result.hi, expected)


@pytest.mark.parametrize(
    ("design", "level", "expected"),
    [
        pytest.param(_D101_VARIANT, 0.95, 32.19, id="variant"),
        pytest.param(_DESIGNS["d101"], 0.995, 39.98, id="d101-0.995"),
    ],
)
def test_close_basin_increasing_mirror_upper_bound_sits_where_the_oracle_deviance_crosses_zero(
    design: _Design, level: float, expected: float
) -> None:
    """The increasing view of the same data (mirrored counts, ``absolute(0.24)``, the
    complement of 0.76) has the same bound. Oracle: the decreasing-frame wide log-``s`` grid,
    since the mirrored problem is the same constrained maximisation."""
    result = threshold(
        _increasing_fit(design),
        definition=Definition.absolute(0.24),
        interval_method="profile",
        level=level,
    )

    assert result.status is Status.OK
    _assert_crossing_at(design, _D101_TARGET, level, result.hi, expected)


# --- the first crossing, found lazily -------------------------------------------------------------


def _lazy_case_g(theta: float) -> float | None:
    """``g = theta - 1`` (crossing at 1.0); the inner fit fails on ``[1.2, 1.5)``."""
    if 1.2 <= theta < 1.5:
        return None
    return theta - 1.0


_LAZY_UPPER_LIMIT = 44.64  # strides land at 0.9 and about 1.53, bracketing the crossing and the gap


def test_failure_beyond_the_first_crossing_does_not_open_the_side() -> None:
    """Oracle: a synthetic ``g(theta) = theta - 1`` (the root is exactly 1.0) whose inner fit
    fails on ``[1.2, 1.5)``. The stride bracket ``[0.9, 1.53]`` holds both the crossing and the
    gap; the sub-intervals are evaluated lazily from the estimate and stop at the first sign
    change, so the failures beyond it are never evaluated and the bound is 1.0. This calls the
    private bracket solver with an injected ``g``."""
    bracket = iv._bracket_root(_lazy_case_g, 0.0, _LAZY_UPPER_LIMIT, 1.0)
    assert bracket[0] == "bracket"
    assert bracket[1] < 1.0 < 1.2 and bracket[2] > 1.5  # type: ignore[misc]

    result = iv._solve_profile_bracket(
        _lazy_case_g,
        theta_hat=0.0,
        lower_limit=-_LAZY_UPPER_LIMIT,
        upper_limit=_LAZY_UPPER_LIMIT,
        scale="linear",
        subdivisions=iv._ESTIMATED_BRACKET_SUBDIVISIONS,
    )

    assert result.censoring is Censoring.OPEN_LOWER
    assert result.hi == pytest.approx(1.0, abs=1e-8)
    assert not any("inner fit failed" in w for w in result.warnings)


def test_failure_before_the_first_crossing_still_opens_the_side() -> None:
    """Control for the lazy rule: ``g(theta) = theta - 2`` (root 2.0) with failures on
    ``[1.7, 1.8)``, which lies inside the stride bracket ``[1.53, 2.6]`` and before the crossing.
    The first sign change cannot be reached without evaluating a failed point, so the side stays
    open and says why, never a number."""

    def g(theta: float) -> float | None:
        return None if 1.7 <= theta < 1.8 else theta - 2.0

    bracket = iv._bracket_root(g, 0.0, _LAZY_UPPER_LIMIT, 1.0)
    assert bracket[0] == "bracket"
    assert bracket[1] < 1.7 and bracket[2] > 2.0  # type: ignore[misc]

    result = iv._solve_profile_bracket(
        g,
        theta_hat=0.0,
        lower_limit=-_LAZY_UPPER_LIMIT,
        upper_limit=_LAZY_UPPER_LIMIT,
        scale="linear",
        subdivisions=iv._ESTIMATED_BRACKET_SUBDIVISIONS,
    )

    assert result.hi is None
    assert any("inner fit failed" in w for w in result.warnings)


# --- a failed alternative start is reported -------------------------------------------------------


def _m2_decreasing_fit() -> Fit:
    return fit_dose_response(
        _m2_observations("decreasing"),
        model="binomial",
        link="cloglog",
        upper=1.0,
        lower="estimate",
    )


def test_alternative_start_that_raises_is_reported_in_the_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first alternative start's optimiser run raises (injected: ``_bfgs_fit`` raises on the
    second call within each inner evaluation; the first is the pilot). The threshold is still
    computed and its warnings say an alternative failed. The wording is not pinned: the stable
    substring is ``alternative``. Control: the unpatched run has no such warning."""
    fit = _m2_decreasing_fit()
    definition = Definition.absolute(0.3)
    control = threshold(fit, definition=definition, interval_method="profile", level=_LEVEL)
    assert not any("alternative" in w.lower() for w in control.warnings)

    real_bfgs = iv._bfgs_fit
    real_profile = iv._estimated_profile_llf
    calls = {"n": 0}

    def counting_profile(*args: Any, **kwargs: Any) -> float | None:
        calls["n"] = 0
        return real_profile(*args, **kwargs)

    def failing_bfgs(model: Any, start: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise np.linalg.LinAlgError("injected")
        return real_bfgs(model, start)

    monkeypatch.setattr(iv, "_estimated_profile_llf", counting_profile)
    monkeypatch.setattr(iv, "_bfgs_fit", failing_bfgs)

    result = threshold(fit, definition=definition, interval_method="profile", level=_LEVEL)

    assert result.status is Status.OK
    assert result.value is not None
    assert any("alternative" in w.lower() for w in result.warnings), result.warnings


def test_alternative_start_with_a_non_finite_log_likelihood_is_reported_in_the_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first alternative start's optimiser run completes but returns a non-finite
    log-likelihood (injected on the second ``_bfgs_fit`` call within each inner evaluation; the
    first is the pilot). That run is skipped like one that raises, and counted the same way: the
    interval's warnings say an alternative start was skipped. Control: the unpatched run has no
    such warning."""
    fit = _m2_decreasing_fit()
    definition = Definition.absolute(0.3)
    control = threshold(fit, definition=definition, interval_method="profile", level=_LEVEL)
    assert not any("alternative start(s)" in w for w in control.warnings)

    real_bfgs = iv._bfgs_fit
    real_profile = iv._estimated_profile_llf
    calls = {"n": 0}

    def counting_profile(*args: Any, **kwargs: Any) -> float | None:
        calls["n"] = 0
        return real_profile(*args, **kwargs)

    def non_finite_bfgs(model: Any, start: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            return SimpleNamespace(llf=math.nan, params=np.asarray(start, dtype=np.float64))
        return real_bfgs(model, start)

    monkeypatch.setattr(iv, "_estimated_profile_llf", counting_profile)
    monkeypatch.setattr(iv, "_bfgs_fit", non_finite_bfgs)

    result = threshold(fit, definition=definition, interval_method="profile", level=_LEVEL)

    assert result.status is Status.OK
    assert result.value is not None
    assert any("alternative start(s)" in w for w in result.warnings), result.warnings


# --- the estimate's own inner fit must not beat the reported maximum ---------------------------


def _profile_interval_with_inner_fit_at_the_estimate_shifted(
    monkeypatch: pytest.MonkeyPatch, shift: float
) -> iv.IntervalResult:
    """``profile_interval`` on the estimated-asymptote path where the inner fit at ``theta_hat``
    reports ``llf_hat + shift`` (injected; every other theta returns the real value)."""
    fit = _m2_decreasing_fit()
    definition = Definition.absolute(0.3)
    theta_hat, _ = iv.solve_value(fit, target=0.3)
    llf_hat = iv._llf_hat(fit)
    real_profile = iv._estimated_profile_llf

    def shifted_profile(theta: float, **kwargs: Any) -> float | None:
        if theta == theta_hat:
            return llf_hat + shift
        return real_profile(theta, **kwargs)

    monkeypatch.setattr(iv, "_estimated_profile_llf", shifted_profile)
    return iv.profile_interval(
        fit, definition=definition, target=0.3, level=_LEVEL, theta_hat=theta_hat
    )


def test_inner_fit_at_the_estimate_that_beats_the_reported_maximum_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the inner fit at ``theta_hat`` exceeds the reported log-likelihood by more than the
    Newton tolerance, the reported maximum is not a maximum and every bound would be measured from
    the wrong level, so the interval raises rather than return a result."""
    with pytest.raises(ValueError, match="MLE"):
        _profile_interval_with_inner_fit_at_the_estimate_shifted(monkeypatch, 1.0e-3)


def test_inner_fit_at_the_estimate_within_the_newton_tolerance_of_the_maximum_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An excess below the Newton tolerance (here 1e-8) is optimiser noise, not a better
    maximum, so the interval is computed. Control for the test above."""
    assert _NEWTON_DECREMENT_TOLERANCE > 1.0e-8
    result = _profile_interval_with_inner_fit_at_the_estimate_shifted(monkeypatch, 1.0e-8)

    assert result.lo is not None or result.hi is not None


# --- baseline_multiple's derived upper asymptote never exceeds 1 ---------------------------
#
# In the mirrored frame ``u = (F l + k - 1) / (k - S)``. Where ``F`` is near 1e-17, ``S = 1 - F``
# rounds to 1 and the quotient can come out one ulp above 1.

_TINY_F_Z_STAR = {
    "probit": np.linspace(-9.0, -8.0, 401),
    "logit": np.linspace(-41.0, -36.0, 1001),
    "cloglog": np.linspace(-41.0, -36.0, 1001),
}


@pytest.mark.parametrize("estimate_lower", [True, False], ids=["both-estimated", "upper-only"])
@pytest.mark.parametrize("link", _LINKS)
def test_baseline_multiple_derived_upper_asymptote_is_finite_and_at_most_one_at_tiny_f(
    link: str, estimate_lower: bool
) -> None:
    """Oracle: the feasibility contract ``0 <= l < u <= 1`` over a dense sweep of ``z*`` across
    the band where ``F`` is near 1e-17 (probit about -8.5, logit and cloglog -37 to -40), for
    several values of the lower asymptote (as ``logit(l)`` when both are estimated, else fixed).
    A point in the band gives asymptotes, not ``None``."""
    for l_value in (0.01, 0.1, 0.3, 0.5, 0.8, 0.95):
        extra = np.array([float(logit(l_value))]) if estimate_lower else np.array([])
        for z_star in _TINY_F_Z_STAR[link]:
            result = iv._zstar_chart_asymptotes(
                float(z_star),
                extra,
                definition=Definition.baseline_multiple(2.0),
                link=link,
                estimate_lower=estimate_lower,
                estimate_upper=True,
                lower_value=math.nan if estimate_lower else l_value,
                upper_value=math.nan,
            )
            assert result is not None, (l_value, z_star)
            lower, upper = result
            assert math.isfinite(lower) and math.isfinite(upper)
            assert 0.0 <= lower < upper <= 1.0, (l_value, z_star, lower, upper)


# --- the z* chart's log-likelihood is the full binomial log-likelihood -----------------------


@pytest.mark.parametrize("link", _LINKS)
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_zstar_log_likelihood_equals_the_full_binomial_log_likelihood_exactly_at_the_walls(
    case: _Case, link: str
) -> None:
    """Oracle: ``models._full_binomial_log_likelihood`` called here at the natural-scale
    parameters ``(mu = theta - s z*, s, u, l)``, with ``(l, u)`` from the chart's own algebra
    (checked against the defining identity above). Compared with ``==`` at ``z* = +/-40`` and
    ``log s = +/-300``, and at an interior point, so the chart's ``__call__`` cannot be using a
    different summation. Where no feasible asymptote exists the chart returns the unreachable
    constant."""
    chart = iv._ZStarLogLikelihood(**_llf_args(case, link))
    for z_star in (-40.0, -1.0, 0.5, 40.0):
        for log_s in (-300.0, 0.0, 300.0):
            params = np.array([log_s, z_star, *case.extra])
            asymptotes = _asymptotes(case, z_star, link)
            value = chart(params)
            if asymptotes is None:
                assert value == iv._UNREACHABLE_LOGLIKELIHOOD
                continue
            lower, upper = asymptotes
            s = math.exp(log_s)
            expected = _full_binomial_log_likelihood(
                _SEVERITY,
                _SUCCESSES,
                _TRIALS,
                mu=_THETA - s * z_star,
                s=s,
                upper=upper,
                lower=lower,
                scale="log",
                link=link,
            )
            assert value == expected, (z_star, log_s, value, expected)


# --- box faces, narrow basins and ``relative`` with estimated asymptotes ---------------------
#
# Three ways the estimated-asymptote inner fit could stop short of the constrained maximum, each
# tested against a maximum that avoids the package's optimiser and charts (every oracle value is an
# attained, feasible log-likelihood, so an oracle ``g < 0`` outside a reported bound proves the
# bound too narrow).
#
# * Face. With one asymptote estimated and the constraint fixing the curve at ``theta``, the
#   estimated asymptote can sit on its box wall (0 for a lower, 1 for an upper). Pinning it there
#   leaves both asymptotes fixed, so ``z*`` is a constant and the problem is one-dimensional in
#   ``log s``: the wide fixed-asymptote grid above is exact.
# * Narrow basin. With both asymptotes estimated, a step between ``theta`` and the next cell
#   (``s -> 0``, plateaus at the cell-group proportions, ``z*`` solved from the constraint) is an
#   explicit curve whose log-likelihood is an attained lower bound on the profile maximum.
# * ``relative`` with both estimated. ``z*`` is a constant, ``p_i = u S_i + l F_i`` is linear in
#   ``(u, l)`` for fixed ``s``, so the binomial log-likelihood is concave there: an exact 2-D
#   maximisation per ``log s`` on a dense grid.


@functools.cache
def _hat_llf(design: _Design) -> float:
    llf_hat = _decreasing_fit(design).log_likelihood
    assert llf_hat is not None
    return float(llf_hat)


def _theta_of(design: _Design, x: float) -> float:
    return math.log(x) if design.scale == "log" else x


def _deviance_over(design: _Design, llf_p: float, level: float) -> float:
    return 2.0 * (_hat_llf(design) - llf_p) - float(chi2.ppf(level, 1))


def _face_deviance(
    design: _Design,
    *,
    pinned: Literal["upper", "lower"],
    face: float,
    target: float,
    x: float,
    level: float,
) -> float:
    """``g(x)`` with the estimated asymptote pinned at ``face`` (decreasing frame; ``target`` in
    that frame), by the wide fixed-asymptote log-``s`` grid."""
    fixed = replace(design, upper=face) if pinned == "upper" else replace(design, lower=face)
    llf_p = _wide_fixed_asymptote_profile_llf(fixed, _theta_of(design, x), target)
    return _deviance_over(design, llf_p, level)


def _step_witness_llf(design: _Design, x: float, target_of: Any) -> float:
    """Log-likelihood of an explicit curve through ``(theta, target)`` that is a step just beyond
    ``theta`` (decreasing frame, an upper-side threshold). Cells left of ``theta`` sit on the upper
    plateau, cells beyond the first cell ``c`` right of ``theta`` on the lower plateau, each at its
    cell-group proportion (with no cell beyond ``c`` the lower plateau is half of ``c``'s
    proportion). ``c`` itself is saturated at its own proportion by the scale, ``s = (theta_c -
    theta) / (z_c - z*)``, and ``z*`` is solved from the constraint. Where those orderings fail,
    the plain step (``c`` on the lower plateau, ``s = 1e-9``) is used. Every value is an attained
    feasible log-likelihood, so the larger is a lower bound on the profile maximum; ``-inf`` where
    neither exists."""
    theta = _theta_of(design, x)
    axis = _theta_axis(np.asarray(design.severity, dtype=np.float64), design.scale)
    k = np.asarray(design.successes, dtype=np.float64)
    n = np.asarray(design.trials, dtype=np.float64)
    left = axis < theta
    if not left.any() or left.all():
        return -math.inf
    upper = float(k[left].sum() / n[left].sum())
    right = np.flatnonzero(~left)
    best = -math.inf

    lower = float(k[right].sum() / n[right].sum())
    target = target_of(upper, lower)
    if lower < target < upper:
        z_star = _inv_cdf((upper - target) / (upper - lower), design.link)
        s = 1.0e-9
        best = float(_llf(design, mu=theta - s * z_star, s=s, upper=upper, lower=lower))

    c, rest = right[0], right[1:]
    p_c = float(k[c] / n[c])
    lower = float(k[rest].sum() / n[rest].sum()) if rest.size else p_c / 2.0
    target = target_of(upper, lower)
    if lower < p_c < target < upper:
        z_star = _inv_cdf((upper - target) / (upper - lower), design.link)
        z_c = _inv_cdf((upper - p_c) / (upper - lower), design.link)
        s = float((axis[c] - theta) / (z_c - z_star))
        best = max(best, float(_llf(design, mu=theta - s * z_star, s=s, upper=upper, lower=lower)))
    return best


def _relative_profile_llf(design: _Design, x: float, p: float) -> float:
    """Constrained maximum of ``relative(p)`` with both asymptotes estimated. The curve at ``theta``
    is ``u - p (u - l)``, so ``F(z*) = p``; for a fixed ``s``, ``p_i = u S_i + l F_i`` with
    ``F_i = F(z* + (theta_i - theta) / s)``. The log-likelihood is concave in ``(u, l)``: SLSQP
    on the box ``0 <= l <= u <= 1``. The outer search is a ``log s`` grid refined by a bounded
    scalar search around the best three cells. The value returned is exact at the clamped
    solution."""
    theta = _theta_of(design, x)
    axis = _theta_axis(np.asarray(design.severity, dtype=np.float64), design.scale)
    k = np.asarray(design.successes, dtype=np.float64)
    n = np.asarray(design.trials, dtype=np.float64)
    z_star = _inv_cdf(p, design.link)
    finite = axis[np.isfinite(axis)]
    span = float(finite.max() - finite.min())

    def at_s(log_s: float) -> float:
        with np.errstate(all="ignore"):
            z = z_star + (axis - theta) / math.exp(log_s)
            cdf = _vec_cdf(z, design.link)
        survival = 1.0 - cdf

        def negative(v: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            prob = np.clip(v[0] * survival + v[1] * cdf, 1e-12, 1.0 - 1e-12)
            value = float(np.sum(k * np.log(prob) + (n - k) * np.log1p(-prob)))
            slope = k / prob - (n - k) / (1.0 - prob)
            return -value, -np.array([np.sum(slope * survival), np.sum(slope * cdf)])

        best = -math.inf
        for start in ((0.9, 0.1), (float(np.max(k / n)), float(np.min(k / n)))):
            result = minimize(
                negative,
                np.array(start),
                jac=True,
                method="SLSQP",
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                constraints=[{"type": "ineq", "fun": lambda v: v[0] - v[1]}],
                options={"ftol": 1e-13, "maxiter": 300},
            )
            upper = min(max(float(result.x[0]), 0.0), 1.0)
            lower = min(max(float(result.x[1]), 0.0), upper)
            exact = float(
                binom.logpmf(k, n, np.clip(upper * survival + lower * cdf, 0.0, 1.0)).sum()
            )
            if math.isfinite(exact):
                best = max(best, exact)
        return best

    grid = np.linspace(math.log(span) - 12.0, math.log(span) + 3.0, 301)
    values = np.array([at_s(float(t)) for t in grid])
    result = float(np.max(values))
    for index in np.argsort(values)[-3:]:
        refined = minimize_scalar(
            lambda t: -at_s(float(t)),
            bounds=(grid[max(index - 1, 0)], grid[min(index + 1, grid.size - 1)]),
            method="bounded",
            options={"xatol": 1e-8},
        )
        result = max(result, -float(refined.fun))
    return result


# --- the box face --------------------------------------------------------------------------------

_E: Literal["estimate"] = "estimate"
_FACE_D147 = replace(_DESIGNS["d147"], link="cloglog", upper=0.95, lower=_E)
_FACE_D101 = _Design(
    "linear",
    "cloglog",
    _E,
    0.05,
    (0.0, 4.915, 19.341, 36.922, 74.506, 77.938),
    (17, 20, 15, 7, 3, 2),  # the increasing counts (3, 0, 5, 13, 17, 18) complemented
    (20,) * 6,
)
_FACE_D260 = _Design(
    "log",
    "cloglog",
    _E,
    0.05,
    (3.052, 5.879, 6.611, 49.88, 62.578, 168.689),
    (71, 56, 42, 6, 11, 9),
    (100,) * 6,
)


@dataclass(frozen=True)
class _FaceCase:
    """One estimated asymptote, pinned on its box wall. ``design`` is in the decreasing frame, the
    frame of the oracle, and ``view`` is the orientation the package is given. ``target`` is the
    constraint in that frame. ``pre_fix`` is the lower bound the code reported before faces were
    tried, ``crossing`` the oracle's crossing (found by bracketing the oracle's own sign change)."""

    design: _Design
    view: Literal["dec", "inc"]
    definition: Definition
    level: float
    pinned: Literal["upper", "lower"]
    face: float
    target: float
    pre_fix: float
    crossing: float


_FACE_CASES = {
    # decreasing, upper 0.95, lower estimated -> lower pinned at 0; target 0.782
    "d147-absolute-cloglog": _FaceCase(
        _FACE_D147, "dec", Definition.absolute(0.782), 0.95, "lower", 0.0, 0.782, 0.306611, 0.30426
    ),
    # increasing view of the same data, lower 0.05, upper estimated; baseline is the lower
    # asymptote, so the decreasing-frame target is 1 - 3.322 (1 - 0.95)
    "d147-multiple-increasing": _FaceCase(
        _FACE_D147,
        "inc",
        Definition.baseline_multiple(3.322),
        0.95,
        "lower",
        0.0,
        1.0 - 3.322 * 0.05,
        0.130410,
        0.128791,
    ),
    # increasing, upper 0.95, lower estimated -> decreasing frame: upper estimated, lower 0.05,
    # pinned at 1; absolute(0.347) increasing is absolute(0.653) decreasing
    "linear-absolute-increasing": _FaceCase(
        _FACE_D101,
        "inc",
        Definition.absolute(0.347),
        0.95,
        "upper",
        1.0,
        1.0 - 0.347,
        18.5664,
        18.5146,
    ),
    # decreasing, upper estimated, lower 0.05, pinned at 1; baseline_fraction target is f u = 0.62
    "d260-fraction-cloglog": _FaceCase(
        _FACE_D260,
        "dec",
        Definition.baseline_fraction(0.62),
        0.90,
        "upper",
        1.0,
        0.62,
        3.24525,
        3.1348,
    ),
}
_FACE_PARAMS = [
    "d147-absolute-cloglog",
    "d147-multiple-increasing",
    pytest.param("linear-absolute-increasing", marks=_SLOW),
    pytest.param("d260-fraction-cloglog", marks=_SLOW),
]


def _face_g(case: _FaceCase, x: float) -> float:
    return _face_deviance(
        case.design,
        pinned=case.pinned,
        face=case.face,
        target=case.target,
        x=x,
        level=case.level,
    )


@functools.cache
def _face_threshold(name: str) -> Threshold:
    case = _FACE_CASES[name]
    fit = _increasing_fit(case.design) if case.view == "inc" else _decreasing_fit(case.design)
    return threshold(fit, definition=case.definition, interval_method="profile", level=case.level)


@pytest.mark.parametrize("name", _FACE_PARAMS)
def test_face_oracle_is_negative_at_the_pre_fix_bound_and_crosses_zero_at_the_recorded_point(
    name: str,
) -> None:
    """Oracle: the estimated asymptote pinned on its wall, so both asymptotes are fixed and the
    constraint fixes ``z*``; the inner problem is one-dimensional in ``log s`` and is solved by the
    wide log-``s`` grid with bounded refinement. Its value is an attained feasible
    log-likelihood, so ``g < 0`` at the bound the code reported before faces were tried proves
    that bound too narrow. The oracle crosses zero within ``2e-3`` relative of the recorded
    crossing (``g`` moves by 0.002 or more over that distance, against an oracle error near
    1e-8)."""
    case = _FACE_CASES[name]

    assert _face_g(case, case.pre_fix) < 0.0
    assert _face_g(case, case.crossing * (1.0 - 2.0e-3)) > 0.0
    assert _face_g(case, case.crossing * (1.0 + 2.0e-3)) < 0.0


@pytest.mark.parametrize("name", _FACE_PARAMS)
def test_lower_bound_reaches_the_crossing_the_box_face_oracle_finds(name: str) -> None:
    """Oracle: as above. The face model is a feasible curve, so the true lower bound is at or
    beyond the face crossing: ``lo <= crossing (1 + 2e-3)``."""
    case = _FACE_CASES[name]
    result = _face_threshold(name)

    assert result.lo is not None
    assert result.lo <= case.crossing * (1.0 + 2.0e-3), (result.lo, case.crossing)


@pytest.mark.parametrize("name", _FACE_PARAMS)
def test_box_face_oracle_deviance_is_not_negative_just_outside_the_lower_bound(
    name: str,
) -> None:
    """Oracle: as above. An oracle ``g < 0`` just outside the reported bound would put an attained
    feasible curve inside the confidence set beyond it. At ``1e-3`` relative outside, ``g >= 0``."""
    case = _FACE_CASES[name]
    result = _face_threshold(name)

    assert result.lo is not None
    assert _face_g(case, result.lo * (1.0 - 1.0e-3)) >= 0.0


# --- a basin narrower than the grid -----------------------------------------------------------


@dataclass(frozen=True)
class _BasinCase:
    """``gap`` holds natural-axis thresholds below the tested level at which the oracle's ``g``
    is negative; ``floor`` is the value ``hi`` must reach."""

    design: _Design
    definition: Definition
    level: float
    target_of: Any
    gap: tuple[float, ...]
    floor: float


_BASIN_CASES = {
    # probit, both estimated, baseline_fraction: g = -1.989 up to the cell at 7.714
    "d18-fraction-0.995": _BasinCase(
        _Design(
            "log",
            "probit",
            _E,
            _E,
            (0.0, 0.705, 7.714, 20.729, 26.179),
            (256, 241, 158, 122, 95),
            (300,) * 5,
        ),
        Definition.baseline_fraction(0.821),
        0.995,
        lambda u, lower: 0.821 * u,
        (7.04, 7.5, 7.713),
        7.713 * (1.0 - 2.0e-3),
    ),
    # logit, both estimated: g = -1.510 up to the top cell at 40.792
    "top-cell-fraction-0.995": _BasinCase(
        _Design(
            "log",
            "logit",
            _E,
            _E,
            (0.0, 0.83, 2.352, 3.091, 40.792),
            (9, 7, 4, 5, 1),
            (10,) * 5,
        ),
        Definition.baseline_fraction(0.7),
        0.995,
        lambda u, lower: 0.7 * u,
        (28.6, 35.0, 40.79),
        40.79 * (1.0 - 2.0e-3),
    ),
    # linear axis, logit, both estimated, absolute: g = -1.1396 up to the cell at 67.4
    "d73-absolute-0.95": _BasinCase(
        _DESIGNS["d73"],
        Definition.absolute(0.8),
        0.95,
        lambda u, lower: 0.8,
        (65.7, 66.5, 67.39),
        67.39 * (1.0 - 2.0e-3),
    ),
}
_BASIN_PARAMS = [
    "d18-fraction-0.995",
    pytest.param("top-cell-fraction-0.995", marks=_SLOW),
    pytest.param("d73-absolute-0.95", marks=_SLOW),
]


@pytest.mark.parametrize("name", _BASIN_PARAMS)
def test_step_curve_witness_keeps_the_deviance_negative_up_to_the_next_cell(name: str) -> None:
    """Oracle: an explicit feasible curve through ``(theta, target)`` that is a step just beyond
    ``theta``: plateaus at the cell-group proportions, the first cell beyond ``theta`` saturated at
    its own proportion by a scale of order its distance from ``theta``, ``z*`` solved from the
    constraint (``_step_witness_llf``). Its log-likelihood is attained, so ``g < -1`` at every
    ``gap`` threshold proves the interval extends at least that far. No optimiser is used."""
    case = _BASIN_CASES[name]
    for x in case.gap:
        llf = _step_witness_llf(case.design, x, case.target_of)
        assert _deviance_over(case.design, llf, case.level) < -1.0, x


@pytest.mark.parametrize("name", _BASIN_PARAMS)
def test_upper_bound_reaches_the_next_cell_when_a_step_curve_keeps_the_deviance_negative(
    name: str,
) -> None:
    """Oracle: the step-curve witness above. It puts every threshold up to the cell inside the
    confidence set, so ``hi`` is at or beyond the cell, within ``2e-3`` relative. The bounds
    reported before the change (7.034, 28.53 and 65.62) fall short of it by 9%, 30% and 3%."""
    case = _BASIN_CASES[name]
    fit = _decreasing_fit(case.design)

    result = threshold(fit, definition=case.definition, interval_method="profile", level=case.level)

    assert result.hi is not None
    assert result.hi >= case.floor, result.hi


# --- ``relative`` with both asymptotes estimated -------------------------------------------

_RELATIVE_C = _Design(
    "log",
    "probit",
    _E,
    _E,
    (3.052, 5.879, 6.611, 49.88, 62.578, 168.689),
    (68, 55, 38, 7, 9, 14),
    (100,) * 6,
)
_RELATIVE_CASES = {
    # crossing located by bracketing the oracle's own sign change (5.9397)
    "relative-probit-0.995": (_RELATIVE_C, 0.662, 0.995, "lo", 5.9397, 6.435524705958676),
    # on this design the exact crossing is the value returned before the change (32.37517)
    "relative-logit-0.95": (_DESIGNS["d153"], 0.5, 0.95, "lo", 32.37517, 32.57404611504277),
}
_RELATIVE_PARAMS = [
    pytest.param("relative-probit-0.995", marks=_SLOW),
    pytest.param("relative-logit-0.95", marks=_SLOW),
]


def _relative_g(design: _Design, p: float, level: float, x: float) -> float:
    return _deviance_over(design, _relative_profile_llf(design, x, p), level)


@pytest.mark.parametrize("name", _RELATIVE_PARAMS)
def test_relative_oracle_is_negative_at_the_pre_fix_bound_and_crosses_zero_at_the_crossing(
    name: str,
) -> None:
    """Oracle: ``relative(p)`` fixes ``F(z*) = p``; for each ``log s`` of a dense grid (refined by a
    bounded search) the log-likelihood is concave in ``(u, l)`` and is maximised by SLSQP on the
    box. The value is exact at the clamped solution. ``g < 0`` at the bound reported before the
    change; it crosses zero within ``1e-3`` relative of the recorded crossing (``|g|`` is about
    0.02 to 0.04 there)."""
    design, p, level, _, crossing, pre_fix = _RELATIVE_CASES[name]

    assert _relative_g(design, p, level, pre_fix) < 0.0
    assert _relative_g(design, p, level, crossing * (1.0 - 1.0e-3)) > 0.0
    assert _relative_g(design, p, level, crossing * (1.0 + 1.0e-3)) < 0.0


@pytest.mark.parametrize("name", _RELATIVE_PARAMS)
def test_relative_lower_bound_sits_where_the_concave_oracle_deviance_crosses_zero(
    name: str,
) -> None:
    """Oracle: the concave per-``s`` oracle above, exact for ``relative``. The reported ``lo`` has
    ``g >= 0`` at ``1e-3`` relative outside and ``g <= 0`` at ``1e-3`` inside, and is no further
    in than the crossing within ``1e-3``."""
    design, p, level, _, crossing, _ = _RELATIVE_CASES[name]
    fit = _decreasing_fit(design)

    result = threshold(
        fit, definition=Definition.relative(p), interval_method="profile", level=level
    )

    assert result.lo is not None
    assert result.lo <= crossing * (1.0 + 1.0e-3), result.lo
    assert _relative_g(design, p, level, result.lo * (1.0 - 1.0e-3)) >= 0.0
    assert _relative_g(design, p, level, result.lo * (1.0 + 1.0e-3)) <= 0.0


# --- the sign alone is enough away from the interval's edge --------------------------------------


def _record_inner_evaluations(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[float, list[float]]]:
    """Patch ``iv._estimated_profile_llf`` and ``iv._bfgs_fit`` to record, for each inner
    evaluation, ``(theta, [llf of each BFGS run in call order])``."""
    records: list[tuple[float, list[float]]] = []
    real_bfgs = iv._bfgs_fit
    real_profile = iv._estimated_profile_llf

    def recording_profile(theta: float, *args: Any, **kwargs: Any) -> float | None:
        records.append((theta, []))
        return real_profile(theta, *args, **kwargs)

    def recording_bfgs(model: Any, start: Any) -> Any:
        result = real_bfgs(model, start)
        records[-1][1].append(float(result.llf))
        return result

    monkeypatch.setattr(iv, "_estimated_profile_llf", recording_profile)
    monkeypatch.setattr(iv, "_bfgs_fit", recording_bfgs)
    return records


def test_evaluation_whose_first_run_is_well_inside_the_interval_runs_no_other_optimiser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assumes: each inner evaluation is one call of ``iv._estimated_profile_llf`` (looked up by
    name at call time), and the first ``iv._bfgs_fit`` call inside it is the run from the fit's own
    parameters; any later ``_bfgs_fit`` call inside the same evaluation is an alternative start
    or a face fit. Oracle: the first run's own log-likelihood, a lower bound on the evaluation's
    (the Newton polish and the alternatives can only raise it). Where that first run alone gives
    ``g < -1`` the sign is settled, so no other run is needed: exactly one ``_bfgs_fit`` call.
    The evaluation at the estimate itself is excluded, and at least one evaluation must qualify."""
    fit = _m2_decreasing_fit()
    definition = Definition.absolute(0.3)
    control = threshold(fit, definition=definition, interval_method="profile", level=_LEVEL)
    assert control.value is not None
    llf_hat = fit.log_likelihood
    assert llf_hat is not None
    records = _record_inner_evaluations(monkeypatch)

    threshold(fit, definition=definition, interval_method="profile", level=_LEVEL)

    inside = [
        (theta, runs)
        for theta, runs in records
        if runs
        and abs(theta - control.value) > 1.0e-9 * (1.0 + abs(control.value))
        and 2.0 * (llf_hat - runs[0]) - _CHI2 < -1.0
    ]
    assert inside, "no evaluation had a first run well inside the interval"
    for theta, runs in inside:
        assert len(runs) == 1, (theta, len(runs))


def test_alternatives_still_run_at_the_estimate_and_where_the_first_run_is_outside_the_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assumes: as in the test above. The sign shortcut must not remove the alternatives where the
    first run's ``g >= 0`` (the profile may be higher than the first run found, so the sign is
    not settled) or at the estimate (where the value feeds the interval's centre). Oracle: the
    first run's log-likelihood as above; those evaluations make more than one ``_bfgs_fit``
    call."""
    fit = _m2_decreasing_fit()
    definition = Definition.absolute(0.3)
    control = threshold(fit, definition=definition, interval_method="profile", level=_LEVEL)
    assert control.value is not None
    llf_hat = fit.log_likelihood
    assert llf_hat is not None
    records = _record_inner_evaluations(monkeypatch)

    threshold(fit, definition=definition, interval_method="profile", level=_LEVEL)

    at_estimate = [
        runs for theta, runs in records if abs(theta - control.value) <= 1.0e-9 * (1.0 + abs(theta))
    ]
    outside = [runs for _, runs in records if runs and 2.0 * (llf_hat - runs[0]) - _CHI2 >= 0.0]
    assert at_estimate and outside
    for runs in at_estimate + outside:
        assert len(runs) > 1


# --- known residual: upper bounds the inner fit still reports too narrow -------------------------
#
# Five designs, both asymptotes estimated, on which independent attained-likelihood witnesses show
# the profile deviance is still below the chi-square target well beyond the reported ``hi``. The
# witnesses hold today (the non-xfail witness test below); the tests marked
# ``xfail(strict=True)`` pin the package's bound, so a fix turns them into XPASS,
# which fails the suite until the marker is removed.
#
# * ``step`` (design B): the profile is flat at ``g = -0.1196`` from 10.85 up to the tested level
#   11.612, where the best curve is a step between 2.078 and 11.612. Witness:
#   ``_step_witness_llf``, an explicit curve with no optimiser.
# * ``wall`` (designs D1 to D4): far extrapolation. The constrained maximum sits at the
#   reachability wall, where the lower asymptote (decreasing frame) tends to the target while the
#   transition stays inside the data. Witness: ``_wall_fit`` (Nelder-Mead over ``(mu, log s,
#   logit u)`` with the lower asymptote set to the target, written here) gives the wall supremum;
#   ``_wall_deviance`` then places the crossing at any ``x`` by lowering the lower asymptote just
#   below the target (``l = u - (u - target) / F(z*)``), which keeps the same ``(mu, s, u)`` and
#   the same likelihood to the working precision (the lower asymptote moves by far less than
#   ``1e-12`` relative at ``z* >= 50``).


@dataclass(frozen=True)
class _UnderCoveredCase:
    design: _Design  # decreasing frame, both asymptotes estimated
    definition: Definition  # in the frame of ``view``
    level: float
    target_of: Any  # (u, l) -> target, decreasing frame
    view: Literal["inc", "dec"]
    kind: Literal["step", "wall"]
    witnesses: tuple[float, ...]  # thresholds beyond the reported bound, on the natural axis
    ceiling: float  # every witness deviance must be below this (negative)
    bound: float  # a correct hi is at or beyond this, or the side is open


def _complement_design(
    scale: str,
    link: Literal["probit", "logit", "cloglog"],
    severity: tuple[float, ...],
    events: tuple[int, ...],
    trials: tuple[int, ...],
) -> _Design:
    """Decreasing frame of an increasing dataset: successes are the non-events."""
    return _Design(
        scale,
        link,
        _E,
        _E,
        severity,
        tuple(n - k for k, n in zip(events, trials, strict=True)),
        trials,
    )


_UNDER_COVERED = {
    "B-step": _UnderCoveredCase(
        _complement_design(
            "log",
            "cloglog",
            (0.0, 2.078, 11.612, 20.614, 40.39, 60.449, 162.712),
            (66, 83, 233, 235, 239, 244, 241),
            (300,) * 7,
        ),
        Definition.absolute(0.66),
        0.9,
        lambda u, lower: 1.0 - 0.66,
        "inc",
        "step",
        (10.95, 11.3, 11.611),
        -0.05,
        11.612 * (1.0 - 1.0e-6),
    ),
    "D1-wall": _UnderCoveredCase(
        _complement_design(
            "linear",
            "logit",
            (0.0, 4.915, 19.341, 36.922, 74.506, 77.938),
            (4, 1, 6, 12, 18, 17),
            (20,) * 6,
        ),
        Definition.absolute(0.789),
        0.9,
        lambda u, lower: 1.0 - 0.789,
        "inc",
        "wall",
        (1000.0, 3154.0),
        -0.5,
        3154.0,
    ),
    "D2-wall": _UnderCoveredCase(
        _Design(
            "linear",
            "cloglog",
            _E,
            _E,
            (0.0, 5.933, 28.364, 73.146, 80.612, 97.057),
            (288, 291, 294, 268, 218, 151),
            (300,) * 6,
        ),
        Definition.baseline_fraction(0.554),
        0.99,
        lambda u, lower: 0.554 * u,
        "dec",
        "wall",
        (1500.0, 9700.0),
        -5.0,
        9700.0,
    ),
    "D3-wall": _UnderCoveredCase(
        _complement_design(
            "linear",
            "cloglog",
            (0.0, 11.823, 19.899, 38.872, 58.246, 60.065, 66.593, 75.892),
            (1, 1, 1, 1, 2, 1, 4, 6),
            (10,) * 8,
        ),
        Definition.absolute(0.496),
        0.8,
        lambda u, lower: 1.0 - 0.496,
        "inc",
        "wall",
        (1500.0, 2502.0),
        -1.0,
        2502.0,
    ),
    "D4-wall": _UnderCoveredCase(
        _Design(
            "linear",
            "logit",
            _E,
            _E,
            (0.0, 8.742, 15.612, 30.255, 67.4, 91.752, 94.917),
            (89, 93, 88, 87, 58, 42, 26),
            (100,) * 7,
        ),
        Definition.absolute(0.285),
        0.99,
        lambda u, lower: 0.285,
        "dec",
        "wall",
        (700.0, 2282.0),
        -5.0,
        2282.0,
    ),
}


@functools.cache
def _wall_fit(name: str) -> tuple[float, float, float]:
    """``(mu, s, u)`` maximising the log-likelihood with the lower asymptote equal to the target
    (decreasing frame). The target may depend on ``u`` (``baseline_fraction``). Nelder-Mead from
    nine starts around the unconstrained fit, in a chart written here."""
    case = _UNDER_COVERED[name]
    design = case.design
    fit = _decreasing_fit(design)
    assert fit.params is not None
    mu_hat, s_hat = fit.params["mu"].value, fit.params["s"].value
    u_hat = fit.params["upper"].value

    def negative(p: NDArray[np.float64]) -> float:
        mu, s, u = float(p[0]), math.exp(p[1]), float(expit(p[2]))
        target = case.target_of(u, 0.0)
        if not 0.0 < target < u:
            return 1.0e12
        with np.errstate(all="ignore"):
            value = float(_llf(design, mu=mu, s=s, upper=u, lower=target))
        return -value if math.isfinite(value) else 1.0e12

    best: Any = None
    for shift in (-1.0, 0.0, 1.0):
        for step in (-1.0, 0.0, 1.0):
            start = np.array([mu_hat + step * s_hat, math.log(s_hat) + shift, logit(u_hat)])
            result = minimize(
                negative,
                start,
                method="Nelder-Mead",
                options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 6000, "maxfev": 6000},
            )
            if best is None or result.fun < best.fun:
                best = result
    return float(best.x[0]), math.exp(best.x[1]), float(expit(best.x[2]))


def _wall_deviance(name: str, x: float) -> float:
    """``g(x)`` bounded above by an explicit feasible curve: the wall fit's ``(mu, s, u)`` with the
    lower asymptote ``l = u - (u - target) / F((x - mu) / s)``, so the curve passes through
    ``(x, target)``. ``l`` is held at ``target * (1 - 1e-12)`` where ``F`` rounds to 1; the curve
    then misses the target by about ``1e-12``, far below the deviance margins asserted. The
    attained log-likelihood is a lower bound on the profile maximum, so the returned ``g`` is an
    upper bound on the true ``g``."""
    case = _UNDER_COVERED[name]
    design = case.design
    mu, s, u = _wall_fit(name)
    target = case.target_of(u, 0.0)
    with np.errstate(all="ignore"):
        f_at_x = float(_vec_cdf(np.array([(x - mu) / s]), design.link)[0])
        lower = min(u - (u - target) / f_at_x, target * (1.0 - 1.0e-12))
        assert 0.0 <= lower < target < u
        llf_curve = float(_llf(design, mu=mu, s=s, upper=u, lower=lower))
    return _deviance_over(design, llf_curve, case.level)


def _witness_deviance(name: str, x: float) -> float:
    case = _UNDER_COVERED[name]
    if case.kind == "step":
        with np.errstate(all="ignore"):
            llf = _step_witness_llf(case.design, x, case.target_of)
        return _deviance_over(case.design, llf, case.level)
    return _wall_deviance(name, x)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(_UNDER_COVERED))
def test_witness_keeps_the_deviance_negative_beyond_the_reported_upper_bound(name: str) -> None:
    """Oracle: an explicit feasible curve per case (``_step_witness_llf`` for the step case,
    ``_wall_deviance`` for the far-extrapolation cases). Its attained log-likelihood is a lower
    bound on the profile maximum, so ``g < 0`` at each witness proves the true interval extends
    to it. The reported bounds are 10.89, 847.8, 1325.3, 1371.4 and 646.8, all below the first
    witness. This test is not expected to fail."""
    case = _UNDER_COVERED[name]
    for x in case.witnesses:
        assert _witness_deviance(name, x) < case.ceiling, x


def _under_covered_upper_bound(name: str) -> Threshold:
    case = _UNDER_COVERED[name]
    fit = _increasing_fit(case.design) if case.view == "inc" else _decreasing_fit(case.design)
    return threshold(fit, definition=case.definition, interval_method="profile", level=case.level)


def _expect_inner_fit_miss(reason: str) -> Any:
    return [
        pytest.mark.slow,
        pytest.mark.xfail(strict=True, reason=reason),
    ]


_STEP_MISS = "inner fit misses a step basin narrower than the starting grid"
_WALL_MISS = "inner fit misses the reachability-wall maximum far beyond the tested range"


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("B-step", marks=_expect_inner_fit_miss(_STEP_MISS)),
        pytest.param("D1-wall", marks=_expect_inner_fit_miss(_WALL_MISS)),
        pytest.param("D2-wall", marks=_expect_inner_fit_miss(_WALL_MISS)),
        pytest.param("D3-wall", marks=_expect_inner_fit_miss(_WALL_MISS)),
        pytest.param("D4-wall", marks=_expect_inner_fit_miss(_WALL_MISS)),
    ],
)
def test_upper_bound_reaches_the_witnessed_extent_or_the_side_is_open(name: str) -> None:
    """Pins the known residual: the reported ``hi`` is still short of the extent the witnesses
    prove (``_UnderCoveredCase.bound``). A correct ``hi`` is at or beyond it, or the upper side is
    open. Strict xfail: a fix makes this XPASS, and the marker is then to be removed."""
    result = _under_covered_upper_bound(name)

    assert result.hi is None or result.hi >= _UNDER_COVERED[name].bound
