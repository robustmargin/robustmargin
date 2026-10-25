"""Fieller interval shapes: the full decision tree.

This file computes the closed form independently and compares ``ratio_interval()`` against it:
solve ``A*rho^2 - 2*B*rho + K = 0`` for ``A = theta_b^2 - z^2*Vb``,
``B = theta_a*theta_b - z^2*C``, ``K = theta_a^2 - z^2*Va`` (``C = 0`` under
``dependence="independent"``, so ``B > 0`` whenever both severities are positive); the confidence
set is ``{rho : A*rho^2 - 2*B*rho + K <= 0}``, intersected with the positive parameter space
before being reported. The full branch table (``_expected_shape_and_bounds`` below is this table,
executable):

- **``A > 0``** (``g < 1``): always ``BOUNDED`` -- the discriminant is always positive here (the
  point estimate itself is a root of the boundary). If the lower root is negative (``K <= 0``),
  it is clipped to ``0`` -- a negative ratio is not a possible one -- with ``hi`` untouched and
  the unclipped root named in ``warnings``. The upper-only geometry stays ``BOUNDED``.
- **``A == 0``** (``g == 1`` exactly): the quadratic degenerates to the linear
  ``-2*B*rho + K = 0``, root ``rho = K/(2*B)``. If ``K > 0`` this is a genuine positive lower
  bound -- ``HALF_OPEN`` with ``lo = K/(2*B)``. If ``K <= 0`` the root is at or below zero, so
  intersected with the positive axis the set is all of ``(0, inf)`` -- ``UNBOUNDED``.
- **``A < 0``** with a non-positive discriminant: ``UNBOUNDED`` (there are no real roots to
  intersect at all).
- **``A < 0``** with a positive discriminant: the two roots are opposite-signed whenever
  ``K > 0`` (product of roots ``= K/A < 0``), so intersected with the positive axis only the
  positive root survives -- ``HALF_OPEN`` with ``lo`` = that positive root. (Reporting this as
  ``EXCLUSIVE`` with a negative lower bound would be wrong: this occurs on an open region of the
  parameter space, not only in a measure-zero case.) When ``K <= 0`` instead, both roots are
  negative and the set is again all of ``(0, inf)`` -- ``UNBOUNDED``.
- **``EXCLUSIVE`` is unreachable through ``ratio_interval()`` in this release.** It would need
  same-signed roots with ``A < 0``, which needs ``B <= 0``, which needs ``C > 0`` -- only
  possible once ``dependence="paired"`` is implemented. The member stays in ``IntervalShape`` and
  ``Ratio``'s invariants for it stay enforced (tested directly, not through ``ratio_interval``,
  in ``TestExclusiveRemainsAValidRatioShapeForPairedDependence``).

**The primary versions of every shape here construct their ``Fit``/``Threshold`` objects
directly rather than fitting real data**, so the expected ``Va``/``Vb`` are known exactly and
``ratio_interval()``'s output can be checked against the closed form above to a tight tolerance,
not just by shape name. Every case here goes through the public ``ratio_interval()`` end-to-end
on hand-built inputs, which is the smallest thing that can be tested through the public API --
**with one documented exception**, ``TestEmptyConfidenceSetRaises`` at the end of this file: the
``A == 0, B == 0, K > 0`` empty-confidence-set case is mathematically
unreachable through any real ``Threshold`` (``B = theta_a*theta_b == 0`` forces
``theta_a == 0``, which in turn forces ``K = -z^2*Va <= 0``, so ``K > 0`` can never coincide with
it for a genuine, non-negative delta-method variance) while ``dependence="independent"`` fixes
``C = 0``; that one test calls ``robustmargin.ratio``'s private ``_fieller_shape`` directly, with a
deliberately-impossible negative ``va``, and says so rather than pretending a public-API route
exists.

**``g >= 1`` is also reachable through a real fit, confirmed by search, not assumed impossible.**
It needs a denominator threshold whose relative log-scale uncertainty exceeds ``1/z`` (about 51%
at ``level=0.95``), which turns out to need a design sitting right at the edge of separation: two
severity levels, a small ``n`` (20), and a near-flat, barely-decreasing success pattern
(``11/20`` then ``9/20``) that barely constrains the slope at all.
``TestHalfOpenAndUnboundedFromRealFits`` below reproduces both ``HALF_OPEN`` and ``UNBOUNDED``
this way (same poorly-determined denominator fit paired with a tightly- or loosely-determined
numerator respectively), confirming the constructed cases above are not testing a mathematical
curiosity that never arises in practice -- only that it takes a design bad enough to be at the
edge of ``SEPARATION``/``NOT_CONVERGED`` territory, not an ordinary one. Because a real fit's
``Va``/``Vb`` are whatever the optimizer converges to (not a round number chosen ahead of time),
that class checks ``shape`` and containment rather than exact endpoints -- the exact-value check
is what the constructed cases above are for.

The natural-scale variance identity used to build these fixtures precisely,
``Va = value^2 * sigma_log^2`` (``sigma_log`` being the same log-scale delta standard error
``intervals.delta_interval`` already computes internally, and the same quantity
``tests/reference/test_threshold_drc_ed.py`` reconstructs from a real ``Threshold``'s own
``lo``/``hi``), is an exact consequence of the delta method's chain rule
(``Var(exp(theta)) = exp(theta)^2 * Var(theta)`` to first order, for *any* function of the
underlying parameters, not an extra approximation on top of it) -- not an assumption specific to
this test file.
"""

from __future__ import annotations

import math

import pytest
from scipy.stats import norm

from robustmargin import (
    Axis,
    Cell,
    Censoring,
    Definition,
    Fit,
    IntervalShape,
    Observations,
    Ratio,
    Status,
    Threshold,
    fit_dose_response,
    ratio_interval,
    threshold,
)
from robustmargin.models import Covariance, Parameter
from robustmargin.ratio import _fieller_shape

_AXIS = Axis(name="severity", unit="unit", scale="log")
_LEVEL = 0.95
_Z = float(norm.ppf(0.5 + _LEVEL / 2.0))
_CELLS = (
    Cell(severity=0.0, successes=100, trials=100),
    Cell(severity=1.0, successes=50, trials=100),
)


def _fit_with_known_natural_variance(*, value: float, variance: float) -> Fit:
    """A ``status=OK`` ``Fit`` whose ``absolute(0.5)`` threshold is exactly ``value`` with a
    known natural-scale (delta-method) variance ``variance = Var(value)``.

    ``target=0.5`` with ``upper=1.0, lower=0.0`` fixed gives ``z* = 0`` for a symmetric link
    (probit/logit; not cloglog, which is asymmetric around ``z=0``), so ``theta_hat = mu``
    exactly and the delta-method gradient's ``s`` component (``z*``) is zero: only
    ``Cov(mu, mu)`` contributes to ``Var(theta_hat) = Var(log(value))`` on this log axis.
    ``Var(value) = value^2 * Var(log(value))`` (module docstring), so ``Cov(mu, mu) =
    variance / value^2`` reproduces the requested natural-scale variance exactly.
    """
    var_log = variance / value**2
    return Fit(
        axis=_AXIS,
        outcome="success",
        direction="decreasing",
        model="binomial",
        link="logit",
        params={
            "mu": Parameter(value=math.log(value), fixed=False),
            "s": Parameter(value=1.0, fixed=False),
            "upper": Parameter(value=1.0, fixed=True),
            "lower": Parameter(value=0.0, fixed=True),
        },
        covariance=Covariance(names=("mu", "s"), matrix=((var_log, 0.0), (0.0, 1.0))),
        log_likelihood=-10.0,
        cells=_CELLS,
        status=Status.OK,
    )


def _threshold_with_known_natural_variance(*, value: float, variance: float) -> Threshold:
    fit = _fit_with_known_natural_variance(value=value, variance=variance)
    sigma_log = math.sqrt(variance) / value
    lo = value * math.exp(-_Z * sigma_log)
    hi = value * math.exp(_Z * sigma_log)
    return Threshold(
        axis=_AXIS,
        definition=Definition.absolute(0.5),
        direction="decreasing",
        value=value,
        lo=lo,
        hi=hi,
        level=_LEVEL,
        interval_method="delta",
        censoring=Censoring.NONE,
        status=Status.OK,
        dependence="independent",
        fit=fit,
    )


def _expected_fieller(
    *, theta_a: float, va: float, theta_b: float, vb: float, z: float
) -> tuple[IntervalShape, float | None, float | None]:
    """The classic Fieller quadratic (module docstring), solved independently of
    ``ratio_interval()`` -- this file's own reference computation, not a call into robustmargin.
    """
    a_coef = theta_b**2 - z**2 * vb
    b_coef = theta_a * theta_b
    k_coef = theta_a**2 - z**2 * va
    disc = b_coef**2 - a_coef * k_coef
    if a_coef > 0.0:
        assert disc > 0.0, "g < 1 must always yield two real roots"
        sq = math.sqrt(disc)
        lo, hi = sorted([(b_coef - sq) / a_coef, (b_coef + sq) / a_coef])
        return IntervalShape.BOUNDED, lo, hi
    if disc > 0.0:
        sq = math.sqrt(disc)
        lo, hi = sorted([(b_coef - sq) / a_coef, (b_coef + sq) / a_coef])
        return IntervalShape.EXCLUSIVE, lo, hi
    return IntervalShape.UNBOUNDED, None, None


def _expected_shape_and_bounds(
    *, theta_a: float, va: float, theta_b: float, vb: float, z: float
) -> tuple[IntervalShape, float | None, float | None]:
    """The *reported* shape, per the full branch table in this module's docstring -- the Fieller set
    intersected with the positive parameter space, unlike ``_expected_fieller`` above, which returns
    the raw, unclipped classic-quadratic answer. Solved independently of ``ratio_interval()``.
    """
    a_coef = theta_b**2 - z**2 * vb
    b_coef = theta_a * theta_b
    k_coef = theta_a**2 - z**2 * va

    if a_coef > 0.0:
        disc = b_coef**2 - a_coef * k_coef
        sq = math.sqrt(disc)
        lo, hi = sorted([(b_coef - sq) / a_coef, (b_coef + sq) / a_coef])
        return IntervalShape.BOUNDED, max(lo, 0.0), hi  # negative lower root clipped to 0

    if a_coef == 0.0:
        root = k_coef / (2.0 * b_coef)
        if k_coef > 0.0:
            return IntervalShape.HALF_OPEN, root, None  # a genuine positive lower bound
        return IntervalShape.UNBOUNDED, None, None  # root at or below zero

    # a_coef < 0.0
    disc = b_coef**2 - a_coef * k_coef
    if disc <= 0.0:
        return IntervalShape.UNBOUNDED, None, None  # no real roots
    sq = math.sqrt(disc)
    r1, r2 = sorted([(b_coef - sq) / a_coef, (b_coef + sq) / a_coef])
    if k_coef <= 0.0:
        return IntervalShape.UNBOUNDED, None, None  # both roots negative
    return IntervalShape.HALF_OPEN, r2, None  # opposite-signed roots: only the positive one


class TestBoundedShapeFromConstructedInputs:
    """``g = z^2 * Vb / theta_b^2 < 1``: theta_a=0.5 (Va=0.05), theta_b=1.0 (Vb=0.1) gives
    ``g ~= 0.384``."""

    def test_shape_is_bounded_and_matches_the_closed_form(self) -> None:
        t_a = _threshold_with_known_natural_variance(value=0.5, variance=0.05)
        t_b = _threshold_with_known_natural_variance(value=1.0, variance=0.1)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)

        expected_shape, expected_lo, expected_hi = _expected_fieller(
            theta_a=0.5, va=0.05, theta_b=1.0, vb=0.1, z=_Z
        )
        assert expected_shape is IntervalShape.BOUNDED
        assert result.shape is IntervalShape.BOUNDED
        assert result.estimate == pytest.approx(0.5, rel=1e-9)
        assert result.lo == pytest.approx(expected_lo, rel=1e-6)
        assert result.hi == pytest.approx(expected_hi, rel=1e-6)
        assert result.lo <= result.estimate <= result.hi


class TestBoundedShapeFromARealFit:
    """The same ``g < 1`` shape, from an actual ``fit_dose_response`` call rather than a
    constructed ``Fit`` -- showing ``BOUNDED`` is the ordinary case for a reasonably informative
    real design, not just a property of hand-picked numbers."""

    def _fit(self, *, successes: list[int]) -> Fit:
        obs = Observations.from_counts(
            _AXIS,
            severity=[0.0, 0.5, 1.0, 2.0, 4.0],
            successes=successes,
            trials=[200, 200, 200, 200, 200],
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        assert fit.status is Status.OK
        return fit

    def test_shape_is_bounded(self) -> None:
        fit_a = self._fit(successes=[200, 150, 100, 40, 5])
        fit_b = self._fit(successes=[200, 170, 130, 70, 15])
        t_a = threshold(
            fit_a, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
        )
        t_b = threshold(
            fit_b, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
        )
        assert t_a.censoring is Censoring.NONE and t_b.censoring is Censoring.NONE

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)

        assert result.shape is IntervalShape.BOUNDED
        assert result.lo is not None and result.hi is not None
        assert result.lo <= result.estimate <= result.hi


class TestHalfOpenAndUnboundedFromRealFits:
    """``g >= 1`` reached through real ``fit_dose_response`` calls (module docstring), found by
    search rather than assumed unreachable: two severity levels (``1`` and ``4``), ``n=20``, and a
    near-flat success pattern (``11/20`` then ``9/20``) barely constrain the fitted slope at all,
    giving a denominator threshold with a large log-scale standard error (``sigma_log ~= 1.10``,
    ``g ~= 4.6`` at ``level=0.95`` -- well past the ``g=1`` boundary). Pairing that same
    denominator with a *tightly*-determined numerator (a four-level, ``n=200`` design with a clear
    gradient) gives a positive discriminant and ``K > 0`` (``HALF_OPEN``); pairing it with an
    equally poorly-determined numerator (severities ``1`` and ``2``, the same ``11/20`` then
    ``9/20`` pattern, ``sigma_log ~= 0.55``, ``g ~= 1.15``) makes the discriminant non-positive
    instead (``UNBOUNDED``, since there are no real roots to intersect at all) -- exactly the
    discriminant-sign distinction the constructed cases above make with hand-picked numbers, now
    from data. Every fit here is ``Status.OK`` with ``Censoring.NONE``
    (confirmed empirically, mirroring ``tests/unit/test_threshold_status_coverage.py``'s own
    convention for search-found fixtures): the profile interval comes back open on both
    sides for the two poorly-determined fits because the deviance does not reach the chi-square
    target before either search limit (no ``Censoring`` member exists for that, so it is reported
    as ``NONE`` with ``lo=hi=None`` -- ``intervals.py``'s own documented "open on both sides"
    outcome), never because an inner fit failed. That does not affect ``ratio_interval()``, since
    that reads each fit's own covariance directly rather than the input ``Threshold``'s
    ``lo``/``hi``.
    """

    def _poorly_determined_denominator(self) -> Threshold:
        obs = Observations.from_counts(
            _AXIS,
            severity=[1.0, 4.0],
            successes=[11, 9],
            trials=[20, 20],
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        assert fit.status is Status.OK
        t = threshold(
            fit, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
        )
        assert t.censoring is Censoring.NONE
        assert t.lo is None and t.hi is None
        assert not any("inner fit failed" in w for w in t.warnings)
        return t

    def _tightly_determined_numerator(self) -> Threshold:
        obs = Observations.from_counts(
            _AXIS,
            severity=[0.5, 1.0, 2.0, 4.0],
            successes=[190, 150, 60, 5],
            trials=[200, 200, 200, 200],
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        assert fit.status is Status.OK
        t = threshold(
            fit, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
        )
        assert t.censoring is Censoring.NONE
        return t

    def _loosely_determined_numerator(self) -> Threshold:
        obs = Observations.from_counts(
            _AXIS,
            severity=[1.0, 2.0],
            successes=[11, 9],
            trials=[20, 20],
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        assert fit.status is Status.OK
        t = threshold(
            fit, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
        )
        assert t.censoring is Censoring.NONE
        assert t.lo is None and t.hi is None
        assert not any("inner fit failed" in w for w in t.warnings)
        return t

    def test_shape_is_half_open_with_a_tight_numerator(self) -> None:
        t_a = self._tightly_determined_numerator()
        t_b = self._poorly_determined_denominator()

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)

        assert result.shape is IntervalShape.HALF_OPEN
        assert result.lo is not None and result.lo > 0.0
        assert result.hi is None
        assert result.estimate is not None and result.lo <= result.estimate

    def test_shape_is_unbounded_with_an_equally_loose_numerator(self) -> None:
        t_a = self._loosely_determined_numerator()
        t_b = self._poorly_determined_denominator()

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)

        assert result.shape is IntervalShape.UNBOUNDED
        assert result.lo is None and result.hi is None
        assert result.estimate is not None


class TestGEqualsOneBoundary:
    """At ``A = theta_b^2 - z^2*Vb == 0`` exactly, the quadratic degenerates to the linear
    ``-2*B*rho + K = 0``, root ``rho = K/(2*B)``. **``K > 0`` is ``HALF_OPEN`` with
    ``lo = K/(2*B)``** (a genuine positive lower bound must not be discarded as ``UNBOUNDED``).
    ``K <= 0``: the root sits at or below zero, so intersected with the positive axis the set is
    all of ``(0, inf)`` -- ``UNBOUNDED``.

    **Hitting ``A == 0`` exactly in floating point is constructible, with no tolerance widening
    needed** -- verified empirically for the values used here, not merely
    assumed. Choosing ``theta_b = 1.0`` makes ``theta_b**2 == 1.0`` and
    ``value_b**2 * var_mu_b == var_mu_b`` exact regardless of how squaring is implemented
    (multiplying by exactly ``1.0`` introduces no rounding), which removes one whole source of
    floating-point noise from the construction. Setting ``var_mu_b = 1.0 / z**2`` (a single
    division) and checking ``1.0 - z**2 * var_mu_b`` reproduces ``0.0`` bit-for-bit -- confirmed
    directly in Python for ``level`` in ``{0.5, 0.8, 0.9, 0.95}`` and several ``theta_b`` values,
    and true regardless of whether the squaring is written ``z**2`` or ``z*z`` (both give the
    identical ``float``). This is not a universal IEEE-754 guarantee (``a * (1/a) == 1`` can fail
    for adversarial ``a``), only an empirical confirmation for the natural, un-adversarial values
    a real ``level``/``Vb`` combination produces -- stated plainly, rather than
    silently building in a tolerance this construction did not turn out to need. Both sub-tests
    below reuse the identical ``theta_b``/``var_mu_b`` pair, varying only the numerator, so both
    hit ``A == 0`` bit-for-bit.
    """

    def test_k_greater_than_zero_is_half_open_at_the_exact_g_equals_one_boundary(self) -> None:
        theta_b = 1.0
        var_mu_b = 1.0 / (_Z**2)
        assert theta_b**2 - _Z**2 * var_mu_b == 0.0

        theta_a, va = 0.5, 0.05  # K = theta_a^2 - z^2*va ~= 0.058 > 0
        expected_shape, expected_lo, expected_hi = _expected_shape_and_bounds(
            theta_a=theta_a, va=va, theta_b=theta_b, vb=var_mu_b, z=_Z
        )
        assert expected_shape is IntervalShape.HALF_OPEN
        assert expected_lo is not None and expected_lo > 0.0
        assert expected_hi is None

        t_b = _threshold_with_known_natural_variance(value=theta_b, variance=var_mu_b)
        t_a = _threshold_with_known_natural_variance(value=theta_a, variance=va)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)

        assert result.shape is IntervalShape.HALF_OPEN
        assert result.lo == pytest.approx(expected_lo, rel=1e-6)
        assert result.hi is None
        assert result.estimate is not None and result.lo <= result.estimate

    def test_k_at_most_zero_stays_unbounded_at_the_exact_g_equals_one_boundary(self) -> None:
        theta_b = 1.0
        var_mu_b = 1.0 / (_Z**2)
        assert theta_b**2 - _Z**2 * var_mu_b == 0.0

        theta_a, va = 0.5, 0.1  # K = theta_a^2 - z^2*va ~= -0.134 <= 0
        expected_shape, _expected_lo, _expected_hi = _expected_shape_and_bounds(
            theta_a=theta_a, va=va, theta_b=theta_b, vb=var_mu_b, z=_Z
        )
        assert expected_shape is IntervalShape.UNBOUNDED

        t_b = _threshold_with_known_natural_variance(value=theta_b, variance=var_mu_b)
        t_a = _threshold_with_known_natural_variance(value=theta_a, variance=va)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)

        assert result.shape is IntervalShape.UNBOUNDED
        assert result.lo is None
        assert result.hi is None
        assert result.estimate is not None
        assert len(result.warnings) >= 1
        assert any(len(warning.strip()) > 20 for warning in result.warnings)


class TestHalfOpenFromConstructedInputs:
    """``g >= 1`` with a positive discriminant and ``K > 0``: theta_a=0.5 (Va=0.05, tight
    numerator), theta_b=1.0 (Vb=0.5, very uncertain denominator) gives ``g ~= 1.92``, a positive
    discriminant, and ``K = theta_a^2 - z^2*Va ~= 0.058 > 0``.

    **Reporting this as ``EXCLUSIVE`` would be wrong:** it would give ``lo=-1.141, hi=0.055`` --
    a negative lower bound on a severity ratio, reported with no warning. The property test
    ``tests/property/test_ratio_interval_properties.py``'s ``TestFiellerLowerAndUpperNeverNegative``
    pins that a bound is never negative, and this construction shows the case occupies an open
    region of the parameter space (under independence): intersected with the positive parameter
    space, ``(-inf, -1.141] union [0.055, inf)`` is exactly ``[0.055, inf)`` -- ``HALF_OPEN`` with
    ``lo = 0.055`` (the positive root), not ``EXCLUSIVE``.
    ``TestBothRootsNegativeIsUnbounded`` below is the sibling case (``K <= 0``) that stays
    ``UNBOUNDED``.
    """

    def test_shape_is_half_open_and_matches_the_closed_form(self) -> None:
        t_a = _threshold_with_known_natural_variance(value=0.5, variance=0.05)
        t_b = _threshold_with_known_natural_variance(value=1.0, variance=0.5)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)

        # The raw, unclipped classic-quadratic answer is still EXCLUSIVE with lo < 0 < hi --
        # confirming this construction hits the case that is reclassified on intersection, not
        # some other branch -- before checking what the *reported*, intersected shape must be.
        raw_shape, raw_lo, raw_hi = _expected_fieller(
            theta_a=0.5, va=0.05, theta_b=1.0, vb=0.5, z=_Z
        )
        assert raw_shape is IntervalShape.EXCLUSIVE
        assert raw_lo is not None and raw_lo < 0.0 < raw_hi

        expected_shape, expected_lo, expected_hi = _expected_shape_and_bounds(
            theta_a=0.5, va=0.05, theta_b=1.0, vb=0.5, z=_Z
        )
        assert expected_shape is IntervalShape.HALF_OPEN
        assert expected_lo == pytest.approx(raw_hi, rel=1e-9)  # the positive root, unchanged
        assert expected_hi is None

        assert result.shape is IntervalShape.HALF_OPEN
        assert result.lo == pytest.approx(expected_lo, rel=1e-6)
        assert result.hi is None
        assert result.estimate is not None and result.lo <= result.estimate


class TestUnboundedShapeFromConstructedInputs:
    """``g >= 1`` with a non-positive discriminant: theta_a=0.5 with a *large* Va=0.5 (the
    numerator is now uncertain enough, relative to theta_a^2, that the confidence set covers the
    whole line) alongside the same theta_b=1.0, Vb=0.5 denominator as the ``HALF_OPEN`` case
    above -- the discriminant's sign is what separates the two, not ``g`` itself. This branch
    (no real roots at all) is independent of the rules for what to do when real roots exist."""

    def test_shape_is_unbounded_and_matches_the_closed_form(self) -> None:
        t_a = _threshold_with_known_natural_variance(value=0.5, variance=0.5)
        t_b = _threshold_with_known_natural_variance(value=1.0, variance=0.5)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)

        expected_shape, expected_lo, expected_hi = _expected_fieller(
            theta_a=0.5, va=0.5, theta_b=1.0, vb=0.5, z=_Z
        )
        assert expected_shape is IntervalShape.UNBOUNDED
        assert result.shape is IntervalShape.UNBOUNDED
        assert result.lo is None
        assert result.hi is None
        # UNBOUNDED still carries a point estimate (Ratio's own invariant: "it carries
        # estimate"), never clipped into looking like a degenerate bounded interval.
        assert result.estimate == pytest.approx(0.5, rel=1e-9)


class TestNegativeLowerRootIsClippedToZero:
    """Whenever the numerator's own natural-scale interval covers zero
    (``K = theta_a^2 - z^2*Va <= 0``), the Fieller lower root falls below zero -- not a possible
    ratio on either a log or a linear axis. Rather than report a negative ``lo`` (correct
    arithmetic, meaningless as a severity ratio) or discard the information as ``UNBOUNDED``
    (wrong direction -- the set genuinely is bounded above), the reported interval is
    intersected with the positive parameter space: ``lo = 0.0``, ``shape`` stays
    ``BOUNDED``, ``hi`` is untouched, and the unclipped root is named in ``warnings``.

    ``K <= 0`` is a condition on the *numerator* alone (``theta_a``, ``Va``), independent of
    ``g``/``A`` (which is about the denominator) -- so this is reachable whether or not the
    overall shape would otherwise have been comfortably ``BOUNDED``, which is exactly why it is
    easy to construct: theta_a=0.5, Va=0.1 already gives ``K = 0.25 - z^2*0.1 ~= -0.134 < 0``,
    with a well-determined denominator (theta_b=1.0, Vb=0.01, so ``A > 0`` keeps the shape
    ``BOUNDED``) keeping the rest of the picture ordinary.
    """

    def test_negative_lower_root_reported_as_zero_with_hi_unchanged_and_a_warning(self) -> None:
        t_a = _threshold_with_known_natural_variance(value=0.5, variance=0.1)
        t_b = _threshold_with_known_natural_variance(value=1.0, variance=0.01)

        expected_shape, expected_lo, expected_hi = _expected_fieller(
            theta_a=0.5, va=0.1, theta_b=1.0, vb=0.01, z=_Z
        )
        assert expected_shape is IntervalShape.BOUNDED
        assert expected_lo is not None and expected_lo < 0.0, "the construction must hit K <= 0"

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)

        assert result.shape is IntervalShape.BOUNDED
        assert result.lo == 0.0
        assert result.hi == pytest.approx(expected_hi, rel=1e-6)
        assert result.estimate is not None and result.lo <= result.estimate <= result.hi
        assert len(result.warnings) >= 1
        assert any(len(warning.strip()) > 20 for warning in result.warnings)

    def test_negative_lower_root_from_a_real_fit_is_also_clipped(self) -> None:
        """The same ``K <= 0`` condition, from real ``fit_dose_response`` calls: the same
        poorly-determined two-level, ``n=20`` design ``TestHalfOpenAndUnboundedFromRealFits``
        uses (there, as the *denominator*, to reach ``g >= 1``; here severities ``1`` and ``4``,
        ``sigma_log ~= 1.10``) reaches ``K <= 0`` instead when used as the *numerator* against a
        tightly-determined denominator, since ``K`` depends on the numerator's own uncertainty,
        not the denominator's. The unclipped lower root is about ``-1.6``."""
        numerator_obs = Observations.from_counts(
            _AXIS,
            severity=[1.0, 4.0],
            successes=[11, 9],
            trials=[20, 20],
            outcome="success",
            direction="decreasing",
        )
        denominator_obs = Observations.from_counts(
            _AXIS,
            severity=[0.5, 1.0, 2.0, 4.0],
            successes=[190, 150, 60, 5],
            trials=[200, 200, 200, 200],
            outcome="success",
            direction="decreasing",
        )
        fit_a = fit_dose_response(
            numerator_obs, model="binomial", link="probit", upper=1.0, lower=0.0
        )
        fit_b = fit_dose_response(
            denominator_obs, model="binomial", link="probit", upper=1.0, lower=0.0
        )
        assert fit_a.status is Status.OK and fit_b.status is Status.OK
        t_a = threshold(
            fit_a, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
        )
        t_b = threshold(
            fit_b, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
        )
        assert t_a.censoring is Censoring.NONE and t_b.censoring is Censoring.NONE

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)

        assert result.shape is IntervalShape.BOUNDED
        assert result.lo == 0.0
        assert result.hi is not None and result.hi > 0.0
        assert len(result.warnings) >= 1
        assert any(len(warning.strip()) > 20 for warning in result.warnings)


class TestOpenBothSidesInputWarns:
    """When an input ``Threshold`` is ``(OK, NONE)`` but its own
    profile interval did not close on *either* side (``lo is None and hi is None`` -- v1's
    "open on both sides" outcome, ``intervals.py``'s own documented case, since no ``Censoring``
    member exists for it), ``ratio_interval`` still has a covariance to compute a Fieller/
    log-delta variance from, but the caller should be told the input it started from was itself
    unbounded on both ends -- so the ``Ratio`` carries a ``warnings`` entry saying so.

    Built from a real fit whose profile does not cross on either side before the search limits
    (confirmed empirically: severities ``1`` and ``4``, ``11/20`` then ``9/20``, the same
    poorly-determined two-level ``n=20`` design used elsewhere in this file): not
    constructed, since this exact input shape already turned up naturally while building the
    other real-fit tests in this file, so building it again by hand would be redundant.
    """

    def test_open_both_sides_input_produces_a_warning(self) -> None:
        obs = Observations.from_counts(
            _AXIS,
            severity=[1.0, 4.0],
            successes=[11, 9],
            trials=[20, 20],
            outcome="success",
            direction="decreasing",
        )
        fit_a = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        assert fit_a.status is Status.OK
        t_a = threshold(
            fit_a, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
        )
        assert t_a.status is Status.OK
        assert t_a.censoring is Censoring.NONE
        assert t_a.lo is None and t_a.hi is None, "this input must itself be open on both sides"
        assert not any("inner fit failed" in w for w in t_a.warnings)

        t_b = _threshold_with_known_natural_variance(value=1.0, variance=0.01)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)

        assert len(result.warnings) >= 1
        assert any(len(warning.strip()) > 20 for warning in result.warnings)


class TestBothRootsNegativeIsUnbounded:
    """The ``EXCLUSIVE``-branch gap in the clipping rule. When ``A < 0``
    (the ``EXCLUSIVE``-or-``UNBOUNDED`` regime) *and* ``K <= 0``, the two
    roots share a sign (product ``= K/A >= 0``) and that sign is negative (sum ``= 2*B/A < 0``,
    ``B = theta_a*theta_b > 0`` under independence). Both rays of the raw ``EXCLUSIVE`` set then
    lie entirely below zero, so intersected with the positive parameter space the confidence set
    is all of ``(0, inf)`` -- reported as ``UNBOUNDED`` (exactly, not as a conservative
    approximation), with ``lo = hi = None`` and the unclipped roots named in ``warnings``.

    The motivating case: ``theta_a = theta_b = 1.0``,
    ``z^2*Va = z^2*Vb = 1.5`` (so ``A = K = -0.5``, ``disc = 0.75 > 0`` -- genuinely two real
    roots, not a degenerate tangency), giving unclipped roots
    ``(-3.732050807568877, -0.2679491924311227)``, both negative.
    """

    def test_both_roots_negative_reports_unbounded_with_a_warning(self) -> None:
        variance = 1.5 / _Z**2  # so z^2 * variance == 1.5 exactly, matching the case above
        t_a = _threshold_with_known_natural_variance(value=1.0, variance=variance)
        t_b = _threshold_with_known_natural_variance(value=1.0, variance=variance)

        expected_shape, expected_lo, expected_hi = _expected_fieller(
            theta_a=1.0, va=variance, theta_b=1.0, vb=variance, z=_Z
        )
        # _expected_fieller reports the *raw*, unclipped shape (module docstring): confirm this
        # construction really does hit A < 0, K <= 0 before checking what should happen once
        # clipped.
        assert expected_shape is IntervalShape.EXCLUSIVE
        assert expected_lo is not None and expected_lo < 0.0
        assert expected_hi is not None and expected_hi < 0.0

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)

        assert result.shape is IntervalShape.UNBOUNDED
        assert result.lo is None
        assert result.hi is None
        assert result.estimate == pytest.approx(1.0, rel=1e-9)
        assert len(result.warnings) >= 1
        assert any(len(warning.strip()) > 20 for warning in result.warnings)


class TestExclusiveRemainsAValidRatioShapeForPairedDependence:
    """``EXCLUSIVE`` is unreachable *through* ``ratio_interval()`` in this release --
    it needs same-signed roots with ``A < 0``, which needs ``B <= 0``, which needs ``C > 0``,
    which needs ``dependence="paired"`` (not implemented). The member stays in ``IntervalShape``
    and ``Ratio``'s own invariants for it stay enforced, because the type must already be valid
    when paired dependence starts producing it -- so this constructs a ``Ratio``
    directly (not through ``ratio_interval``, which cannot reach this shape) and checks both that
    a well-formed ``EXCLUSIVE`` ``Ratio`` is still accepted and that a malformed one is still
    rejected, exactly as ``tests/unit/test_results.py``'s ``TestRatioShapeExclusive`` already
    establishes -- restated here, next to the rest of this file's shape coverage, specifically
    to record that unreachability is deliberate, not an oversight.
    """

    def test_a_well_formed_exclusive_ratio_is_still_accepted(self) -> None:
        t_a = _threshold_with_known_natural_variance(value=2.0, variance=0.01)
        t_b = _threshold_with_known_natural_variance(value=1.0, variance=0.01)

        ratio = Ratio(
            threshold_a=t_a,
            threshold_b=t_b,
            estimate=2.0,
            lo=-1.0,
            hi=1.5,
            shape=IntervalShape.EXCLUSIVE,
            method="fieller",
            dependence="independent",
            level=_LEVEL,
            censoring=Censoring.NONE,
            status=Status.OK,
        )

        assert ratio.shape is IntervalShape.EXCLUSIVE
        assert ratio.lo == -1.0 and ratio.hi == 1.5
        assert ratio.estimate <= ratio.lo or ratio.estimate >= ratio.hi

    def test_an_exclusive_ratio_with_the_estimate_inside_the_gap_still_raises(self) -> None:
        t_a = _threshold_with_known_natural_variance(value=1.0, variance=0.01)
        t_b = _threshold_with_known_natural_variance(value=1.0, variance=0.01)

        with pytest.raises(ValueError):
            Ratio(
                threshold_a=t_a,
                threshold_b=t_b,
                estimate=1.0,  # strictly inside (lo, hi) = (-1.0, 1.5): not a legal EXCLUSIVE
                lo=-1.0,
                hi=1.5,
                shape=IntervalShape.EXCLUSIVE,
                method="fieller",
                dependence="independent",
                level=_LEVEL,
                censoring=Censoring.NONE,
                status=Status.OK,
            )


class TestEmptyConfidenceSetRaises:
    """Regression: ``A == 0`` degenerates the Fieller inequality to the linear
    ``-2*B*rho + K <= 0``; when ``B`` is *also* ``0`` (only possible under independence via
    ``theta_a == 0``, module docstring) that degenerates further to ``K <= 0``, so ``K > 0``
    describes an **empty** confidence set -- not a half-line, and reporting ``UNBOUNDED`` (the whole
    positive axis) would be the least informative possible answer to what is actually a
    contradictory input. It raises ``ValueError`` naming the empty set, rather than reporting a
    shape at all.

    **Unreachable through the public API in this release**, confirmed by the module docstring's
    argument, not merely asserted: this test calls ``robustmargin.ratio``'s private
    ``_fieller_shape`` directly with ``theta_a=0.0`` and a deliberately-impossible **negative**
    ``va`` -- no real delta-method variance is ever negative, and a genuine non-negative ``va`` with
    ``theta_a == 0`` forces ``K = -z^2*va <= 0``, never ``> 0``. ``dependence="paired"`` (nonzero
    covariance ``C``) can reach ``B == 0`` without ``theta_a == 0``, and that path may reach
    ``K > 0`` genuinely -- this branch is guarded, not dead, but it is guarded against a case this
    release itself cannot construct.
    """

    def test_a_zero_b_zero_k_positive_raises_rather_than_reporting_unbounded(self) -> None:
        z = _Z
        theta_b = 1.0
        vb = 1.0 / (z**2)  # the same bit-exact A == 0 trick used throughout this file
        assert theta_b**2 - z**2 * vb == 0.0

        with pytest.raises(ValueError) as exc_info:
            _fieller_shape(theta_a=0.0, va=-1.0, theta_b=theta_b, vb=vb, z=z)

        message = str(exc_info.value)
        assert "empty" in message.lower()
        assert len(message.strip()) > 20
