"""Property tests for ``ratio_interval()`` (hypothesis).

Six properties, on top of the shape/guard/censoring/reference tests elsewhere:

- **Fieller reduces to the classical Wald/delta-method interval as the denominator's own
  uncertainty vanishes (``g -> 0``).** This is a textbook asymptotic fact about the Fieller
  quadratic itself (worked out in this module's docstring below), not a robustmargin-specific
  choice, and it is what "Fieller for the ratio" is supposed to degenerate to when the
  denominator is well-determined.
- **Reciprocal consistency:** ``ratio_interval(b, a)`` is the reciprocal of ``ratio_interval(a,
  b)`` for the ``BOUNDED`` case (both directions positive).
- **Scale invariance:** multiplying *both* thresholds' severities by the same constant leaves
  the ratio and its interval unchanged -- the ratio is dimensionless, so this is stronger than
  (and different from) the single-threshold "``x -> c*x`` scales ``value`` by ``c``" property in
  ``tests/property/test_threshold_properties.py``.
- **Interval containment:** the point estimate lies in the confidence set described by
  ``shape``, for whichever of the three shapes a given draw produces. The three shapes
  individually, by name, are pinned as fixed examples in
  ``tests/unit/test_ratio_interval_shape.py``; this is the hypothesis-driven generalisation
  across many parameter draws, checking the shape-appropriate condition for whatever comes up.
- **``log_delta`` variance algebra, asserted directly, not just the endpoints:**
  ``Var(log ratio) = Va + Vb`` (``C = 0`` under independence), recovered from the reported
  interval's own log-scale half-width and compared to the two inputs' known log-scale variances.
- **Monotonicity in ``level``:** a 99% interval contains the 95% interval (``BOUNDED`` case).

**Every case here constructs ``Fit``/``Threshold`` objects directly**, the same way
``tests/unit/test_ratio_interval_shape.py`` does, rather than fitting real counts: hypothesis
draws ``(value, variance)`` pairs and this file builds internally-consistent objects at whatever
precision hypothesis wants, using the delta method's exact chain-rule identity
``Var(value) = value^2 * Var(log(value))`` (see that file's docstring) rather than hoping a
randomly generated counts table happens to converge to a chosen variance.

**Worked derivation for the first property (Fieller -> delta as ``Vb -> 0``).** With
``C = 0``, the Fieller quadratic's coefficients are ``A = theta_b^2 - z^2*Vb``,
``B = theta_a*theta_b``, ``K = theta_a^2 - z^2*Va``. As ``Vb -> 0``: ``A -> theta_b^2``,
``disc = B^2 - A*K -> theta_b^2 * z^2 * Va`` (substitute ``A = theta_b^2`` and expand), so the
roots ``(B +/- sqrt(disc)) / A -> theta_a/theta_b +/- z*sqrt(Va)/theta_b`` -- exactly the
classical Wald interval for a ratio with a *known* (non-random) denominator,
``Var(ratio) = Va / theta_b^2``. This is not robustmargin's own ``log_delta`` (which is on the log
scale); it is what "Fieller" itself must converge to as the source of Fieller's own extra
complexity (denominator uncertainty) disappears.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from robustmargin import (
    Axis,
    Cell,
    Censoring,
    Definition,
    Fit,
    IntervalShape,
    Status,
    Threshold,
    ratio_interval,
)
from robustmargin.models import Covariance, Parameter

_MAX_EXAMPLES = 100
_AXIS = Axis(name="severity", unit="unit", scale="log")
_CELLS = (
    Cell(severity=0.0, successes=100, trials=100),
    Cell(severity=1.0, successes=50, trials=100),
)
_LEVEL = 0.95
_Z = float(norm.ppf(0.5 + _LEVEL / 2.0))

_VALUE = st.floats(min_value=0.05, max_value=20.0, allow_nan=False, allow_infinity=False)
# Log-scale standard errors: small enough that g = z^2 * sigma_log_b^2 stays comfortably below 1
# (BOUNDED) for the properties that need an ordinary finite interval on both sides.
_SMALL_LOG_SE = st.floats(min_value=0.01, max_value=0.2, allow_nan=False, allow_infinity=False)
# A wider range for the containment property, which wants a genuine mix of shapes: at
# level=0.95, g = z^2*sigma_log_b^2 >= 1 once sigma_log_b >~ 0.51.
_WIDE_LOG_SE = st.floats(min_value=0.01, max_value=1.5, allow_nan=False, allow_infinity=False)
_RESCALE = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)


def _fit_with_known_log_se(*, value: float, sigma_log: float) -> Fit:
    """Mirrors ``tests/unit/test_ratio_interval_shape.py``'s ``_fit_with_known_natural_variance``,
    parametrised directly by the log-scale standard error instead of the natural-scale variance
    (more convenient for the ``log_delta`` variance-algebra property, which is about the log
    scale directly). ``absolute(0.5)`` with fixed ``upper=1, lower=0`` gives ``z*=0`` for a
    symmetric link, so ``Cov(mu, mu) = sigma_log^2`` reproduces ``Var(log(value)) = sigma_log^2``
    exactly, with no contribution from ``s`` (whose delta-method gradient component is ``z*=0``).
    """
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
        covariance=Covariance(names=("mu", "s"), matrix=((sigma_log**2, 0.0), (0.0, 1.0))),
        log_likelihood=-10.0,
        cells=_CELLS,
        status=Status.OK,
    )


def _threshold_with_known_log_se(
    *, value: float, sigma_log: float, level: float = _LEVEL
) -> Threshold:
    fit = _fit_with_known_log_se(value=value, sigma_log=sigma_log)
    z = float(norm.ppf(0.5 + level / 2.0))
    lo = value * math.exp(-z * sigma_log)
    hi = value * math.exp(z * sigma_log)
    return Threshold(
        axis=_AXIS,
        definition=Definition.absolute(0.5),
        direction="decreasing",
        value=value,
        lo=lo,
        hi=hi,
        level=level,
        interval_method="delta",
        censoring=Censoring.NONE,
        status=Status.OK,
        dependence="independent",
        fit=fit,
    )


class TestFiellerReducesToDeltaAsGApproachesZero:
    @given(theta_a=_VALUE, theta_b=_VALUE, sigma_log_a=_SMALL_LOG_SE)
    @settings(max_examples=_MAX_EXAMPLES, deadline=None)
    def test_matches_the_classical_wald_ratio_interval(
        self, theta_a: float, theta_b: float, sigma_log_a: float
    ) -> None:
        # A denominator sigma_log this small (g = z^2*sigma_log_b^2 ~ 3.84e-8) is close enough
        # to Vb=0 that the module docstring's limiting formula should already hold tightly.
        tiny_sigma_log_b = 1.0e-4
        t_a = _threshold_with_known_log_se(value=theta_a, sigma_log=sigma_log_a)
        t_b = _threshold_with_known_log_se(value=theta_b, sigma_log=tiny_sigma_log_b)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)
        assert result.shape is IntervalShape.BOUNDED
        assert result.lo is not None and result.hi is not None

        va = (theta_a * sigma_log_a) ** 2  # natural-scale variance, module docstring's identity
        expected_estimate = theta_a / theta_b
        expected_half_width = _Z * math.sqrt(va) / theta_b

        assert result.estimate == pytest.approx(expected_estimate, rel=1e-6)
        assert result.lo == pytest.approx(expected_estimate - expected_half_width, rel=1e-3)
        assert result.hi == pytest.approx(expected_estimate + expected_half_width, rel=1e-3)


def _fieller_roots(
    *, theta_a: float, va: float, theta_b: float, vb: float, z: float
) -> tuple[float, float, float]:
    """The classic Fieller quadratic's coefficient ``A`` and its two *unclipped* roots
    (mirrors ``tests/unit/test_ratio_interval_shape.py``'s ``_expected_fieller``, without the
    shape dispatch -- this file's uses of it stay in the ``A > 0`` regime, so only the roots are
    needed, before the negative-root clipping)."""
    a_coef = theta_b**2 - z**2 * vb
    b_coef = theta_a * theta_b
    k_coef = theta_a**2 - z**2 * va
    disc = b_coef**2 - a_coef * k_coef
    sq = math.sqrt(disc)
    r1, r2 = sorted([(b_coef - sq) / a_coef, (b_coef + sq) / a_coef])
    return a_coef, r1, r2


class TestReciprocalConsistency:
    """This property covers the regime **where the negative-root clip does not apply in either
    direction**:
    both ``K_a = theta_a^2 - z^2*Va`` and ``K_b = theta_b^2 - z^2*Vb`` stay positive, which (given
    ``_SMALL_LOG_SE``'s range, at most ``sigma_log = 0.2``) is guaranteed here since
    ``K = theta^2 * (1 - z^2*sigma_log^2)`` and ``z^2 * 0.2^2 ~= 0.154 < 1`` always. Stated as an
    explicit ``assume`` on the natural-scale ``K`` values, computed independently of
    ``ratio_interval``'s own output, rather than ``assume(forward.lo > 0.0)``, which would check
    a *symptom* of the same condition after the fact.
    ``TestReciprocalConsistencyUnderNegativeRootClipping`` below covers the complementary case.
    """

    @given(
        theta_a=_VALUE,
        theta_b=_VALUE,
        sigma_log_a=_SMALL_LOG_SE,
        sigma_log_b=_SMALL_LOG_SE,
    )
    @settings(max_examples=_MAX_EXAMPLES, deadline=None)
    def test_swapping_the_thresholds_gives_the_reciprocal_interval(
        self, theta_a: float, theta_b: float, sigma_log_a: float, sigma_log_b: float
    ) -> None:
        k_a = theta_a**2 - _Z**2 * (theta_a * sigma_log_a) ** 2
        k_b = theta_b**2 - _Z**2 * (theta_b * sigma_log_b) ** 2
        assume(k_a > 0.0 and k_b > 0.0)  # this test's whole scope: neither direction is clipped

        t_a = _threshold_with_known_log_se(value=theta_a, sigma_log=sigma_log_a)
        t_b = _threshold_with_known_log_se(value=theta_b, sigma_log=sigma_log_b)

        forward = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)
        backward = ratio_interval(
            t_b, t_a, method="fieller", dependence="independent", level=_LEVEL
        )
        assume(forward.shape is IntervalShape.BOUNDED)
        assume(backward.shape is IntervalShape.BOUNDED)
        assert forward.lo is not None and forward.hi is not None
        assert forward.lo > 0.0  # guaranteed by k_b > 0 above, not merely assumed post hoc

        assert backward.estimate == pytest.approx(1.0 / forward.estimate, rel=1e-9)
        assert backward.lo == pytest.approx(1.0 / forward.hi, rel=1e-6)
        assert backward.hi == pytest.approx(1.0 / forward.lo, rel=1e-6)


class TestReciprocalConsistencyUnderNegativeRootClipping:
    """The consequence of the negative-root clip, worked out and pinned to an exact form. (The
    *unclipped* algebra would make ``backward`` ``EXCLUSIVE``, but that is not what
    ``ratio_interval()`` reports: it intersects that same branch with the positive parameter
    space too.)

    When ``K_a <= 0`` clips the *forward* ratio's lower root to ``0``, the *backward*
    (reciprocal) ratio is not ``BOUNDED`` at all -- swapping ``a`` and ``b`` swaps the
    quadratic's ``A`` and ``K`` coefficients (``B`` is symmetric), so the backward computation's
    own ``A' = K_a <= 0``. Given ``theta_b``'s own ``K_b = theta_b^2 - z^2*Vb`` is kept positive
    (a well-determined denominator, ``_SMALL_LOG_SE``), the backward computation's own
    ``K' = A_forward > 0`` (forward is genuinely ``BOUNDED``, so ``A_forward > 0`` by
    definition) -- ``A' < 0`` with ``K' > 0`` is exactly the ``HALF_OPEN`` branch, not
    ``EXCLUSIVE``.

    The swapped quadratic's roots are exactly the *reciprocals of the forward computation's own
    unclipped roots* (dividing ``A*rho^2 - 2*B*rho + K = 0`` by ``rho^2`` and substituting
    ``sigma = 1/rho`` gives ``K*sigma^2 - 2*B*sigma + A = 0``, the swapped equation) -- computed
    directly here via ``_fieller_roots``, independently of ``ratio_interval``. The forward
    computation's unclipped roots are ``unclipped_lo < 0 < unclipped_hi``; the swapped roots are
    therefore ``1/unclipped_lo`` (negative) and ``1/unclipped_hi`` (positive). ``HALF_OPEN``
    keeps only the *positive* root, so **the negative one is simply dropped**, not reported as
    anything -- ``backward.lo == 1.0 / unclipped_hi == 1.0 / forward.hi`` exactly
    (``forward.hi`` was never clipped), and ``backward.hi is None``. There is no second endpoint
    to reciprocate at all, because the "other ray" is never a meaningful bound on a severity
    ratio to begin with.
    """

    @given(
        theta_a=_VALUE,
        theta_b=_VALUE,
        sigma_log_a=_WIDE_LOG_SE,
        sigma_log_b=_SMALL_LOG_SE,
    )
    @settings(max_examples=_MAX_EXAMPLES, deadline=None)
    def test_backward_ratio_is_exclusive_with_roots_reciprocal_to_the_unclipped_forward_ones(
        self, theta_a: float, theta_b: float, sigma_log_a: float, sigma_log_b: float
    ) -> None:
        va = (theta_a * sigma_log_a) ** 2
        vb = (theta_b * sigma_log_b) ** 2
        k_a = theta_a**2 - _Z**2 * va
        k_b = theta_b**2 - _Z**2 * vb
        assume(k_a < 0.0)  # this test's scope: the forward direction gets clipped
        assume(k_b > 0.0)  # denominator well-determined, so forward.shape is genuinely BOUNDED

        a_coef, unclipped_lo, unclipped_hi = _fieller_roots(
            theta_a=theta_a, va=va, theta_b=theta_b, vb=vb, z=_Z
        )
        assume(a_coef > 0.0)  # forward must actually be in the BOUNDED (A > 0) regime
        assume(unclipped_lo < 0.0)  # the condition the negative-root clip applies to

        t_a = _threshold_with_known_log_se(value=theta_a, sigma_log=sigma_log_a)
        t_b = _threshold_with_known_log_se(value=theta_b, sigma_log=sigma_log_b)

        forward = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)
        backward = ratio_interval(
            t_b, t_a, method="fieller", dependence="independent", level=_LEVEL
        )

        assert forward.shape is IntervalShape.BOUNDED
        assert forward.lo == 0.0
        assert forward.hi == pytest.approx(unclipped_hi, rel=1e-6)

        assert backward.shape is IntervalShape.HALF_OPEN
        assert backward.lo is not None and backward.hi is None
        assert backward.lo == pytest.approx(1.0 / unclipped_hi, rel=1e-6)
        assert backward.lo == pytest.approx(1.0 / forward.hi, rel=1e-6)
        assert backward.estimate is not None
        assert backward.lo <= backward.estimate


class TestScaleInvariance:
    """Rescaling *both* thresholds' severities by the same constant ``c`` leaves the ratio and
    its interval exactly unchanged: ``c`` cancels out of the classic Fieller quadratic entirely
    (every coefficient scales by ``c^2``), not merely approximately."""

    @given(
        theta_a=_VALUE,
        theta_b=_VALUE,
        sigma_log_a=_WIDE_LOG_SE,
        sigma_log_b=_WIDE_LOG_SE,
        c=_RESCALE,
    )
    @settings(max_examples=_MAX_EXAMPLES, deadline=None)
    def test_rescaling_both_severities_leaves_the_ratio_and_interval_unchanged(
        self, theta_a: float, theta_b: float, sigma_log_a: float, sigma_log_b: float, c: float
    ) -> None:
        # Rescaling severity by c leaves each threshold's *log-scale* standard error unchanged
        # (Var(log(c*x)) = Var(log(c) + log(x)) = Var(log(x)) exactly, log(c) being a constant).
        t_a = _threshold_with_known_log_se(value=theta_a, sigma_log=sigma_log_a)
        t_b = _threshold_with_known_log_se(value=theta_b, sigma_log=sigma_log_b)
        t_a_scaled = _threshold_with_known_log_se(value=c * theta_a, sigma_log=sigma_log_a)
        t_b_scaled = _threshold_with_known_log_se(value=c * theta_b, sigma_log=sigma_log_b)

        for method in ("fieller", "log_delta"):
            if method == "log_delta":
                assume(theta_a > 0.0 and theta_b > 0.0)
            original = ratio_interval(
                t_a, t_b, method=method, dependence="independent", level=_LEVEL
            )
            scaled = ratio_interval(
                t_a_scaled, t_b_scaled, method=method, dependence="independent", level=_LEVEL
            )

            assert scaled.shape is original.shape
            assert scaled.estimate == pytest.approx(original.estimate, rel=1e-9)
            if original.lo is not None:
                assert scaled.lo == pytest.approx(original.lo, rel=1e-6)
            else:
                assert scaled.lo is None
            if original.hi is not None:
                assert scaled.hi == pytest.approx(original.hi, rel=1e-6)
            else:
                assert scaled.hi is None


class TestContainmentAcrossShapes:
    """Whatever shape a draw produces, the point estimate lies in the confidence set that shape
    describes. Named-shape fixed examples live in ``tests/unit/test_ratio_interval_shape.py``;
    this generalises the same check across many random draws, including some that land on
    ``HALF_OPEN``/``UNBOUNDED`` (``sigma_log_b`` is drawn wide enough to cross ``g=1``).

    ``EXCLUSIVE`` is unreachable through ``ratio_interval()`` in this release: ``A < 0`` with
    ``K > 0`` is ``HALF_OPEN``, so the ``EXCLUSIVE`` arm below is unreachable dead code, kept
    only because :class:`~robustmargin.IntervalShape` still has the member and nothing forbids a
    future ``dependence="paired"`` result from reaching it.
    """

    @given(
        theta_a=_VALUE,
        theta_b=_VALUE,
        sigma_log_a=_WIDE_LOG_SE,
        sigma_log_b=_WIDE_LOG_SE,
    )
    @settings(max_examples=_MAX_EXAMPLES, deadline=None)
    def test_estimate_is_always_in_the_reported_confidence_set(
        self, theta_a: float, theta_b: float, sigma_log_a: float, sigma_log_b: float
    ) -> None:
        t_a = _threshold_with_known_log_se(value=theta_a, sigma_log=sigma_log_a)
        t_b = _threshold_with_known_log_se(value=theta_b, sigma_log=sigma_log_b)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)

        assert result.estimate is not None
        if result.shape is IntervalShape.BOUNDED:
            assert result.lo is not None and result.hi is not None
            assert result.lo <= result.estimate <= result.hi
        elif result.shape is IntervalShape.HALF_OPEN:
            assert result.lo is not None and result.hi is None
            assert result.lo <= result.estimate
        elif result.shape is IntervalShape.EXCLUSIVE:  # pragma: no cover - paired dependence only
            assert result.lo is not None and result.hi is not None
            assert result.estimate <= result.lo or result.estimate >= result.hi
        else:
            assert result.shape is IntervalShape.UNBOUNDED
            assert result.lo is None and result.hi is None


class TestLogDeltaVarianceAlgebra:
    """``Var(log ratio) = Va + Vb`` (``C=0`` under independence), asserted directly from the
    reported interval's own implied variance, not merely by checking the endpoints land near
    some expected number."""

    @given(
        theta_a=_VALUE,
        theta_b=_VALUE,
        sigma_log_a=_SMALL_LOG_SE,
        sigma_log_b=_SMALL_LOG_SE,
    )
    @settings(max_examples=_MAX_EXAMPLES, deadline=None)
    def test_variance_of_log_ratio_is_the_sum_of_the_two_log_scale_variances(
        self, theta_a: float, theta_b: float, sigma_log_a: float, sigma_log_b: float
    ) -> None:
        t_a = _threshold_with_known_log_se(value=theta_a, sigma_log=sigma_log_a)
        t_b = _threshold_with_known_log_se(value=theta_b, sigma_log=sigma_log_b)

        result = ratio_interval(
            t_a, t_b, method="log_delta", dependence="independent", level=_LEVEL
        )

        assert result.shape is IntervalShape.BOUNDED
        assert result.lo is not None and result.hi is not None and result.lo > 0.0

        implied_var_log_ratio = (math.log(result.hi / result.lo) / (2.0 * _Z)) ** 2
        expected_var_log_ratio = sigma_log_a**2 + sigma_log_b**2

        assert implied_var_log_ratio == pytest.approx(expected_var_log_ratio, rel=1e-6)


class TestMonotonicityInLevel:
    @given(
        theta_a=_VALUE,
        theta_b=_VALUE,
        sigma_log_a=_SMALL_LOG_SE,
        sigma_log_b=_SMALL_LOG_SE,
    )
    @settings(max_examples=_MAX_EXAMPLES, deadline=None)
    def test_a_99_percent_interval_contains_the_95_percent_interval(
        self, theta_a: float, theta_b: float, sigma_log_a: float, sigma_log_b: float
    ) -> None:
        t_a_95 = _threshold_with_known_log_se(value=theta_a, sigma_log=sigma_log_a, level=0.95)
        t_b_95 = _threshold_with_known_log_se(value=theta_b, sigma_log=sigma_log_b, level=0.95)
        t_a_99 = _threshold_with_known_log_se(value=theta_a, sigma_log=sigma_log_a, level=0.99)
        t_b_99 = _threshold_with_known_log_se(value=theta_b, sigma_log=sigma_log_b, level=0.99)

        result_95 = ratio_interval(
            t_a_95, t_b_95, method="fieller", dependence="independent", level=0.95
        )
        result_99 = ratio_interval(
            t_a_99, t_b_99, method="fieller", dependence="independent", level=0.99
        )
        assume(result_95.shape is IntervalShape.BOUNDED)
        assume(result_99.shape is IntervalShape.BOUNDED)
        assert result_95.lo is not None and result_95.hi is not None
        assert result_99.lo is not None and result_99.hi is not None

        assert result_99.lo <= result_95.lo
        assert result_95.hi <= result_99.hi


class TestFiellerLowerAndUpperNeverNegative:
    """The invariant behind the negative-root clip and the both-roots-negative rule together, stated
    directly rather than case by case: a severity ratio cannot be negative, so **whenever the
    ``fieller`` path reports a finite ``lo`` or ``hi``, neither is negative** -- regardless of which
    branch (``BOUNDED``, ``EXCLUSIVE``, or the ``UNBOUNDED`` produced from the other two) produced
    it. ``_WIDE_LOG_SE`` on both sides sweeps ``A = theta_b^2 - z^2*Vb`` and
    ``K = theta_a^2 - z^2*Va`` through positive and non-positive values on both sides, so this
    single property is meant to reach ``A > 0``, ``A < 0`` with ``K > 0``, and ``A < 0`` with
    ``K <= 0`` all in one sweep, rather than one property per fixed case.
    """

    @given(
        theta_a=_VALUE,
        theta_b=_VALUE,
        sigma_log_a=_WIDE_LOG_SE,
        sigma_log_b=_WIDE_LOG_SE,
    )
    @settings(max_examples=_MAX_EXAMPLES, deadline=None)
    def test_lo_and_hi_are_never_negative(
        self, theta_a: float, theta_b: float, sigma_log_a: float, sigma_log_b: float
    ) -> None:
        t_a = _threshold_with_known_log_se(value=theta_a, sigma_log=sigma_log_a)
        t_b = _threshold_with_known_log_se(value=theta_b, sigma_log=sigma_log_b)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=_LEVEL)

        if result.lo is not None:
            assert result.lo >= 0.0, (
                f"Ratio.lo was negative ({result.lo!r}) for a severity ratio, shape="
                f"{result.shape!r}, theta_a={theta_a!r}, sigma_log_a={sigma_log_a!r}, "
                f"theta_b={theta_b!r}, sigma_log_b={sigma_log_b!r}"
            )
        if result.hi is not None:
            assert result.hi >= 0.0, (
                f"Ratio.hi was negative ({result.hi!r}) for a severity ratio, shape="
                f"{result.shape!r}, theta_a={theta_a!r}, sigma_log_a={sigma_log_a!r}, "
                f"theta_b={theta_b!r}, sigma_log_b={sigma_log_b!r}"
            )
