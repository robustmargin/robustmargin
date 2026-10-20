"""The ratio-of-two-thresholds result and :func:`ratio_interval`, which computes one.

:func:`ratio_interval` solves the classic Fieller quadratic --
``{rho : A*rho^2 - 2*B*rho + K <= 0}`` for ``A = theta_b^2 - z^2*Vb``,
``B = theta_a*theta_b - z^2*C``, ``K = theta_a^2 - z^2*Va`` (``C = 0`` under
``dependence="independent"``) -- on the *natural* severity scale, or the log-delta interval
(``log(ratio)``, variance ``Va + Vb`` on the log scale, exponentiated back around the estimate).
Both reuse the exact per-threshold delta-method variance :mod:`robustmargin.intervals` already
computes for a single threshold's own interval (``intervals._theta_variance``), rather than
reassembling the gradient a second time.

**A note on three different things all called ``theta``.** In the Fieller construction
``theta`` is a threshold's value on the *original* severity scale (:attr:`Threshold.value`); in the
log-delta construction ``theta`` is ``log(x*)``; and :mod:`robustmargin.intervals` writes ``theta``
for yet a third thing, the axis it actually optimises over (``log(severity)`` on a log axis, raw
severity on a linear one). All three coincide only on a log axis. Fieller here always converts
``intervals``' transformed-scale ``Var(theta)`` to the natural scale via the delta method's own
chain rule for ``value = exp(theta)`` on a log axis (``Var(value) = value^2 * Var(theta)``) or the
identity map on a linear axis (``value = theta``, so ``Var(value) = Var(theta)`` directly) -- see
``_fieller_variance``'s docstring for why this is computed directly rather than through a
``value^2 * sigma_log^2`` round trip (the two agree wherever ``value != 0``; the direct route has no
removable singularity at ``value == 0`` on a linear axis).

Several questions the ratio construction leaves open are settled as follows rather than guessed
at: a censored or failed input threshold propagates a status/censoring flag with no numbers,
direction-aware for which side is censored; ``level=None`` inherits from the two inputs, which
must agree, and an explicit mismatching ``level`` also raises; a Fieller lower root below zero is
intersected with the positive parameter space -- ``lo = 0.0``, ``shape`` stays ``BOUNDED``, the
unclipped root goes in ``warnings``; an ``EXCLUSIVE``-candidate set whose two roots are both
negative is exactly the whole positive axis, not a two-ray set to clip -- reported as
``UNBOUNDED``; a ``log_delta`` interval too wide to exponentiate raises rather than inventing a
shape ``Ratio`` has no member for; and with ``K > 0`` the ``EXCLUSIVE``-candidate branch
(``A < 0``, an open region of the parameter space) yields a half-line, reported as
``IntervalShape.HALF_OPEN``. The measure-zero boundary ``A == 0`` next to it yields the same
half-line and is reported the same way. The shared-``Fit`` check of the
``dependence="independent"`` guard is made by :func:`ratio_interval`, not by :class:`Ratio`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from scipy.stats import norm

from robustmargin.intervals import _MAX_LOG_HALF_WIDTH, _theta_variance
from robustmargin.threshold import Threshold, _resolve_target
from robustmargin.types import Censoring, IntervalShape, JSONValue, Status, _check_finite_number

__all__ = ["Ratio", "ratio_interval"]

_VALID_METHODS = frozenset({"fieller", "log_delta"})
_VALID_DEPENDENCE = frozenset({"independent"})


def _check_finite(value: float | None, *, label: str) -> None:
    if value is not None:
        _check_finite_number(value, label=f"Ratio.{label}")


@dataclass(frozen=True, kw_only=True)
class Ratio:
    """The ratio of two thresholds' severities, on the same axis, with a confidence interval.

    A Fieller-type interval is not always a single bounded interval: ``shape`` records
    which geometry ``lo``/``hi`` describe, and that geometry is never clipped to look bounded
    when it isn't.

    **A failed or censored ratio carries no numbers.** Whenever ``status`` is not ``OK`` or
    ``censoring`` is not ``NONE``, ``shape``, ``estimate``, ``lo`` and ``hi`` are all ``None`` --
    :func:`ratio_interval` propagates the input thresholds' status/censoring rather than
    reporting a number, direction-aware for which side is censored. Only on
    the ``OK`` + ``NONE`` path is ``shape`` required to be set, along with the numbers it
    governs.

    Attributes
    ----------
    threshold_a, threshold_b
        The two :class:`~robustmargin.Threshold` results the ratio is between (``a / b``). Their
        axes must match on ``name`` and ``unit``, and their ``definition`` must be equal
        (a ratio between two different target-performance definitions is not one number); a
        mismatch on either raises. With ``dependence="independent"``, the two fits must not be
        the very same object (``threshold_a.fit is threshold_b.fit``, object identity only) and
        must not share a cluster id. Both checks are **necessary, not sufficient**: together they
        can only see object identity and the ``cluster_ids`` a :class:`~robustmargin.Fit` records,
        not whether the two series were drawn from overlapping observations that carry no cluster
        id. Equal fits **by value** are not rejected: two disjoint samples can produce identical
        counts, and that ratio is valid -- only the identical *object* is refused.
        ``ratio_interval`` checks only object identity and ``cluster_ids``, for exactly this reason:
        a full shared-data check would need an observation id that nothing in the package records
        yet.
    estimate
        The point estimate of the ratio. Required and equal to
        ``threshold_a.value / threshold_b.value`` (``math.isclose``, ``rel_tol=1e-9``) when
        ``status`` is ``OK`` and ``censoring`` is ``NONE``; ``None`` otherwise.
    lo, hi
        The interval endpoints, read according to ``shape``, only when ``status`` is ``OK`` and
        ``censoring`` is ``NONE``:

        - ``BOUNDED``: ``[lo, hi]``, both set, ``lo <= hi``, and ``lo <= estimate <= hi``.
          ``method="log_delta"`` additionally requires ``lo > 0``. Also covers a Fieller set
          bounded above but not below on its own, intersected to ``[0, hi]``; ``HALF_OPEN`` is
          never used for that case.
        - ``UNBOUNDED``: both ``None`` (the whole real line, or the positive parameter space
          when intersected); it carries ``estimate``.
        - ``EXCLUSIVE``: both set, ``lo < hi`` strictly, meaning
          ``(-inf, lo] union [hi, inf)``, and ``estimate`` lies in that union (``estimate <=
          lo`` or ``estimate >= hi``) -- the point estimate always lies inside the Fieller
          confidence set, never in the excluded gap. **Unreachable via ``ratio_interval`` in
          this release**: the invariant stays enforced for a directly-constructed
          ``Ratio`` and for a future ``dependence="paired"``.
        - ``HALF_OPEN``: ``lo`` set, ``hi`` is ``None``, meaning ``[lo, inf)``, and
          ``lo <= estimate``. ``method="log_delta"`` additionally requires ``lo > 0``, as for
          ``BOUNDED``. This is what a two-ray Fieller set with ``K > 0`` is once intersected
          with the positive parameter space: an open region of the parameter space, reached
          whenever the denominator is poorly determined and the numerator is not, not a
          measure-zero boundary.

        Both ``None`` otherwise.
    shape
        The :class:`~robustmargin.IntervalShape` of ``(lo, hi)`` when ``status`` is ``OK`` and
        ``censoring`` is ``NONE``; ``None`` otherwise (including when ``status``/``censoring``
        make a shape meaningless).
    method
        ``"fieller"`` or ``"log_delta"``. ``"log_delta"`` additionally requires, on the
        ``OK`` + ``NONE`` path, that ``threshold_a.axis.scale`` and ``threshold_b.axis.scale``
        are both ``"log"`` and that ``threshold_a.value`` and ``threshold_b.value`` are both
        ``> 0`` (the log-scale delta method is undefined off the log axis or through zero).
    dependence
        ``"independent"`` in this release (``"paired"`` is not yet implemented). With
        ``"independent"``, the two thresholds' fits must not share a cluster id.
    level
        The confidence level, strictly between 0 and 1.
    censoring
        The :class:`~robustmargin.Censoring` propagated from the input thresholds,
        direction-aware: read against the ratio itself, not against either threshold's own
        tested range.
    status
        The :class:`~robustmargin.Status` of this ratio.
    warnings
        Free-text notes surfaced alongside the ratio. Empty by default. :func:`ratio_interval`
        carries both ``threshold_a.warnings`` and ``threshold_b.warnings`` here too -- in
        particular a ``"NON_MONOTONE_DATA: "`` entry on either input threshold propagates
        through, on every path (censored/failed and the ``OK``+``NONE`` path alike).
    schema_version
        The serialised-result schema version this object belongs to. Defaults to ``"1"``, the
        only version ``from_dict`` reads; any other version is rejected.
    provenance
        An opaque mapping the caller may attach to record where the inputs came from.
        robustmargin stores it and never interprets it. Empty by default.
    """

    threshold_a: Threshold
    threshold_b: Threshold
    estimate: float | None
    lo: float | None
    hi: float | None
    shape: IntervalShape | None
    method: str
    dependence: str
    level: float
    censoring: Censoring
    status: Status
    warnings: tuple[str, ...] = ()
    schema_version: str = "1"
    provenance: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))

        if self.threshold_a.axis.name != self.threshold_b.axis.name:
            raise ValueError(
                "Ratio.threshold_a and .threshold_b must share an axis name, got "
                f"{self.threshold_a.axis.name!r} and {self.threshold_b.axis.name!r}"
            )
        if self.threshold_a.axis.unit != self.threshold_b.axis.unit:
            raise ValueError(
                "Ratio.threshold_a and .threshold_b must share an axis unit, got "
                f"{self.threshold_a.axis.unit!r} and {self.threshold_b.axis.unit!r}"
            )
        if self.threshold_a.definition != self.threshold_b.definition:
            raise ValueError(
                "Ratio.threshold_a.definition must equal Ratio.threshold_b.definition, got "
                f"{self.threshold_a.definition!r} and {self.threshold_b.definition!r}"
            )

        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"Ratio.method must be one of {sorted(_VALID_METHODS)}, got {self.method!r}"
            )
        if self.dependence not in _VALID_DEPENDENCE:
            raise ValueError(
                f"Ratio.dependence must be one of {sorted(_VALID_DEPENDENCE)}, "
                f"got {self.dependence!r}"
            )
        _check_finite_number(self.level, label="Ratio.level")
        if not (0.0 < self.level < 1.0):
            raise ValueError(f"Ratio.level must satisfy 0 < level < 1, got {self.level!r}")

        # The constructor does not check `threshold_a.fit is threshold_b.fit`. A directly
        # constructed `Ratio` over one `Fit` under `dependence="independent"` is accepted,
        # although its interval is wrong (that ratio is identically 1 with zero variance);
        # `ratio_interval` refuses it, and every real caller goes through that function. The
        # check stays out of the constructor so that the public type's accepted inputs do not
        # change. The cluster-id check below is enforced here.
        if self.dependence == "independent":
            ids_a = self.threshold_a.fit.cluster_ids
            ids_b = self.threshold_b.fit.cluster_ids
            if ids_a is not None and ids_b is not None:
                overlap = set(ids_a) & set(ids_b)
                if overlap:
                    raise ValueError(
                        "Ratio: threshold_a.fit and threshold_b.fit share cluster id(s) "
                        f"{sorted(overlap, key=str)!r}, which is incompatible with "
                        "dependence='independent'"
                    )

        _check_finite(self.estimate, label="estimate")
        _check_finite(self.lo, label="lo")
        _check_finite(self.hi, label="hi")

        is_ok_none = self.status is Status.OK and self.censoring is Censoring.NONE

        if not is_ok_none:
            if (
                self.shape is not None
                or self.estimate is not None
                or self.lo is not None
                or self.hi is not None
            ):
                raise ValueError(
                    "Ratio.shape, .estimate, .lo and .hi must all be None when status is not "
                    "OK or censoring is not NONE"
                )
            return

        if self.shape is None:
            raise ValueError("Ratio.shape must be set when status is OK and censoring is NONE")

        if self.threshold_a.censoring is not Censoring.NONE:
            raise ValueError(
                "Ratio.threshold_a.censoring must be NONE when the ratio's own status is OK "
                f"and censoring is NONE, got {self.threshold_a.censoring!r}"
            )
        if self.threshold_b.censoring is not Censoring.NONE:
            raise ValueError(
                "Ratio.threshold_b.censoring must be NONE when the ratio's own status is OK "
                f"and censoring is NONE, got {self.threshold_b.censoring!r}"
            )

        if self.threshold_a.status is not Status.OK or self.threshold_a.value is None:
            raise ValueError(
                "Ratio.threshold_a must have status OK with value set when the ratio's own "
                "status is OK and censoring is NONE"
            )
        if self.threshold_b.status is not Status.OK or self.threshold_b.value is None:
            raise ValueError(
                "Ratio.threshold_b must have status OK with value set when the ratio's own "
                "status is OK and censoring is NONE"
            )

        if self.estimate is None:
            raise ValueError("Ratio.estimate must be set when status is OK and censoring is NONE")
        estimate = self.estimate

        if not self.threshold_b.value > 0.0:
            raise ValueError(
                f"Ratio.threshold_b.value must be > 0 to divide by, got {self.threshold_b.value!r}"
            )
        expected = self.threshold_a.value / self.threshold_b.value
        if not math.isclose(estimate, expected, rel_tol=1e-9):
            raise ValueError(
                "Ratio.estimate must equal threshold_a.value / threshold_b.value "
                f"({expected!r}), got {estimate!r}"
            )

        if self.shape is IntervalShape.BOUNDED:
            if self.lo is None or self.hi is None:
                raise ValueError(
                    "Ratio.lo and .hi must both be set when shape is BOUNDED and censoring is NONE"
                )
            if not self.lo <= self.hi:
                raise ValueError(
                    f"Ratio.lo must not exceed .hi when shape is BOUNDED, got lo={self.lo!r}, "
                    f"hi={self.hi!r}"
                )
            if not (self.lo <= estimate <= self.hi):
                raise ValueError(
                    "Ratio.estimate must lie within [lo, hi] when shape is BOUNDED, got "
                    f"estimate={estimate!r}, lo={self.lo!r}, hi={self.hi!r}"
                )
            if self.method == "log_delta" and not self.lo > 0.0:
                raise ValueError(
                    f"Ratio.lo must be > 0 when method='log_delta', got lo={self.lo!r}"
                )
        elif self.shape is IntervalShape.UNBOUNDED:
            if self.lo is not None or self.hi is not None:
                raise ValueError(
                    "Ratio.lo and .hi must both be None when shape is UNBOUNDED and censoring "
                    "is NONE"
                )
        elif self.shape is IntervalShape.EXCLUSIVE:
            # Unreachable through `ratio_interval` while `dependence="independent"` fixes the
            # covariance term at 0: every same-signed-root case that would
            # produce this shape needs a nonzero covariance, which only a future
            # `dependence="paired"` can supply. The invariant stays enforced -- for a `Ratio`
            # built directly (as `robustmargin.testing.fake_ratio` does) and for when `paired`
            # lands -- rather than being deleted along with the now-dead production path.
            if self.lo is None or self.hi is None:
                raise ValueError(
                    "Ratio.lo and .hi must both be set when shape is EXCLUSIVE and censoring "
                    "is NONE"
                )
            if not self.lo < self.hi:
                raise ValueError(
                    "Ratio.lo must be strictly less than .hi when shape is EXCLUSIVE, got "
                    f"lo={self.lo!r}, hi={self.hi!r}"
                )
            if not (estimate <= self.lo or estimate >= self.hi):
                raise ValueError(
                    "Ratio.estimate must lie outside (lo, hi) when shape is EXCLUSIVE, got "
                    f"estimate={estimate!r}, lo={self.lo!r}, hi={self.hi!r}"
                )
        elif self.shape is IntervalShape.HALF_OPEN:
            # `[lo, inf)`: always lower-bounded, never upper-bounded -- a set bounded only
            # above is reported as `BOUNDED` `[0, hi]` instead.
            if self.lo is None:
                raise ValueError(
                    "Ratio.lo must be set when shape is HALF_OPEN and censoring is NONE"
                )
            if self.hi is not None:
                raise ValueError(
                    "Ratio.hi must be None when shape is HALF_OPEN and censoring is NONE, got "
                    f"hi={self.hi!r}"
                )
            if not (self.lo <= estimate):
                raise ValueError(
                    "Ratio.estimate must be >= lo when shape is HALF_OPEN, got "
                    f"estimate={estimate!r}, lo={self.lo!r}"
                )
            if self.method == "log_delta" and not self.lo > 0.0:
                raise ValueError(
                    f"Ratio.lo must be > 0 when method='log_delta', got lo={self.lo!r}"
                )

        if self.method == "log_delta":
            if self.threshold_a.axis.scale != "log" or self.threshold_b.axis.scale != "log":
                raise ValueError(
                    "Ratio.method='log_delta' requires threshold_a.axis.scale and "
                    "threshold_b.axis.scale to both be 'log', got "
                    f"{self.threshold_a.axis.scale!r} and {self.threshold_b.axis.scale!r}"
                )
            if not (self.threshold_a.value > 0.0 and self.threshold_b.value > 0.0):
                raise ValueError(
                    "Ratio.method='log_delta' requires threshold_a.value and "
                    f"threshold_b.value to both be > 0, got {self.threshold_a.value!r} and "
                    f"{self.threshold_b.value!r}"
                )
            if self.shape not in (IntervalShape.BOUNDED, IntervalShape.HALF_OPEN):
                raise ValueError(
                    "Ratio.method='log_delta' requires shape=BOUNDED or HALF_OPEN, got "
                    f"shape={self.shape!r}"
                )


# ------------------------------------------------------------------------------------------
# ratio_interval(): guards, per-threshold variance, Fieller/log-delta shape, censoring
# propagation.
# ------------------------------------------------------------------------------------------

# `Censoring` flip for the denominator: a directional flag on `threshold_b`
# is re-read against the ratio in the opposite direction from the same flag on `threshold_a`.
_CENSORING_FLIP: dict[Censoring, Censoring] = {
    Censoring.RIGHT: Censoring.LEFT,
    Censoring.LEFT: Censoring.RIGHT,
    Censoring.OPEN_UPPER: Censoring.OPEN_LOWER,
    Censoring.OPEN_LOWER: Censoring.OPEN_UPPER,
}


def _theta_variance_for(t: Threshold) -> float:
    """``Var(theta)`` on :mod:`robustmargin.intervals`' own transformed axis, for threshold ``t`` --
    delegates the delta-method gradient assembly to ``intervals._theta_variance`` (the same
    function :func:`robustmargin.intervals.delta_interval` uses for its own cross-check interval)
    rather than reconstructing it here, and never reconstructs a variance by inverting ``t.lo``/
    ``t.hi`` -- that fails for a profile interval (whose ``lo``/``hi`` are not delta-method
    endpoints at all) and for the open- or both-sides-open cases, where one or both are ``None``.
    Clamped to ``>= 0`` against floating-point noise, mirroring
    ``intervals.delta_interval``'s own ``max(var_theta, 0.0)``.
    """
    target, _ = _resolve_target(t.definition, fit=t.fit)
    return max(_theta_variance(t.fit, definition=t.definition, target=target), 0.0)


def _fieller_variance(t: Threshold, *, value: float) -> float:
    """``Va``/``Vb`` in the Fieller quadratic: the *natural*-severity-scale
    variance of ``t``'s point estimate (module docstring's "three different thetas" note).

    Converts ``_theta_variance_for``'s transformed-scale ``Var(theta)`` via the delta method's
    own chain rule for ``value = exp(theta)`` on a log axis (``Var(value) = value^2 *
    Var(theta)``), or the identity on a linear axis (``value = theta`` already, so
    ``Var(value) = Var(theta)`` directly). This is algebraically the same
    ``Va = value^2 * sigma_log^2`` identity (``sigma_log`` being ``intervals.delta_interval``'s own
    log-scale standard error) wherever ``value != 0``, but computed without ever dividing by
    ``value``, so a linear-axis threshold whose value is exactly 0 gets the exact answer instead of
    a 0/0 division.
    """
    var_theta = _theta_variance_for(t)
    return value * value * var_theta if t.axis.scale == "log" else var_theta


# How far below zero `disc` may fall, relative to the scale of the two
# terms it is the difference of, before a clamp to 0.0 is worth a warning rather than treated as
# ordinary floating-point rounding. Matches the `rel_tol=1e-9` convention `Covariance` already
# uses for its own symmetry check (`models.py`).
_DISC_RELATIVE_EPS = 1e-9


def _stable_quadratic_roots(
    *, a_coef: float, b_coef: float, k_coef: float, disc: float
) -> tuple[float, float]:
    """The two roots of ``A*rho^2 - 2*B*rho + K = 0`` (note the factor of 2 on ``B``, matching
    this module's Fieller coefficients), computed with the substitution that avoids catastrophic
    cancellation in the naive ``(B +/- sqrt(disc)) / A`` when ``|B|`` and ``sqrt(disc)`` are
    close -- exactly the case at the branch boundary this module cares about (measured relative
    error ``1.35e-6`` at ``A=1e-10``, ``5.6e-5`` at ``A=1e-12`` from the naive formula):
    ``q = B + copysign(sqrt(disc), B)``, ``r1 = q/A``, ``r2 = K/q``. Requires
    ``disc >= 0`` and ``a_coef != 0`` -- both already guaranteed by every call site.
    """
    sq = math.sqrt(disc)
    if b_coef == 0.0 and sq == 0.0:
        # A fully degenerate double root at 0 (only reachable when B and K are both exactly 0
        # too, given disc = B^2 - A*K and a_coef != 0) -- the general formula below would divide
        # 0.0 by 0.0 for the second root.
        return 0.0, 0.0
    q = b_coef + math.copysign(sq, b_coef)
    return q / a_coef, k_coef / q


def _fieller_shape(
    *, theta_a: float, va: float, theta_b: float, vb: float, z: float
) -> tuple[IntervalShape, float | None, float | None, tuple[str, ...]]:
    """Solve ``{rho : A*rho^2 - 2*B*rho + K <= 0}`` (module docstring) for
    ``A = theta_b^2 - z^2*Vb``, ``B = theta_a*theta_b`` (``C = 0`` under
    ``dependence="independent"``), ``K = theta_a^2 - z^2*Va``, then intersects the result with
    the positive parameter space a severity ratio must live in.

    ``A > 0``: ``BOUNDED``. The discriminant is non-negative here: the quadratic at the point
    estimate is ``q(theta_a/theta_b) = -z^2 * (Va + rho_hat^2 * Vb) <= 0`` and ``A > 0``, so two
    real roots bracket the estimate. Under ``C = 0`` this holds unconditionally; the guard in
    the code is kept for a future estimated covariance. A lower root below zero is intersected
    with the positive parameter space -- ``lo = 0.0``, the unclipped root goes in ``warnings``,
    ``shape`` stays ``BOUNDED`` (the upper-only geometry never becomes ``HALF_OPEN``).

    ``A < 0`` with a positive discriminant: ``K > 0`` implies ``theta_a > 0`` and hence
    ``B = theta_a*theta_b > 0`` under independence, so the roots' product (``K/A``) and sum
    (``2*B/A``) pin down exactly two outcomes:

    - ``K > 0``: the roots are opposite-signed (product ``< 0``), so intersected with the
      positive axis the set is exactly ``[hi, inf)`` -- ``HALF_OPEN`` with ``lo`` set to the
      positive root (the larger of the two), **not** ``EXCLUSIVE``.
    - ``K <= 0``: both roots are negative (product ``>= 0``, sum ``< 0``), so the set is exactly
      ``(0, inf)`` -- ``UNBOUNDED``, exact.

    ``EXCLUSIVE`` (same-signed roots with ``A < 0``) needs ``B <= 0``, i.e. a nonzero covariance
    term -- unreachable through this function while ``dependence="independent"`` fixes ``C = 0``.
    :class:`Ratio`'s invariants for it stay enforced against a future ``dependence="paired"``.

    ``A < 0`` with a non-positive discriminant: ``UNBOUNDED`` (no real roots, confidence set is
    the whole line).

    ``A == 0`` exactly (``g == 1``): the quadratic degenerates to the linear
    ``-2*B*rho + K <= 0``, i.e. ``rho >= K/(2*B)`` when ``B > 0``. ``K > 0`` puts that boundary
    strictly above zero -- ``HALF_OPEN`` with ``lo = K/(2*B)``. ``A == 0`` is itself
    measure-zero, but it borders the open ``A < 0, K > 0`` region, which gives the same
    half-line, so the two are reported alike. ``K <= 0`` puts the boundary at or below zero, so
    the intersected set is exactly ``(0, inf)`` -- ``UNBOUNDED``.

    Raises
    ------
    ValueError
        If ``va`` or ``vb`` is not finite: a NaN variance passes none
        of the sign tests above and would otherwise fall through to the ``A == 0`` branch,
        reporting "``A`` is exactly 0" -- a false statement about a quantity that was never
        computed to a real number in the first place.
    """
    if not (math.isfinite(va) and math.isfinite(vb)):
        raise ValueError(
            "ratio_interval: method='fieller' needs finite per-threshold variances, got "
            f"Va={va!r} and Vb={vb!r}. A non-finite delta-method variance means the interval "
            "shape cannot be determined -- this is a distinct failure from the A == 0 boundary "
            "and is not silently routed through it"
        )

    a_coef = theta_b * theta_b - z * z * vb
    b_coef = theta_a * theta_b
    k_coef = theta_a * theta_a - z * z * va
    disc = b_coef * b_coef - a_coef * k_coef

    if a_coef > 0.0:
        collected_warnings: list[str] = []
        scale = max(abs(b_coef * b_coef), abs(a_coef * k_coef), 1.0)
        if disc < -_DISC_RELATIVE_EPS * scale:
            collected_warnings.append(
                f"ratio_interval: the Fieller discriminant is negative ({disc!r}) in the "
                "A > 0 branch, where it is mathematically guaranteed to be >= 0 (the point "
                "estimate lies inside the confidence set, so the roots are real); clamped to "
                "0.0 rather than "
                "silently -- this may indicate numerical trouble upstream, not just "
                "floating-point rounding"
            )
        disc_clamped = max(disc, 0.0)
        r1, r2 = _stable_quadratic_roots(
            a_coef=a_coef, b_coef=b_coef, k_coef=k_coef, disc=disc_clamped
        )
        lo, hi = sorted([r1, r2])
        if lo < 0.0:
            # K <= 0 puts rho=0 in the confidence set, so the lower root is below zero --
            # impossible for a severity ratio. Intersect with the positive parameter space
            # rather than report a negative bound. The upper-only geometry this produces stays
            # BOUNDED, never HALF_OPEN: HALF_OPEN is reserved for the lower-bounded-only shape
            # below.
            collected_warnings.append(
                "ratio_interval: the Fieller lower root is below zero "
                f"(unclipped lo={lo!r}); a severity ratio cannot be negative, so the reported "
                "interval is intersected with the positive parameter space (lo=0.0) -- this "
                "stays a conservative superset of the true confidence set"
            )
            lo = 0.0
        return IntervalShape.BOUNDED, lo, hi, tuple(collected_warnings)

    if a_coef < 0.0 and disc > 0.0:
        r1, r2 = _stable_quadratic_roots(a_coef=a_coef, b_coef=b_coef, k_coef=k_coef, disc=disc)
        lo, hi = sorted([r1, r2])
        if k_coef <= 0.0:
            # Under `dependence="independent"` (C=0), B = theta_a*theta_b > 0, so the roots'
            # product K/A >= 0 (K <= 0, A < 0) and their sum 2*B/A < 0 -- both roots share a
            # sign, and that sign is negative. The right ray [hi, inf) with hi < 0 already
            # contains all of (0, inf), so intersected with the positive parameter space the
            # confidence set is exactly (0, inf): UNBOUNDED, exact.
            warning = (
                "ratio_interval: the Fieller set is two rays both lying below zero "
                f"(unclipped lo={lo!r}, hi={hi!r}); a severity ratio cannot be negative, so the "
                "set intersected with the positive parameter space is unconstrained on (0, inf) "
                "-- reporting shape=UNBOUNDED exactly, not as a conservative superset"
            )
            return IntervalShape.UNBOUNDED, None, None, (warning,)
        # K > 0 with A < 0 forces opposite-signed roots (product K/A < 0), so
        # under independence this branch is always a half-line intersected with the positive
        # axis -- [hi, inf), reported as HALF_OPEN with lo=hi (the positive root), never
        # EXCLUSIVE. EXCLUSIVE needs B <= 0 (a nonzero covariance term), unreachable while C=0.
        return IntervalShape.HALF_OPEN, hi, None, ()

    if a_coef < 0.0:
        return IntervalShape.UNBOUNDED, None, None, ()

    # `a_coef == 0.0` exactly: `g == 1`. The quadratic degenerates to the linear
    # `-2*B*rho + K <= 0`. When `B > 0` (independence's usual case), the solution is
    # `rho >= K/(2*B)`: a genuine half-line, split on K's sign.
    if b_coef > 0.0:
        boundary = k_coef / (2.0 * b_coef)
        if k_coef > 0.0:
            # A genuine finite half-line entirely on the positive side -- HALF_OPEN with
            # lo = K/2B.
            return IntervalShape.HALF_OPEN, boundary, None, ()
        warning = (
            "ratio_interval: the Fieller quadratic's leading coefficient A is exactly 0 (g == "
            f"1); the degenerate half-line's boundary (rho >= {boundary!r}) is at or below "
            "zero, so the confidence set intersected with the positive parameter space is "
            "exactly (0, inf) -- reporting shape=UNBOUNDED, exact rather than a lossy superset "
            "for this sub-case"
        )
        return IntervalShape.UNBOUNDED, None, None, (warning,)

    # B <= 0 cannot arise under dependence="independent" (B = theta_a*theta_b, and both values
    # are non-negative by Threshold's own invariant), so this is a defensive branch matching the
    # general algebra rather than a case reachable in this release.
    if b_coef == 0.0:
        # A == 0 and B == 0 degenerate the inequality to `K <= 0`, so K > 0 is an EMPTY
        # confidence set -- not a half-line, and emphatically not the whole axis. Reporting
        # UNBOUNDED there would be the least informative possible answer to an input that is
        # actually contradictory, so it raises rather than silently falling back. Unreachable
        # while dependence="independent" keeps B = theta_a*theta_b >= 0 (B == 0 forces
        # theta_a == 0, hence K = -z^2*Va <= 0), but a future paired covariance can reach it.
        if k_coef > 0.0:
            raise ValueError(
                "ratio_interval: the Fieller confidence set is empty (A == 0, B == 0, "
                f"K == {k_coef!r} > 0), which no IntervalShape can represent and which means "
                "the inputs are mutually contradictory rather than merely uninformative"
            )
        note = "no finite half-line endpoint either (B == 0 too)"
    elif b_coef < 0.0:
        note = f"rho <= {k_coef / (2.0 * b_coef)!r}"
    else:
        note = f"rho >= {k_coef / (2.0 * b_coef)!r}"
    warning = (
        "ratio_interval: the Fieller quadratic's leading coefficient A is exactly 0 (g == 1); "
        f"the true confidence set is a half-line ({note}), which IntervalShape has no member "
        "for in this direction -- reporting shape=UNBOUNDED, which is exact here rather than "
        "clipping it to look bounded"
    )
    return IntervalShape.UNBOUNDED, None, None, (warning,)


def _log_delta_shape(
    *, theta_a: float, var_log_a: float, theta_b: float, var_log_b: float, z: float
) -> tuple[float, float]:
    """``log(ratio) = log(theta_a) - log(theta_b)``, variance ``Va + Vb`` (``C = 0`` under
    ``dependence="independent"``), exponentiated back around the point estimate. Always
    ``BOUNDED`` -- :class:`Ratio`'s own invariant requires it for ``method="log_delta"`` -- so a
    half-width too large for a finite ``float`` has no shape left to fall back to and raises
    instead: no number is invented, and no shape ``Ratio`` cannot represent for this method is
    reported.
    """
    sigma_log_ratio = math.sqrt(max(var_log_a + var_log_b, 0.0))
    half_width = z * sigma_log_ratio
    estimate = theta_a / theta_b
    if not math.isfinite(half_width) or half_width > _MAX_LOG_HALF_WIDTH:
        raise ValueError(
            "ratio_interval: method='log_delta' has a log-scale standard error too large for a "
            f"finite interval (sigma_log_ratio={sigma_log_ratio!r}); method='log_delta' must "
            "report shape=BOUNDED (Ratio's own invariant) and there is no defensible bounded "
            "interval for this input -- method='fieller' can report shape=UNBOUNDED explicitly "
            "instead"
        )
    lo = estimate * math.exp(-half_width)
    hi = estimate * math.exp(half_width)
    # `lo > 0.0` as well as finiteness: a half-width under _MAX_LOG_HALF_WIDTH can still
    # underflow `lo` to exactly 0.0 for a small enough estimate, and 0.0 is "finite". Without
    # this the failure surfaces from Ratio.__post_init__ as the log_delta lo > 0 invariant
    # rather than as an explanation of what actually went wrong.
    if not (math.isfinite(lo) and math.isfinite(hi) and lo > 0.0):
        raise ValueError(
            "ratio_interval: method='log_delta' produced a non-finite interval endpoint for "
            "this input; there is no defensible bounded interval to report"
        )
    return lo, hi


def _propagate_censoring(
    t_a: Threshold, t_b: Threshold
) -> tuple[Status, Censoring, tuple[str, ...]]:
    """The ``(Status, Censoring)`` a :class:`Ratio` carries when at least one input threshold is
    not ``(OK, NONE)``: a flag, never a number.

    **Status.** ``Status.OK`` only when both inputs are ``OK`` (a ``(OK, RIGHT)`` threshold --
    a converged fit whose bound is censored -- is an ``OK`` input). When exactly one input has a
    failure status, the ratio carries it. When both do, and they **differ**, the ratio carries
    ``threshold_a``'s and records both in ``warnings``; no severity ordering over ``Status`` is
    invented to pick a "worse" one -- none exists anywhere in this package.

    **Censoring.** Direction-aware: a numerator's directional flag
    (``RIGHT``/``LEFT``/``OPEN_UPPER``/``OPEN_LOWER``) carries straight across; a denominator's
    flips (a larger denominator pushes the ratio *down*, and vice versa). When both carry a
    directional flag, there is no coherent direction; ``threshold_a``'s is reported, with a
    warning naming the ambiguity. This is the same ``threshold_a``-wins tie-break as ``Status``
    above, applied to the other field, so the two never disagree about which input won.
    """
    collected_warnings: list[str] = []

    a_failed = t_a.status is not Status.OK
    b_failed = t_b.status is not Status.OK
    if a_failed and b_failed:
        status = t_a.status
        if t_a.status != t_b.status:
            collected_warnings.append(
                "ratio_interval: threshold_a and threshold_b have different failure statuses "
                f"(threshold_a={t_a.status!s}, threshold_b={t_b.status!s}); threshold_a's "
                "status is reported"
            )
    elif a_failed:
        status = t_a.status
    elif b_failed:
        status = t_b.status
    else:
        status = Status.OK

    a_dir = t_a.censoring
    b_dir = t_b.censoring
    a_has_direction = a_dir is not Censoring.NONE
    b_has_direction = b_dir is not Censoring.NONE
    if a_has_direction and b_has_direction:
        censoring = a_dir
        collected_warnings.append(
            "ratio_interval: both threshold_a and threshold_b carry censoring "
            f"(threshold_a={a_dir!s}, threshold_b={b_dir!s}); there is no coherent direction "
            f"for the ratio's bound, so threshold_a's flag ({a_dir!s}) is reported"
        )
    elif a_has_direction:
        censoring = a_dir
    elif b_has_direction:
        censoring = _CENSORING_FLIP[b_dir]
    else:
        censoring = Censoring.NONE

    return status, censoring, tuple(collected_warnings)


def ratio_interval(
    t_a: Threshold,
    t_b: Threshold,
    /,
    *,
    method: str,
    dependence: str,
    level: float | None = None,
) -> Ratio:
    """The ratio of two thresholds' severities, ``t_a / t_b``, on the same axis, with a
    confidence interval.

    ``method="fieller"`` solves the classic Fieller quadratic on the natural severity scale;
    ``method="log_delta"`` uses the log-scale delta method and additionally requires, once both
    inputs are known to be ``(OK, NONE)``, that both axes' ``scale`` are ``"log"`` and both
    thresholds' ``value`` are ``> 0``. Both use the normal quantile ``z``, never a Student-t
    quantile, matching the rest of robustmargin -- R ``drc``'s own
    ``EDcomp(interval="fieller")`` uses ``qt`` instead, which is why its interval endpoints (not
    its point estimates) do not match robustmargin's at the usual reference tolerance
    (the reference comparison at
    https://github.com/robustmargin/robustmargin/blob/main/tests/reference/test_ratio_drc_edcomp.py).

    ``IntervalShape.EXCLUSIVE`` is **unreachable** through this function while
    ``dependence="independent"``: a two-ray Fieller set turns out, once intersected with the
    positive parameter space, to always be either the half-line ``[hi, inf)`` (reported as
    ``IntervalShape.HALF_OPEN``; an open region of the parameter space, not a measure-zero
    boundary -- how often it occurs is a property of the design's power) or the whole positive
    axis (``UNBOUNDED``) -- never a genuine two-ray exclusion, which needs a nonzero covariance
    term that only a future ``dependence="paired"`` can supply. ``EXCLUSIVE``'s :class:`Ratio`
    invariants stay enforced against that future, but no call through this function reaches them
    today.

    Parameters
    ----------
    t_a, t_b
        The two thresholds, ``a / b``. Must share ``axis.name``, ``axis.unit`` and
        ``definition``; under ``dependence="independent"``, their fits must not share a cluster
        id. All three are :class:`Ratio`'s own ``__post_init__`` invariants and are not
        duplicated here -- they fire when the resulting :class:`Ratio` is constructed, on every
        path through this function.
    method
        ``"fieller"`` or ``"log_delta"``.
    dependence
        ``"independent"`` only, in this release. ``"paired"`` raises, naming that it is not
        implemented yet (mirrors :func:`robustmargin.threshold`'s own guard for the same request).
        Under ``"independent"``, ``t_a.fit`` and ``t_b.fit`` must not be the same object and
        must not share a cluster id -- both checks are necessary but **not sufficient** for
        actual independence.

        **This function does not verify independence; establishing it is the caller's
        responsibility.** Nothing in robustmargin records an observation id (only
        :attr:`~robustmargin.Fit.cluster_ids`, one entry per distinct cluster, not per row), so two
        fits built from overlapping rows that carry no cluster ids are accepted here and produce
        a ``Ratio`` whose stated coverage is wrong -- silently, because there is nothing left in
        the package to check it against. This is a documented limitation, not a guarantee this
        function makes.
    level
        ``None`` (the default) inherits the confidence level from ``t_a``/``t_b``, which must
        agree; an explicit ``level`` that disagrees with either also raises -- there is no
        silently-assumed default.

    Returns
    -------
    Ratio
        If either input is censored or has a failed :class:`~robustmargin.Status`, the returned
        ``Ratio`` carries a propagated status/censoring flag and no numbers at all -- never a
        one-sided bound built by mixing an exact censoring bound in one threshold with a
        model-fitted value and its sampling error in the other. ``warnings`` always carries both
        ``t_a.warnings`` and ``t_b.warnings``, whichever path is taken.

    Raises
    ------
    ValueError
        On any of: ``t_a.fit is t_b.fit`` under ``dependence="independent"`` (checked directly
        by this function); mismatched axis name/unit, mismatched ``definition``, or shared cluster
        ids under ``dependence="independent"`` (via :class:`Ratio`'s own validation, once this
        function reaches the point of constructing one); ``method`` outside
        ``{"fieller", "log_delta"}``; ``dependence`` anything other than ``"independent"``
        (``"paired"`` gets its own message, not yet implemented); ``level`` outside ``(0, 1)``, or a
        ``level`` mismatch; ``threshold_b.value == 0`` (nothing to divide by); for
        ``method="log_delta"`` once both inputs are ``(OK, NONE)``, either axis not ``scale="log"``
        or either ``value`` not ``> 0``; for ``method="log_delta"``, a combined log-scale standard
        error too large to exponentiate finitely; or, for ``method="fieller"``, a non-finite
        per-threshold variance.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"ratio_interval: method must be one of {sorted(_VALID_METHODS)}, got {method!r}"
        )
    if dependence == "paired":
        raise ValueError(
            "ratio_interval: dependence='paired' is not implemented in this release; pass "
            "dependence='independent' explicitly to acknowledge independence is being assumed "
            "anyway, or wait for a future release that adds it"
        )
    if dependence not in _VALID_DEPENDENCE:
        raise ValueError(
            "ratio_interval: dependence must be 'independent' ('paired' is not yet "
            f"implemented), got {dependence!r}"
        )

    # The shared-`Fit`-object check lives here, in the function, rather than in
    # `Ratio.__post_init__` where the cluster-id check below (delegated to `Ratio`'s own
    # construction) lives. `Ratio`'s constructor does not check it, so that the public type's
    # accepted inputs do not change; every real caller goes through this function (see the
    # matching comment in `Ratio.__post_init__`).
    if dependence == "independent" and t_a.fit is t_b.fit:
        raise ValueError(
            "ratio_interval: threshold_a.fit and threshold_b.fit are the very same Fit object, "
            "which is incompatible with dependence='independent'. This is an object-identity "
            "check only, not a value-equality one: two disjoint samples may legitimately "
            "produce an equal Fit by value, and that ratio is valid."
        )

    if level is None:
        if t_a.level != t_b.level:
            raise ValueError(
                "ratio_interval: level=None inherits the confidence level from threshold_a and "
                f"threshold_b, which must agree; got threshold_a.level="
                f"{t_a.level!r} and threshold_b.level={t_b.level!r}"
            )
        resolved_level = t_a.level
    else:
        _check_finite_number(level, label="ratio_interval level")
        if not (0.0 < level < 1.0):
            raise ValueError(f"ratio_interval: level must satisfy 0 < level < 1, got {level!r}")
        if level != t_a.level or level != t_b.level:
            raise ValueError(
                f"ratio_interval: level={level!r} does not match threshold_a.level "
                f"({t_a.level!r}) and/or threshold_b.level ({t_b.level!r})"
            )
        resolved_level = level

    # Carry both inputs' own warnings (for example a "NON_MONOTONE_DATA: " token from either
    # threshold) onto the ratio, on every path below -- the censored/failed early return and the
    # OK+NONE path alike.
    collected_warnings: list[str] = [*t_a.warnings, *t_b.warnings]

    a_clean = t_a.status is Status.OK and t_a.censoring is Censoring.NONE
    b_clean = t_b.status is Status.OK and t_b.censoring is Censoring.NONE
    if not (a_clean and b_clean):
        propagated_status, propagated_censoring, propagation_warnings = _propagate_censoring(
            t_a, t_b
        )
        collected_warnings.extend(propagation_warnings)
        return Ratio(
            threshold_a=t_a,
            threshold_b=t_b,
            estimate=None,
            lo=None,
            hi=None,
            shape=None,
            method=method,
            dependence=dependence,
            level=resolved_level,
            censoring=propagated_censoring,
            status=propagated_status,
            warnings=tuple(collected_warnings),
        )

    value_a, value_b = t_a.value, t_b.value
    assert value_a is not None and value_b is not None  # `a_clean`/`b_clean` guarantee this

    # `(Status.OK, Censoring.NONE)` with `lo = hi = None` is a legal `Threshold` -- a converged
    # fit whose profile crossed the target on neither side (`intervals.py`'s own documented
    # "open on both sides" outcome). Nothing else about this function would ever surface that:
    # the delta-method variance used below does not come from `t_a.lo`/`t_a.hi` at all, so a
    # ratio built on such an input looks exactly like an ordinary one unless it is named
    # explicitly here (no silent fallback). `t_a.warnings`/`t_b.warnings` are not repr'd into
    # this message -- `collected_warnings` already starts with both inputs' own warnings in
    # full, so embedding them again here would just duplicate them.
    if t_a.lo is None and t_a.hi is None:
        collected_warnings.append(
            "ratio_interval: threshold_a has status=OK and censoring=NONE but reports no "
            "interval at all (lo and hi are both None); this ratio's variance still comes from "
            "threshold_a's own delta-method covariance"
        )
    if t_b.lo is None and t_b.hi is None:
        collected_warnings.append(
            "ratio_interval: threshold_b has status=OK and censoring=NONE but reports no "
            "interval at all (lo and hi are both None); this ratio's variance still comes from "
            "threshold_b's own delta-method covariance"
        )

    if method == "log_delta":
        if t_a.axis.scale != "log" or t_b.axis.scale != "log":
            raise ValueError(
                "ratio_interval: method='log_delta' requires threshold_a.axis.scale and "
                "threshold_b.axis.scale to both be 'log', got "
                f"{t_a.axis.scale!r} and {t_b.axis.scale!r}"
            )
        # `axis.scale == "log"` already forces `value > 0` (`Threshold`'s own invariant), so this
        # is redundant given the check above -- kept anyway, both to mirror `Ratio`'s own
        # invariant exactly and to fail explicitly rather than rely on that invariant continuing
        # to hold if it is ever loosened.
        if not (value_a > 0.0 and value_b > 0.0):
            raise ValueError(
                "ratio_interval: method='log_delta' requires threshold_a.value and "
                f"threshold_b.value to both be > 0, got {value_a!r} and {value_b!r}"
            )
    elif not value_b > 0.0:
        # Fieller divides by threshold_b.value. A log axis already guarantees `> 0`; a linear
        # axis only guarantees `>= 0` (`Threshold`'s own invariant), so this is the one case
        # Fieller itself must refuse rather than crash on with a raw ZeroDivisionError.
        raise ValueError(
            f"ratio_interval: threshold_b.value must be > 0 to divide by, got {value_b!r}"
        )

    z = float(norm.ppf(0.5 + resolved_level / 2.0))
    estimate = value_a / value_b

    if method == "fieller":
        va = _fieller_variance(t_a, value=value_a)
        vb = _fieller_variance(t_b, value=value_b)
        shape, lo, hi, shape_warnings = _fieller_shape(
            theta_a=value_a, va=va, theta_b=value_b, vb=vb, z=z
        )
        collected_warnings.extend(shape_warnings)
    else:
        # log_delta: both axes are confirmed 'log' above, so `_theta_variance_for` already IS
        # `Var(log(value))` -- no natural-scale detour needed (module docstring).
        var_log_a = _theta_variance_for(t_a)
        var_log_b = _theta_variance_for(t_b)
        lo, hi = _log_delta_shape(
            theta_a=value_a, var_log_a=var_log_a, theta_b=value_b, var_log_b=var_log_b, z=z
        )
        shape = IntervalShape.BOUNDED

    return Ratio(
        threshold_a=t_a,
        threshold_b=t_b,
        estimate=estimate,
        lo=lo,
        hi=hi,
        shape=shape,
        method=method,
        dependence=dependence,
        level=resolved_level,
        censoring=Censoring.NONE,
        status=Status.OK,
        warnings=tuple(collected_warnings),
    )
