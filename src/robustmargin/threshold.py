"""The threshold result and :func:`threshold`, which solves for one.

:func:`threshold` resolves the target performance for a :class:`~robustmargin.Definition`,
classifies the fit against it (:mod:`robustmargin.censoring`), and -- when the
classification says to -- solves the closed form and runs the requested interval method
(:mod:`robustmargin.intervals`). ``interval_method`` is a *request*: every censored or failed
result records ``"exact_bound"`` regardless of what was asked for, because that is the method
that actually produced its numbers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from robustmargin.censoring import _baseline_rate, classify
from robustmargin.empirical import BaselineRate, GridBreakPoint
from robustmargin.intervals import delta_interval, profile_interval, solve_value
from robustmargin.models import Fit, _mirror_fit
from robustmargin.types import (
    Axis,
    Cell,
    Censoring,
    Definition,
    JSONValue,
    Status,
    _check_finite_number,
)

__all__ = ["Threshold", "threshold"]

_VALID_INTERVAL_METHODS = frozenset({"profile", "delta", "exact_bound"})
_TERMINAL_STATUSES = frozenset(
    {Status.UNREACHABLE, Status.CONTROL_INCOMPATIBLE, Status.FAILS_AT_BASELINE}
)
_NONCONVERGENT_STATUSES = frozenset({Status.SEPARATION, Status.NOT_CONVERGED})
_VALID_DIRECTIONS = frozenset({"decreasing", "increasing"})

# A threshold with a value, with status OK, or with an open-sided censoring needs the fit
# underneath it to have actually converged. UNREACHABLE does not: a target outside (0, 1) is
# unreachable whatever the fit did, so it may sit on a failed fit as well as on a converged one.
# It may not sit on a CONTROL_INCOMPATIBLE fit, which is reported as itself before any target is
# examined.
_REQUIRE_FIT_OK_STATUSES = frozenset({Status.OK})
_UNREACHABLE_FIT_STATUSES = frozenset({Status.OK, Status.SEPARATION, Status.NOT_CONVERGED})
_REQUIRE_FIT_OK_CENSORING = frozenset({Censoring.OPEN_UPPER, Censoring.OPEN_LOWER})
# SEPARATION, NOT_CONVERGED and CONTROL_INCOMPATIBLE are properties of the fit itself; a
# threshold reporting one of these must be built on a fit reporting the very same status.
_MIRROR_FIT_STATUS_STATUSES = frozenset(
    {Status.SEPARATION, Status.NOT_CONVERGED, Status.CONTROL_INCOMPATIBLE}
)
_VALID_DEPENDENCE = frozenset({"independent"})


def _check_value(value: float | None, *, label: str, scale: str) -> None:
    if value is None:
        return
    _check_finite_number(value, label=f"Threshold.{label}")
    if scale == "log":
        if not value > 0.0:
            raise ValueError(f"Threshold.{label} must be > 0 on a log axis, got {value!r}")
    elif value < 0.0:
        raise ValueError(f"Threshold.{label} must be non-negative, got {value!r}")


@dataclass(frozen=True, kw_only=True)
class Threshold:
    """Where a fitted curve crosses a target performance, with a confidence interval, or an
    explicit record of why no value is reported.

    Every censoring rule below assumes the **true** performance curve is monotone in severity
    (the fitted curve, being drawn from a monotone parametric family, is monotone by
    construction regardless). A ``RIGHT`` bound reads "threshold > lo (one-sided, exact,
    monotonicity assumed)", never as a value on its own; the mirrored reading applies to
    ``LEFT``.

    **Outcomes that rise with severity.** For an increasing fit, "passing" a target means the
    rate is below it, so ``RIGHT`` reads "still passes at the highest level tested", ``LEFT`` and
    ``FAILS_AT_BASELINE`` read the same way round, and every statement below about a rate being
    at least the target is, for an increasing fit, a statement about it being at most the
    target. The threshold is solved on the complementary outcome with the target complemented;
    severities are never complemented, so ``value``, ``lo``, ``hi`` and ``grid`` are on the
    caller's severity axis, and warnings, the target and ``baseline`` are on the scale of the
    outcome as counted.

    **Exact results share one error budget.** The exact results are ``FAILS_AT_BASELINE``,
    ``RIGHT``, ``LEFT``, and the bracket or one-sided bound of a ``SEPARATION``/``NOT_CONVERGED``
    fit. All of them are read off one-sided exact (Clopper-Pearson) tests, and the error budget
    ``1 - level`` is split four ways, ``(1 - level) / 4`` each, so every test runs at confidence
    ``1 - (1 - level) / 4`` (0.9875 at ``level = 0.95``). The four parts are:

    - an ascending run over the nonzero severities from the lowest: each severity whose lower
      bound clears the target extends the run, and the run stops at the first that does not. Its
      end is the lower bound;
    - the highest nonzero severity alone: its lower bound clearing the target gives ``RIGHT``;
    - a descending run over the nonzero severities from the highest, on upper bounds below the
      target, stopping at the first that does not qualify. Its end is the upper bound;
    - the lowest nonzero severity alone: its upper bound falling below the target gives the upper
      bound at that severity, the upper half of ``LEFT`` (which also needs the control's lower
      bound to clear the target) and half of ``FAILS_AT_BASELINE`` (which also needs the control's
      upper bound to fall below it).

    Each run is a fixed-sequence procedure (Hsu & Berger, 1999, *Journal of the American
    Statistical Association* 94:468-482): under a monotone curve it can only err by wrongly
    passing, or wrongly failing, the first severity past the true crossing, whatever the number of
    tested severities. A bracket needs both sides. If only one side is determined it is reported
    alone, at that same confidence, with a warning naming it; nothing is rescanned at a looser
    one. No bound is ever at the control.

    **The joint guarantee.** Under the assumptions below, the probability that ``threshold()``
    returns an exact result of any kind that is wrong -- one whose claim about where the curve
    crosses the resolved target is false for the true curve -- is at most ``1 - level`` in total,
    whichever kind the data selects. This is a probability over repeated data, not the probability
    that a result, once seen, is wrong. The assumptions are:

    - the true curve is monotone in the stated direction over all tested severities, the control
      included;
    - trials are independent;
    - the target is fixed, as it is under ``absolute``.

    **A target taken from the control.** On a failed fit (``SEPARATION`` or ``NOT_CONVERGED``),
    ``baseline_fraction(p)`` resolves its target from the control's observed rate, ``p * k0 / n0``
    for ``k0`` successes in ``n0`` control trials. The exact claims are then tests against that
    plug-in target. They carry no coverage guarantee for ``p`` times the true zero-severity rate,
    which is the quantity the definition names, and the joint bound above is not proved for a
    target that depends on the control's counts. For a covered claim on a failed fit, use
    ``absolute(a)``. ``baseline_multiple`` has no failed-fit target: it raises.

    **Exact ties.** A claim is read non-strictly at the tested severities: a lower bound means the
    true rate there is at least the target, an upper bound that it is at most the target. If a
    tested severity's true rate equals the target exactly, the strict reading of a claim is not
    covered by the bound above; the proven bound for that reading is ``1.5 * (1 - level)``, and
    the worst case simulated (every tested severity exactly at the target, ``level = 0.95``) was
    below 4%.

    **Scope.** The guarantee is for the exact results only. A model-based ``value`` with its
    profile or delta interval keeps its own model-based coverage, and the guarantee is not proved
    for a converged fit whose asymptotes are estimated under ``baseline_fraction``,
    ``baseline_multiple`` (with an estimated ``lower``) or ``relative``, where the target depends
    on the fit.

    Invariants enforced in ``__post_init__`` (the censoring table, plus the fit-status and
    ordering rules):

    - ``status`` in ``{UNREACHABLE, CONTROL_INCOMPATIBLE, FAILS_AT_BASELINE}``: ``value``,
      ``lo`` and ``hi`` are all ``None``; ``censoring`` is ``NONE``; ``interval_method`` is
      ``"exact_bound"``. There is no threshold at all in these cases -- either it cannot be
      reached (the target lies outside ``(0, 1)``, which no binomial curve crosses, or outside
      what the fitted asymptotes allow), or the question of a crossing does not arise because
      the control already fails.
    - ``status`` in ``{SEPARATION, NOT_CONVERGED}``: ``value`` is ``None``; ``censoring`` is one
      of ``{NONE, RIGHT, LEFT}``; ``interval_method`` is ``"exact_bound"``; ``lo`` and ``hi``
      may be set, as the exact-bound fallback.
    - ``censoring == RIGHT``: ``value`` is ``None``, ``lo`` is set, ``hi`` is ``None``,
      ``interval_method == "exact_bound"``.
    - ``censoring == LEFT``: ``value`` is ``None``, ``hi`` is set, ``lo`` is ``None``,
      ``interval_method == "exact_bound"``.
    - ``censoring == OPEN_UPPER``: ``status == OK``, ``value`` and ``lo`` are set, ``hi`` is
      ``None``, ``lo <= value``, ``interval_method == "profile"``. ``OPEN_LOWER`` mirrors it,
      with ``hi`` set, ``lo`` ``None`` and ``value <= hi``.
    - ``censoring == NONE`` with ``status == OK``: ``value``, ``lo`` and ``hi`` are all set,
      with ``lo <= value <= hi``, and ``interval_method`` is ``"profile"`` or ``"delta"``.
    - When both ``lo`` and ``hi`` are set: ``lo < hi`` strictly for a ``SEPARATION`` /
      ``NOT_CONVERGED`` bracket, ``lo <= hi`` otherwise.
    - On a log axis, ``value``, ``lo`` and ``hi`` must each be ``> 0`` when present (a linear
      axis keeps ``>= 0``).
    - ``fit.status`` must equal ``OK`` whenever ``value`` is set, ``status`` is ``OK``, or
      ``censoring`` is ``OPEN_UPPER``/``OPEN_LOWER``. ``fit.status`` must equal ``status`` itself
      when ``status`` is ``SEPARATION``, ``NOT_CONVERGED`` or ``CONTROL_INCOMPATIBLE``.
      ``UNREACHABLE`` and ``FAILS_AT_BASELINE`` place no constraint on ``fit.status``: a target
      outside ``(0, 1)`` is unreachable whatever the fit did, and the control's exact test does
      not depend on the rest of the curve having converged.

    Attributes
    ----------
    axis
        The :class:`~robustmargin.Axis` this threshold was solved on. Must equal ``fit.axis``.
    definition
        Which target performance this threshold solves for.
    direction
        ``"decreasing"`` or ``"increasing"``. Carried over from ``fit``.
    value
        The estimated crossing severity, or ``None`` per the rules above.
    lo
        The interval's lower side, or a one-sided bound, or ``None``.
    hi
        The interval's upper side, or a one-sided bound, or ``None``.
    level
        The confidence level, strictly between 0 and 1.
    interval_method
        ``"profile"``, ``"delta"`` or ``"exact_bound"``.
    censoring
        The :class:`~robustmargin.Censoring` classification.
    status
        The :class:`~robustmargin.Status` of this threshold.
    dependence
        ``"independent"`` in this release. Required, with no default: a threshold built from
        clustered data must say so explicitly rather than silently assuming independence.
    fit
        The :class:`~robustmargin.Fit` this threshold was solved from. Per-level trial counts
        live on ``fit.cells``, not on a separate field.
    warnings
        Free-text notes surfaced alongside the threshold. Empty by default. Carries a
        ``"NON_MONOTONE_DATA: "``-prefixed entry whenever ``fit.monotonicity.any_flagged``,
        whatever ``status``/``censoring`` turn out to be -- distinct from
        :mod:`robustmargin.censoring`'s own ``"NON_MONOTONE: "`` token, which compares exact
        one-sided bounds against the *fitted* curve rather than testing raw adjacent levels; the two
        may disagree and both are reported when they fire. Carries an ``"EXTRAPOLATED: "``-prefixed
        entry, appended after the interval method's own warnings, whenever a model-based ``value``
        is positive and lies strictly below the lowest or strictly above the highest nonzero tested
        severity. It is report-only: it describes ``value`` alone, whatever the interval method or
        outcome, and changes no number, status or censoring.
    schema_version
        The serialised-result schema version this object belongs to. Defaults to ``"2"``, the
        version ``to_dict`` writes; ``from_dict`` also reads ``"1"``.
    provenance
        An opaque mapping the caller may attach to record where the inputs came from.
        robustmargin stores it and never interprets it. Empty by default.
    baseline
        The zero-severity control's exact rate, set exactly when ``status`` is
        ``FAILS_AT_BASELINE``; ``None`` otherwise. For an increasing fit, ``successes`` and
        ``rate`` are the control's events and event rate.
    grid
        The grid break-point value computed as part of the ``SEPARATION``/``NOT_CONVERGED``
        exact-bound fallback, set exactly on that path; ``None`` otherwise.
    """

    axis: Axis
    definition: Definition
    direction: str
    value: float | None
    lo: float | None
    hi: float | None
    level: float
    interval_method: str
    censoring: Censoring
    status: Status
    dependence: str
    fit: Fit
    warnings: tuple[str, ...] = ()
    schema_version: str = "2"
    provenance: Mapping[str, JSONValue] = field(default_factory=dict)
    baseline: BaselineRate | None = None
    grid: GridBreakPoint | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))

        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"Threshold.direction must be one of {sorted(_VALID_DIRECTIONS)}, "
                f"got {self.direction!r}"
            )
        if self.definition.kind == "baseline_multiple" and self.direction != "increasing":
            raise ValueError(
                "Threshold: definition.kind='baseline_multiple' belongs to an increasing outcome, "
                f"got direction={self.direction!r}"
            )
        if self.definition.kind == "baseline_multiple" and self.axis.scale != "log":
            raise ValueError(
                "Threshold: definition.kind='baseline_multiple' is defined on a log axis only, "
                f"got axis.scale={self.axis.scale!r}"
            )
        if self.definition.kind == "baseline_fraction" and self.direction != "decreasing":
            raise ValueError(
                "Threshold: definition.kind='baseline_fraction' belongs to a decreasing outcome, "
                f"got direction={self.direction!r}"
            )
        if self.interval_method not in _VALID_INTERVAL_METHODS:
            raise ValueError(
                f"Threshold.interval_method must be one of {sorted(_VALID_INTERVAL_METHODS)}, "
                f"got {self.interval_method!r}"
            )
        if self.dependence not in _VALID_DEPENDENCE:
            raise ValueError(
                f"Threshold.dependence must be one of {sorted(_VALID_DEPENDENCE)}, "
                f"got {self.dependence!r}"
            )
        _check_finite_number(self.level, label="Threshold.level")
        if not (0.0 < self.level < 1.0):
            raise ValueError(f"Threshold.level must satisfy 0 < level < 1, got {self.level!r}")
        if self.axis != self.fit.axis:
            raise ValueError("Threshold.axis must equal Threshold.fit.axis")
        if self.direction != self.fit.direction:
            raise ValueError("Threshold.direction must equal Threshold.fit.direction")

        _check_value(self.value, label="value", scale=self.axis.scale)
        _check_value(self.lo, label="lo", scale=self.axis.scale)
        _check_value(self.hi, label="hi", scale=self.axis.scale)

        requires_fit_ok = (
            self.value is not None
            or self.status in _REQUIRE_FIT_OK_STATUSES
            or self.censoring in _REQUIRE_FIT_OK_CENSORING
        )
        if requires_fit_ok and self.fit.status is not Status.OK:
            raise ValueError(
                "Threshold.fit.status must be OK when value is set, status is OK, or "
                f"censoring is OPEN_UPPER/OPEN_LOWER; got fit.status={self.fit.status!r}"
            )
        if self.status is Status.UNREACHABLE and self.fit.status not in _UNREACHABLE_FIT_STATUSES:
            raise ValueError(
                "Threshold.fit.status must be OK, SEPARATION or NOT_CONVERGED when status is "
                f"UNREACHABLE; got fit.status={self.fit.status!r}"
            )
        if self.status in _MIRROR_FIT_STATUS_STATUSES and self.fit.status is not self.status:
            raise ValueError(
                f"Threshold.fit.status must equal Threshold.status ({self.status!r}) when "
                "status is SEPARATION, NOT_CONVERGED or CONTROL_INCOMPATIBLE; got "
                f"fit.status={self.fit.status!r}"
            )

        if self.lo is not None and self.hi is not None:
            if self.status in _NONCONVERGENT_STATUSES:
                if not self.lo < self.hi:
                    raise ValueError(
                        "Threshold.lo must be strictly less than .hi for a "
                        f"SEPARATION/NOT_CONVERGED bracket, got lo={self.lo!r}, hi={self.hi!r}"
                    )
            elif not self.lo <= self.hi:
                raise ValueError(
                    f"Threshold.lo must not exceed .hi, got lo={self.lo!r}, hi={self.hi!r}"
                )

        if self.status in _TERMINAL_STATUSES:
            if self.value is not None or self.lo is not None or self.hi is not None:
                raise ValueError(
                    f"Threshold.value, .lo and .hi must all be None when status={self.status!r}"
                )
            if self.censoring is not Censoring.NONE:
                raise ValueError(
                    f"Threshold.censoring must be NONE when status={self.status!r}, "
                    f"got {self.censoring!r}"
                )
            if self.interval_method != "exact_bound":
                raise ValueError(
                    "Threshold.interval_method must be 'exact_bound' when "
                    f"status={self.status!r}, got {self.interval_method!r}"
                )
            return

        if self.status in _NONCONVERGENT_STATUSES:
            if self.interval_method != "exact_bound":
                raise ValueError(
                    f"Threshold.interval_method must be 'exact_bound' when "
                    f"status={self.status!r}, got {self.interval_method!r}"
                )
            if self.value is not None:
                raise ValueError(f"Threshold.value must be None when status={self.status!r}")
            if self.censoring not in (Censoring.NONE, Censoring.RIGHT, Censoring.LEFT):
                raise ValueError(
                    f"Threshold.censoring must be one of NONE, RIGHT, LEFT when "
                    f"status={self.status!r}, got {self.censoring!r}"
                )

        if self.censoring is Censoring.RIGHT:
            if self.interval_method != "exact_bound":
                raise ValueError(
                    "Threshold.interval_method must be 'exact_bound' when censoring is RIGHT, "
                    f"got {self.interval_method!r}"
                )
            if self.value is not None:
                raise ValueError("Threshold.value must be None when censoring is RIGHT")
            if self.lo is None:
                raise ValueError("Threshold.lo must be set when censoring is RIGHT")
            if self.hi is not None:
                raise ValueError("Threshold.hi must be None when censoring is RIGHT")
        elif self.censoring is Censoring.LEFT:
            if self.interval_method != "exact_bound":
                raise ValueError(
                    "Threshold.interval_method must be 'exact_bound' when censoring is LEFT, "
                    f"got {self.interval_method!r}"
                )
            if self.value is not None:
                raise ValueError("Threshold.value must be None when censoring is LEFT")
            if self.hi is None:
                raise ValueError("Threshold.hi must be set when censoring is LEFT")
            if self.lo is not None:
                raise ValueError("Threshold.lo must be None when censoring is LEFT")
        elif self.censoring is Censoring.OPEN_UPPER:
            if self.status is not Status.OK:
                raise ValueError("Threshold.censoring OPEN_UPPER requires status OK")
            if self.interval_method != "profile":
                raise ValueError(
                    "Threshold.interval_method must be 'profile' when censoring is OPEN_UPPER, "
                    f"got {self.interval_method!r}"
                )
            if self.value is None or self.lo is None:
                raise ValueError("Threshold.value and .lo must be set when censoring is OPEN_UPPER")
            if self.hi is not None:
                raise ValueError("Threshold.hi must be None when censoring is OPEN_UPPER")
            if not self.lo <= self.value:
                raise ValueError(
                    "Threshold requires lo <= value when censoring is OPEN_UPPER, got "
                    f"lo={self.lo!r}, value={self.value!r}"
                )
        elif self.censoring is Censoring.OPEN_LOWER:
            if self.status is not Status.OK:
                raise ValueError("Threshold.censoring OPEN_LOWER requires status OK")
            if self.interval_method != "profile":
                raise ValueError(
                    "Threshold.interval_method must be 'profile' when censoring is OPEN_LOWER, "
                    f"got {self.interval_method!r}"
                )
            if self.value is None or self.hi is None:
                raise ValueError("Threshold.value and .hi must be set when censoring is OPEN_LOWER")
            if self.lo is not None:
                raise ValueError("Threshold.lo must be None when censoring is OPEN_LOWER")
            if not self.value <= self.hi:
                raise ValueError(
                    "Threshold requires value <= hi when censoring is OPEN_LOWER, got "
                    f"value={self.value!r}, hi={self.hi!r}"
                )
        elif self.censoring is Censoring.NONE and self.status is Status.OK:
            # `value` is still required. `lo`/`hi` are not: a converged fit on a design too
            # coarse to constrain the parameter can have a profile that crosses the chi-square
            # target on neither side, and v1 has no `Censoring` member for "open on both sides"
            # (`OPEN_UPPER`/`OPEN_LOWER` each require their closed side to exist). Reporting the
            # estimate with no bounds, and the reason in `warnings`, is the honest encoding;
            # inventing a bound at the search limit would not be. Loosening an invariant is a
            # safe (additive) change, but a card that relies on it needs a reader at least as
            # new as the writer.
            if self.value is None:
                raise ValueError(
                    "Threshold.value must be set when censoring is NONE and status is OK"
                )
            if (self.lo is None) != (self.hi is None):
                raise ValueError(
                    "Threshold.lo and .hi must either both be set or both be None when "
                    f"censoring is NONE and status is OK, got lo={self.lo!r}, hi={self.hi!r}"
                )
            if (
                self.lo is not None
                and self.hi is not None
                and not (self.lo <= self.value <= self.hi)
            ):
                raise ValueError(
                    f"Threshold requires lo <= value <= hi, got lo={self.lo!r}, "
                    f"value={self.value!r}, hi={self.hi!r}"
                )
            if self.interval_method not in ("profile", "delta"):
                raise ValueError(
                    "Threshold.interval_method must be 'profile' or 'delta' when censoring is "
                    f"NONE and status is OK, got {self.interval_method!r}"
                )

        # baseline/grid only ever *tighten* -- an existing invariant above is never weakened to
        # admit them. Each is legal only alongside the status it documents; neither is required
        # by that status (a threshold built without them has both `None`).
        if self.baseline is not None and self.status is not Status.FAILS_AT_BASELINE:
            raise ValueError(
                "Threshold.baseline may only be set when status is FAILS_AT_BASELINE, got "
                f"status={self.status!r}"
            )
        if self.grid is not None and self.status not in _NONCONVERGENT_STATUSES:
            raise ValueError(
                "Threshold.grid may only be set when status is SEPARATION or NOT_CONVERGED, got "
                f"status={self.status!r}"
            )


_VALID_INTERVAL_METHOD_REQUESTS = frozenset({"profile", "delta"})
_VALID_DEPENDENCE_INPUTS = frozenset({"independent"})


def _control_cell_or_none(fit: Fit) -> Cell | None:
    for cell in fit.cells:
        if cell.severity == 0.0:
            return cell
    return None


def _resolve_target(definition: Definition, *, fit: Fit) -> tuple[float, str | None]:
    """The already-resolved target performance ``P*``, on the scale of the outcome as counted,
    plus an optional warning naming where an asymptote came from.

    ``absolute(a)`` never needs an asymptote at all: ``P* = a``. The other definitions need the
    asymptotes: the fitted values when ``fit.status is Status.OK``, or -- since there are no
    fitted asymptotes on the exact-bound fallback path -- the zero-severity control cell's own
    observed rate in place of the upper asymptote for ``baseline_fraction``, with a warning naming
    that substitution. ``baseline_fraction`` belongs to a decreasing fit and ``baseline_multiple``
    to an increasing one; the other pairing raises. A baseline at or below zero has no multiple, so
    ``baseline_multiple`` raises for a ``lower`` fixed at zero. It also raises on any failed fit,
    where there is no fitted baseline and a multiple of the control's observed rate would be an
    uncovered target, and on a linear axis, where the lower asymptote is not the performance at
    zero severity.

    ``relative(p)`` is the fraction ``p`` of the dynamic range traversed from the zero-severity
    end: ``u - p * (u - l)`` for a decreasing fit and ``l + p * (u - l)`` for an increasing one.
    """
    increasing = fit.direction == "increasing"
    if definition.kind == "baseline_fraction" and increasing:
        raise ValueError(
            "threshold: definition.kind='baseline_fraction' falls to a fraction of the "
            "zero-severity performance, which only a decreasing outcome does; this fit is "
            "increasing. Use baseline_multiple(k) or absolute(a)"
        )
    if definition.kind == "baseline_multiple" and not increasing:
        raise ValueError(
            "threshold: definition.kind='baseline_multiple' rises to a multiple of the "
            "zero-severity performance, which only an increasing outcome does; this fit is "
            "decreasing. Use baseline_fraction(p) or absolute(a)"
        )

    if definition.kind == "baseline_multiple" and fit.axis.scale != "log":
        raise ValueError(
            "threshold: definition.kind='baseline_multiple' is defined on a log axis only. On a "
            "linear axis the lower asymptote is not the performance at zero severity, so a "
            "multiple of it is not 'k times the performance at severity 0'. Use absolute(a)"
        )

    if definition.kind == "absolute":
        return definition.value, None

    if fit.status is Status.OK:
        assert fit.params is not None
        upper = fit.params["upper"].value
        lower = fit.params["lower"].value
        if definition.kind == "baseline_multiple" and fit.params["lower"].fixed and lower <= 0.0:
            raise ValueError(
                "threshold: definition.kind='baseline_multiple' needs a baseline above zero, and "
                f"this fit's lower asymptote (the rate at zero severity) is fixed at {lower!r}; a "
                'multiple of zero is not a threshold. Refit with lower="estimate" or a fixed '
                "lower above zero"
            )
        warning = None
    else:
        if definition.kind == "relative":
            # `relative(p)` needs BOTH asymptotes. When the fit failed, one comes from the control
            # cell's observed rate, but there is no source for the other, because nothing in the
            # data speaks to the far asymptote.
            # Substituting `0` for it would silently turn `relative(p)` into
            # `baseline_fraction(1 - p)` and, whenever the true asymptote is elsewhere, aim at a
            # target that is easier to clear -- an optimistic threshold produced on a failed fit
            # by an assumption nothing in the data supports, with no warning naming it.
            # Refuse instead, as this function already refuses when there is no control cell to
            # supply the other.
            anchored = "baseline_multiple(k)" if increasing else "baseline_fraction(p)"
            raise ValueError(
                f"threshold: definition.kind='relative' needs both asymptotes to resolve a "
                f"target, and fit.status is {fit.status!r} (no fitted asymptotes). There is no "
                "defensible fallback for the far asymptote -- nothing in the data speaks to "
                "it -- so no target is resolved. Use absolute(a), which needs no asymptote, or "
                f"{anchored}, which needs only the control rate"
            )
        control = _control_cell_or_none(fit)
        anchor = "lower" if increasing else "upper"
        if control is None:
            raise ValueError(
                f"threshold: definition.kind={definition.kind!r} needs the {anchor} asymptote to "
                f"resolve a target, fit.status is {fit.status!r} (no fitted asymptotes), and "
                "there is no zero-severity control cell to fall back to"
            )
        if definition.kind == "baseline_multiple":
            raise ValueError(
                "threshold: definition.kind='baseline_multiple' has no target on a failed fit "
                f"(fit.status is {fit.status.value}): the control's observed rate "
                f"{control.successes}/{control.trials} is only an estimate of the rate at zero "
                "severity, and exact claims about a multiple of an estimate carry no coverage "
                "guarantee. Use absolute(a), or refit so that the fit converges, for example "
                "with lower fixed above zero"
            )
        upper = lower = control.successes / control.trials
        warning = (
            f"threshold: fit.status is {fit.status!r} (no fitted asymptotes); the target for "
            f"definition.kind={definition.kind!r} was resolved from the zero-severity control "
            f"cell's observed rate ({control.successes}/{control.trials} = {upper!r}), not a "
            f"fitted {anchor} asymptote"
        )

    if definition.kind == "baseline_fraction":
        return definition.value * upper, warning
    if definition.kind == "baseline_multiple":
        return definition.value * lower, warning
    if increasing:
        return lower + definition.value * (upper - lower), warning  # relative
    return upper - definition.value * (upper - lower), warning  # relative


@dataclass(frozen=True)
class _SolveInputs:
    """What the censoring rules, the closed-form solve and the interval methods are run on: a
    decreasing fit, the definition and target to solve for on it, and -- only when that fit is the
    complementary view of an increasing one -- the target as the caller stated it."""

    fit: Fit
    definition: Definition
    target: float
    original_target: float | None


def _solve_inputs(definition: Definition, *, fit: Fit) -> tuple[_SolveInputs, str | None]:
    """Resolve the target for ``fit`` and put the problem in the form every downstream rule is
    written for. A decreasing fit passes through untouched. An increasing fit is replaced by its
    complementary decreasing view (:func:`robustmargin.models._mirror_fit`): performance ``1 - P``,
    so ``absolute(a)`` becomes ``absolute(1 - a)``, ``relative(p)`` is unchanged, and
    ``baseline_multiple(k)`` is carried as it is -- its mirrored target ``1 - k * (1 - u')`` is
    computed from the candidate asymptotes by the interval code. Severity is never mirrored.

    Returns the inputs and the substitution warning of :func:`_resolve_target`, if any.
    """
    target, warning = _resolve_target(definition, fit=fit)
    if fit.direction == "decreasing":
        return _SolveInputs(fit, definition, target, None), warning
    view_definition = (
        Definition.absolute(1.0 - definition.value) if definition.kind == "absolute" else definition
    )
    return _SolveInputs(_mirror_fit(fit), view_definition, 1.0 - target, target), warning


def _reported_baseline(
    baseline: BaselineRate | None, *, fit: Fit, level: float
) -> BaselineRate | None:
    """``baseline`` as the control's counts of the outcome as supplied. The classification reads
    the complementary view of an increasing fit, where the control's successes are its non-events;
    the rate and its exact interval are recomputed from the original counts, which is the same
    interval reflected about one half."""
    if baseline is None or fit.direction == "decreasing":
        return baseline
    control = _control_cell_or_none(fit)
    assert control is not None  # a baseline is only built from a control cell
    return _baseline_rate(control, level=level)


def _extrapolation_warning(fit: Fit, *, value: float) -> str | None:
    """The report-only ``"EXTRAPOLATED: "`` warning, or ``None`` when it does not apply.

    It applies to a solved ``value`` that is positive and lies strictly outside the range of the
    nonzero tested severities: strictly below the lowest, or strictly above the highest. The
    control is not the bottom of the range, since every value is above severity 0, and a value
    exactly at either end is inside it. It describes ``value`` only, never an interval endpoint,
    and it changes no number: it records that the fitted curve was read where no severity was
    tested.
    """
    nonzero = [cell.severity for cell in fit.cells if cell.severity > 0.0]
    if not nonzero or not value > 0.0:
        return None
    lowest, highest = min(nonzero), max(nonzero)
    if value < lowest:
        side, where, endpoint = "below", "lowest", lowest
    elif value > highest:
        side, where, endpoint = "above", "highest", highest
    else:
        return None
    return (
        f"EXTRAPOLATED: threshold: the reported value {value!r} lies {side} the {where} nonzero "
        f"tested severity {endpoint!r}; it is the fitted curve's crossing read outside the range "
        "the data covers, not a severity the tested levels bracket"
    )


def threshold(
    fit: Fit,
    *,
    definition: Definition,
    interval_method: str,
    level: float,
    dependence: str | None = None,
) -> Threshold:
    """Solve for where ``fit``'s curve crosses ``definition``'s target performance, with a
    confidence interval, or report why no value exists.

    An increasing fit is solved on its complementary decreasing view: the same censoring rules,
    exact tests and interval methods run on performance ``1 - P`` with the target complemented,
    and the result is read back on the outcome as counted (see :class:`Threshold`).

    ``interval_method`` is a *request*, not a promise: every censored or failed result records
    ``"exact_bound"`` regardless of what was requested, because that is the method that actually
    produced its numbers. The order in which the outcomes below are checked is
    :mod:`robustmargin.censoring`'s module docstring, not this one's.

    Parameters
    ----------
    fit
        The fit to solve a threshold from.
    definition
        Which target performance to solve for.
    interval_method
        ``"profile"`` (primary) or ``"delta"`` (cross-check). Required, with no default.
    level
        The confidence level, strictly between 0 and 1.
    dependence
        ``None`` (the default) resolves to ``"independent"`` unless ``fit.cluster_ids`` is set,
        in which case omitting it raises (a method assuming independence must refuse clustered
        input unless the caller explicitly opts in). ``"independent"`` may always be passed
        explicitly. ``"paired"`` raises, naming that it is not implemented in this release. The
        resulting ``Threshold.dependence`` is always ``"independent"`` in this release.

    Returns
    -------
    Threshold

    Raises
    ------
    ValueError
        If ``interval_method`` or ``level`` is invalid; if ``dependence`` is omitted while
        ``fit.cluster_ids`` is set, or is any value other than ``None``/``"independent"``/
        ``"paired"``; if ``dependence="paired"`` is requested (not yet implemented); if
        ``definition.kind`` is ``"baseline_fraction"`` and the fit is increasing, or
        ``"baseline_multiple"`` and the fit is decreasing; if ``definition.kind`` is
        ``"baseline_multiple"`` and the fit is ``OK`` with ``lower`` fixed at or below zero (a
        multiple of zero is not a threshold), or the fit failed, or the axis is linear; if
        ``definition.kind`` is ``"relative"`` and ``fit.status`` is not ``OK``; or if
        ``definition.kind`` is ``"baseline_fraction"`` or ``"baseline_multiple"``,
        ``fit.status`` is not ``OK``, and no zero-severity control cell exists to resolve a target
        from.
    """
    if interval_method not in _VALID_INTERVAL_METHOD_REQUESTS:
        raise ValueError(
            "threshold: interval_method must be one of "
            f"{sorted(_VALID_INTERVAL_METHOD_REQUESTS)}, got {interval_method!r}"
        )
    _check_finite_number(level, label="threshold level")
    if not (0.0 < level < 1.0):
        raise ValueError(f"threshold: level must satisfy 0 < level < 1, got {level!r}")

    if dependence == "paired":
        raise ValueError(
            "threshold: dependence='paired' is not implemented in this release; pass "
            "dependence='independent' explicitly to acknowledge independence is being assumed "
            "anyway, or wait for a future release that adds it"
        )
    if dependence is not None and dependence not in _VALID_DEPENDENCE_INPUTS:
        raise ValueError(
            "threshold: dependence must be None, 'independent', or (not yet implemented) "
            f"'paired', got {dependence!r}"
        )
    if fit.cluster_ids is not None and dependence is None:
        raise ValueError(
            "threshold: fit carries cluster ids (fit.cluster_ids is set); a method that assumes "
            "independence must not run on clustered data unless the caller passes "
            "dependence='independent' explicitly"
        )

    inputs, target_warning = _solve_inputs(definition, fit=fit)
    collected_warnings: list[str] = [] if target_warning is None else [target_warning]

    # Applies whatever the resulting censoring or status, because every threshold rule and every
    # fitted model assume the true performance curve is monotone in severity. Never changes
    # value/lo/hi/status/censoring -- only adds a warning.
    if fit.monotonicity is not None and fit.monotonicity.any_flagged:
        flagged_pairs = ", ".join(
            f"({pair.severity_low!r}, {pair.severity_high!r})"
            for pair in fit.monotonicity.pairs
            if pair.flagged
        )
        collected_warnings.append(
            "NON_MONOTONE_DATA: threshold: fit.monotonicity flags an adjacent-level reversal "
            f"at severity pair(s) {flagged_pairs}; every threshold rule and every fitted model "
            "here assume the true performance curve is monotone in severity"
        )

    view = inputs.fit
    classification = classify(
        view, target=inputs.target, level=level, original_target=inputs.original_target
    )
    collected_warnings.extend(classification.warnings)

    if classification.solve:
        theta_hat, value = solve_value(view, target=inputs.target)
        if value <= 0.0:
            # A linear axis can solve to a crossing at or below zero severity. Severity is
            # non-negative (`Threshold` enforces it), so such a crossing is not reachable on the
            # axis as tested. It is reported as UNREACHABLE with a warning that names the solved
            # value, so it is distinguishable from the asymptote-driven UNREACHABLE. Allowing
            # negative thresholds later is a loosening, and therefore safe.
            already = "above" if fit.direction == "increasing" else "below"
            collected_warnings.append(
                f"threshold: the fitted curve is already {already} the target at severity 0 -- it "
                f"crosses at {value!r}, at or below the axis origin, so on this axis the "
                "criterion fails before any severity is applied. Reported as UNREACHABLE "
                "because this release's severity axis is non-negative and there is no "
                "model-based equivalent of FAILS_AT_BASELINE for this case"
            )
            return Threshold(
                axis=fit.axis,
                definition=definition,
                direction=fit.direction,
                value=None,
                lo=None,
                hi=None,
                level=level,
                interval_method="exact_bound",
                censoring=Censoring.NONE,
                status=Status.UNREACHABLE,
                dependence="independent",
                fit=fit,
                warnings=tuple(collected_warnings),
                baseline=None,
                grid=None,
            )
        if interval_method == "profile":
            interval_result = profile_interval(
                view,
                definition=inputs.definition,
                target=inputs.target,
                level=level,
                theta_hat=theta_hat,
            )
        else:
            interval_result = delta_interval(
                view,
                definition=inputs.definition,
                target=inputs.target,
                level=level,
                theta_hat=theta_hat,
                value=value,
            )
        collected_warnings.extend(interval_result.warnings)
        # Report-only and last: it describes `value` alone, whatever the interval method or its
        # outcome, and changes no number, status, censoring or method.
        extrapolation = _extrapolation_warning(fit, value=value)
        if extrapolation is not None:
            collected_warnings.append(extrapolation)
        return Threshold(
            axis=fit.axis,
            definition=definition,
            direction=fit.direction,
            value=value,
            lo=interval_result.lo,
            hi=interval_result.hi,
            level=level,
            interval_method=interval_method,
            censoring=interval_result.censoring,
            status=Status.OK,
            dependence="independent",
            fit=fit,
            warnings=tuple(collected_warnings),
            baseline=None,
            grid=None,
        )

    return Threshold(
        axis=fit.axis,
        definition=definition,
        direction=fit.direction,
        value=None,
        lo=classification.lo,
        hi=classification.hi,
        level=level,
        interval_method="exact_bound",
        censoring=classification.censoring,
        status=classification.status,
        dependence="independent",
        fit=fit,
        warnings=tuple(collected_warnings),
        baseline=_reported_baseline(classification.baseline, fit=fit, level=level),
        grid=classification.grid,
    )
