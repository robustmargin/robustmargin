"""robustmargin's shared vocabulary for severity and outcome data.

``Censoring`` is the model-based and grid-rule censoring classification (documented on the
class itself). Everything else here is new to robustmargin: ``Axis`` describes the
severity axis a series of observations was measured on; ``Observations`` holds the raw
aggregated-or-per-observation data a fit is built from; ``Definition`` names which of three
target-performance definitions a threshold solves for; ``Cell`` is one pooled row of counts at
a single severity; ``Status`` and ``IntervalShape`` are the enums results report through.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "Axis",
    "Cell",
    "Censoring",
    "Definition",
    "IntervalShape",
    "JSONValue",
    "Observations",
    "Status",
]

# A JSON value as this package writes and reads it: the recursive union `report.py` validates
# provenance mappings against and the schema-loading code returns. Not itself validated here --
# `report.to_dict` is what actually walks a provenance mapping and rejects anything outside
# this union (str keys only, no tuple, no set, no NaN).
JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class Censoring(StrEnum):
    """How a result's tested range relates to a criterion.

    Every member is a ``str`` subclass whose value equals its member name (``Censoring.RIGHT
    == "RIGHT"``). That value is the literal string a serialised result writes for this field,
    not an incidental implementation detail.

    **Model-based thresholds** use the full classification. It assumes performance is
    monotone in severity, and a result that uses it says so. ``RIGHT`` and ``LEFT`` are decided
    by exact one-sided tests at the edge of the tested range, not by point estimates.

    **The grid rule** (:func:`robustmargin.grid_break_point`) uses only ``NONE`` and ``RIGHT``, and
    only as point-estimate labels, with no exact test and no monotonicity assumption. Their
    meaning for a grid result is documented on :class:`robustmargin.GridBreakPoint` and is weaker
    than the definitions below.

    Attributes
    ----------
    NONE
        Model-based threshold: none of the conditions below applies, so a value is reported
        with both interval sides closed. Grid result: at least one tested level failed. This
        does not imply the criterion was bracketed.
    RIGHT
        Model-based threshold: the exact one-sided bound at the highest tested level shows the
        criterion still holds there. No value is reported. The top of the tested range is
        reported as a one-sided lower bound ("held up to the highest level tested") and is
        never phrased as "does not fail" or "unbreakable". Grid result: no tested level's
        performance fell below the criterion.
    LEFT
        Model-based threshold: the exact one-sided bound at the lowest nonzero tested level
        shows it already fails, while the control passes. No value is reported. The lowest
        nonzero level is reported as a one-sided upper bound. Never produced by the grid rule.
    OPEN_UPPER
        Model-based threshold: the value lies inside the tested range, but the upper side of
        its profile interval does not close within the search limit. The value and the closed
        lower side are still reported. Never produced by the grid rule.
    OPEN_LOWER
        The mirror of ``OPEN_UPPER``: the lower side of the profile interval does not close
        within the search limit. The value and the closed upper side are still reported. Never
        produced by the grid rule.
    """

    NONE = "NONE"
    RIGHT = "RIGHT"
    LEFT = "LEFT"
    OPEN_UPPER = "OPEN_UPPER"
    OPEN_LOWER = "OPEN_LOWER"


class Status(StrEnum):
    """How a fit or a threshold derived from it stands, following ``Censoring``: each member's
    value equals its name.

    Attributes
    ----------
    OK
        The fit converged and the threshold, if any, was solved for.
    SEPARATION
        The data are completely or quasi-completely separated: some or all outcomes are
        perfectly predicted by severity, so the likelihood has no finite maximum on at least
        one parameter. No model-based interval is produced; a threshold built from this fit
        falls back to the exact censoring bound.
    NOT_CONVERGED
        The optimizer did not converge, or converged to a point where the Hessian is not
        positive-definite. Same fallback as ``SEPARATION``.
    UNREACHABLE
        No severity solves for the target performance implied by a threshold's
        ``Definition``. Either the target lies at or below 0 or at or above 1, which no
        binomial performance curve crosses, whatever the fit's status; or, on a converged fit,
        it lies outside the fitted curve's range (at or beyond one of the asymptotes); or, on a
        linear axis, the fitted curve crosses it only at or below zero severity.
    FAILS_AT_BASELINE
        The zero-severity control, and the lowest nonzero severity when there is one, both
        confidently fail the criterion by exact one-sided tests, and no nonzero severity
        confidently passes it: the threshold does not exist because the system was never
        passing to begin with.
    CONTROL_INCOMPATIBLE
        The upper asymptote is fixed (commonly at 1) but the zero-severity control has
        failures, which drives the likelihood to zero for that fixed value. Refit with
        ``upper="estimate"``.
    """

    OK = "OK"
    SEPARATION = "SEPARATION"
    NOT_CONVERGED = "NOT_CONVERGED"
    UNREACHABLE = "UNREACHABLE"
    FAILS_AT_BASELINE = "FAILS_AT_BASELINE"
    CONTROL_INCOMPATIBLE = "CONTROL_INCOMPATIBLE"


class IntervalShape(StrEnum):
    """The geometry of a ratio's confidence interval on the real line, following ``Censoring``:
    each member's value equals its name.

    A Fieller interval for the ratio of two normal-ish quantities is not always a bounded
    interval; which shape it takes depends on how much relative uncertainty the denominator
    threshold carries.

    Attributes
    ----------
    BOUNDED
        The usual case: ``[lo, hi]``, a single closed interval. Also covers a Fieller set bounded
        above but not below on its own: intersected with the positive parameter space that is
        ``[0, hi]``, and ``0`` is a true lower bound for a severity ratio, so ``BOUNDED`` --
        never ``HALF_OPEN`` -- is reported (the upper-only geometry stays ``BOUNDED``).
        ``HALF_OPEN`` is therefore always lower-bounded, never upper-bounded, and that asymmetry
        is deliberate.
    UNBOUNDED
        The confidence set, intersected with the positive parameter space, is the whole of
        ``(0, inf)``; ``lo`` and ``hi`` are both ``None``.
    EXCLUSIVE
        Two disjoint rays, ``(-inf, lo] union [hi, inf)``, with both ``lo`` and ``hi`` on the
        same side of zero (both negative is the ``UNBOUNDED`` case; a positive ``lo`` never
        arises under ``dependence="independent"``). **Unreachable in v0.1**: under
        ``dependence="independent"`` the covariance term is 0, which forces opposite-signed roots
        whenever this branch's other conditions hold, so the real result is always ``HALF_OPEN``
        instead. The member and :class:`~robustmargin.Ratio`'s invariants for it stay enforced
        against v0.2's ``dependence="paired"``, which can make the covariance term nonzero and
        reach this shape for the first time.
    HALF_OPEN
        One ray, ``[lo, inf)``: ``lo`` is set, ``hi`` is ``None``. This is what a two-ray
        Fieller set with ``K > 0`` is once intersected with the positive parameter space. It arises
        whenever the denominator threshold is poorly determined (``z^2*Vb > theta_b^2``) while
        the numerator is not -- an open region of the parameter space, not a measure-zero
        boundary.
    """

    BOUNDED = "BOUNDED"
    UNBOUNDED = "UNBOUNDED"
    EXCLUSIVE = "EXCLUSIVE"
    HALF_OPEN = "HALF_OPEN"


_VALID_SCALES = frozenset({"log", "linear"})
_VALID_DIRECTIONS = frozenset({"decreasing", "increasing"})
_VALID_DEFINITION_KINDS = frozenset({"absolute", "baseline_fraction", "relative"})


def _check_finite_number(value: float, *, label: str) -> None:
    """Reject a bool masquerading as a number, and reject a non-finite value (``nan``,
    ``inf`` or ``-inf``).

    Every numeric field across the package that must be a genuine finite number --
    ``Definition.value``, ``Cell.severity``, ``Observations.severity``, ``Fit.log_likelihood``,
    ``Threshold``'s ``value``/``lo``/``hi``/``level``, ``Ratio``'s ``estimate``/``lo``/``hi``/
    ``level``, and every ``Covariance.matrix`` entry -- routes its bool/finite check through
    this one function, so the rule cannot silently drift between them.
    """
    if isinstance(value, bool):
        raise ValueError(f"{label} must not be a bool, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")


def _as_whole_int(value: float, *, label: str) -> int:
    """Coerce a whole-number ``int`` or ``float`` (for example ``200`` or ``200.0``) to a
    genuine ``int``. A bool, a non-finite value, or a non-whole float (``2.5``) raises
    ``ValueError``.
    """
    _check_finite_number(value, label=label)
    as_int = int(value)
    if as_int != value:
        raise ValueError(f"{label} must be a whole number, got {value!r}")
    return as_int


@dataclass(frozen=True, kw_only=True)
class Axis:
    """The severity axis a series of observations was measured on.

    Attributes
    ----------
    name
        A short, non-empty, domain-neutral name for the axis (for example ``"noise"``).
    unit
        The unit severity is expressed in (for example ``"m"``). Non-empty. A ratio between two
        thresholds is allowed only when both axes match on ``name`` and ``unit``.
    scale
        Either ``"log"`` or ``"linear"``: which scale the dose-response model is fit on.
    citation
        An optional free-text citation for where the axis or its unit convention comes from (for
        example a datasheet reference). ``None`` by default; if given, must be non-empty.
    """

    name: str
    unit: str
    scale: str
    citation: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Axis.name must be non-empty (after stripping whitespace)")
        if not self.unit.strip():
            raise ValueError("Axis.unit must be non-empty (after stripping whitespace)")
        if self.scale not in _VALID_SCALES:
            raise ValueError(
                f"Axis.scale must be one of {sorted(_VALID_SCALES)}, got {self.scale!r}"
            )
        if self.citation is not None and not self.citation.strip():
            raise ValueError("Axis.citation, if given, must be non-empty")


@dataclass(frozen=True, kw_only=True)
class Definition:
    """Which target performance a threshold solves for.

    Three definitions are in play, and they coincide only when the fitted curve's asymptotes
    are exactly 1 and 0:

    - ``absolute(a)``: performance crosses a fixed level ``a``.
      Whether ``a`` is reachable given the fitted asymptotes is decided at threshold-solving
      time (:data:`Status.UNREACHABLE`), not at construction time, so any finite ``a`` --
      including one that looks out of range, such as 1.5 -- constructs successfully. An ``a``
      at or below 0 or at or above 1 is :data:`Status.UNREACHABLE` on every fit.
    - ``baseline_fraction(p)``: performance falls to ``p`` times the performance at zero
      severity. Requires ``0 < p < 1``.
    - ``relative(p)``: a fraction ``p`` of the fitted dynamic range (upper minus lower
      asymptote) is lost (R ``drc``'s ``type = "relative"``). Requires ``0 < p < 1``.

    Attributes
    ----------
    kind
        One of ``"absolute"``, ``"baseline_fraction"``, ``"relative"``.
    value
        The threshold parameter: ``a`` for ``absolute``, ``p`` for the other two. Must be
        finite.
    """

    kind: str
    value: float

    def __post_init__(self) -> None:
        if self.kind not in _VALID_DEFINITION_KINDS:
            raise ValueError(
                f"Definition.kind must be one of {sorted(_VALID_DEFINITION_KINDS)}, "
                f"got {self.kind!r}"
            )
        _check_finite_number(self.value, label="Definition.value")
        if self.kind in ("baseline_fraction", "relative") and not (0.0 < self.value < 1.0):
            raise ValueError(
                f"Definition.value for kind={self.kind!r} must satisfy 0 < value < 1, "
                f"got {self.value!r}"
            )

    @classmethod
    def absolute(cls, a: float, /) -> Definition:
        """Build an ``absolute(a)`` definition: performance crosses the fixed level ``a``."""
        return cls(kind="absolute", value=a)

    @classmethod
    def baseline_fraction(cls, p: float, /) -> Definition:
        """Build a ``baseline_fraction(p)`` definition: performance falls to ``p`` times the
        zero-severity performance. Requires ``0 < p < 1``."""
        return cls(kind="baseline_fraction", value=p)

    @classmethod
    def relative(cls, p: float, /) -> Definition:
        """Build a ``relative(p)`` definition: a fraction ``p`` of the fitted dynamic range is
        lost. Requires ``0 < p < 1``."""
        return cls(kind="relative", value=p)


def _check_counts(
    *, severity: float, successes: float, trials: float, label: str
) -> tuple[int, int]:
    """Validate one ``(severity, successes, trials)`` row and return ``successes``/``trials``
    coerced to ``int``.

    ``successes`` and ``trials`` each accept a whole-number ``float`` (for example ``200.0``)
    as well as a genuine ``int``: a JSON-Schema "integer" cannot tell ``200.0`` from ``200``, so
    both direct construction and ``from_dict`` must accept it. A bool or a
    non-whole float (``2.5``) still raises. The coerced ``int`` values are what gets stored on
    the dataclass, never the original float, so ``type(cell.successes) is int`` always holds.
    """
    _check_finite_number(severity, label=f"{label}.severity")
    successes_int = _as_whole_int(successes, label=f"{label}.successes")
    trials_int = _as_whole_int(trials, label=f"{label}.trials")
    if severity < 0.0:
        raise ValueError(f"{label}.severity must be non-negative, got {severity!r}")
    if trials_int < 1:
        raise ValueError(f"{label}.trials must be >= 1, got {trials_int!r}")
    if successes_int < 0:
        raise ValueError(f"{label}.successes must be non-negative, got {successes_int!r}")
    if successes_int > trials_int:
        raise ValueError(
            f"{label}.successes ({successes_int!r}) must not exceed {label}.trials ({trials_int!r})"
        )
    return successes_int, trials_int


@dataclass(frozen=True, kw_only=True)
class Cell:
    """One pooled row of counts at a single severity: the unit :meth:`Observations.cells`
    returns and :class:`robustmargin.Fit` stores its per-level table as.

    Attributes
    ----------
    severity
        The (non-negative, finite) severity level this cell was pooled at. Must not be a
        ``bool``.
    successes
        The number of successes observed at this level. Accepts a whole-number ``float`` (for
        example ``200.0``) as well as an ``int``; either way, the stored value is a genuine
        ``int`` (``type(cell.successes) is int``). A ``bool`` or a non-whole float (``2.5``)
        raises.
    trials
        The number of trials observed at this level. Must be at least 1, and at least
        ``successes``. Accepts a whole-number ``float`` the same way ``successes`` does.
    """

    severity: float
    successes: int
    trials: int

    def __post_init__(self) -> None:
        successes, trials = _check_counts(
            severity=self.severity, successes=self.successes, trials=self.trials, label="Cell"
        )
        object.__setattr__(self, "successes", successes)
        object.__setattr__(self, "trials", trials)


@dataclass(frozen=True, kw_only=True)
class Observations:
    """Raw severity/outcome data for one series, either aggregated counts or per-observation
    rows -- both stored the same way, with each per-observation row as ``trials=1``.

    Severity is non-negative: a signed axis (for example a bidirectional mass offset) is split
    by the caller into two ``Observations``, one per side. On a log axis, a severity of
    exactly ``0.0`` is the zero-severity control; it is handled in the fitted likelihood,
    never by taking ``log(0)`` and never by an epsilon substitute.

    Attributes
    ----------
    axis
        The :class:`Axis` this series was measured on.
    outcome
        A short, non-empty, domain-neutral name for what "success" means (for example
        ``"success"`` or ``"consistent"``). Required, with no default: a wrong default would
        silently flip a threshold.
    direction
        Either ``"decreasing"`` (performance falls as severity rises) or ``"increasing"``.
        Required, with no default, for the same reason as ``outcome``.
    severity
        One entry per row, non-negative and finite. Must not be a ``bool``.
    successes
        One entry per row, a non-negative whole number not exceeding the matching ``trials``
        entry. Accepts a whole-number ``float`` (for example ``200.0``) as well as an ``int``;
        either way the stored value is a genuine ``int``. A ``bool`` or a non-whole float
        (``2.5``) raises.
    trials
        One entry per row, a whole number of at least 1. Accepts a whole-number ``float`` the
        same way ``successes`` does.
    cluster
        Optional cluster ids, one per row, each a ``str`` or non-bool ``int`` (never ``None``
        or a ``bool``). Any method assuming independence must refuse clustered input unless the
        caller explicitly asks for it; ``Observations`` itself never drops or ignores these
        ids.
    """

    axis: Axis
    outcome: str
    direction: str
    severity: tuple[float, ...]
    successes: tuple[int, ...]
    trials: tuple[int, ...]
    cluster: tuple[str | int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", tuple(self.severity))
        object.__setattr__(self, "successes", tuple(self.successes))
        object.__setattr__(self, "trials", tuple(self.trials))
        if self.cluster is not None:
            object.__setattr__(self, "cluster", tuple(self.cluster))

        if not self.outcome:
            raise ValueError("Observations.outcome must be non-empty")
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"Observations.direction must be one of {sorted(_VALID_DIRECTIONS)}, "
                f"got {self.direction!r}"
            )

        n = len(self.severity)
        if n == 0:
            raise ValueError("Observations must not be empty")
        if len(self.successes) != n or len(self.trials) != n:
            raise ValueError(
                "Observations.severity, .successes and .trials must have equal length: "
                f"got {n}, {len(self.successes)}, {len(self.trials)}"
            )

        # ``_check_counts`` rejects a bool or non-finite severity, and accepts a whole-number
        # float for successes/trials (for example ``200.0``) the same way direct ``Cell``
        # construction does, coercing both to ``int`` for storage.
        coerced_successes: list[int] = []
        coerced_trials: list[int] = []
        for severity, successes, trials in zip(
            self.severity, self.successes, self.trials, strict=True
        ):
            row_successes, row_trials = _check_counts(
                severity=severity, successes=successes, trials=trials, label="Observations"
            )
            coerced_successes.append(row_successes)
            coerced_trials.append(row_trials)
        object.__setattr__(self, "successes", tuple(coerced_successes))
        object.__setattr__(self, "trials", tuple(coerced_trials))

        if self.cluster is not None:
            if len(self.cluster) != n:
                raise ValueError(
                    f"Observations.cluster must have length {n} to match severity, "
                    f"got {len(self.cluster)}"
                )
            for cluster_id in self.cluster:
                if cluster_id is None or isinstance(cluster_id, bool):
                    raise ValueError(
                        "Observations.cluster ids must be a str or int, never None or bool, "
                        f"got {cluster_id!r}"
                    )
                if not isinstance(cluster_id, (str, int)):
                    raise ValueError(
                        f"Observations.cluster ids must be a str or int, got {cluster_id!r}"
                    )

    @classmethod
    def from_counts(
        cls,
        axis: Axis,
        *,
        severity: Sequence[float],
        successes: Sequence[float],
        trials: Sequence[float],
        outcome: str,
        direction: str,
        cluster: Sequence[str | int] | None = None,
    ) -> Observations:
        """Build ``Observations`` from already-aggregated ``(severity, successes, trials)``
        rows.

        ``successes`` and ``trials`` accept whole-number floats (for example ``200.0``) as well
        as ints; a non-whole value (``2.5``) raises.
        """
        return cls(
            axis=axis,
            outcome=outcome,
            direction=direction,
            severity=tuple(float(v) for v in severity),
            successes=tuple(_as_whole_int(v, label="successes") for v in successes),
            trials=tuple(_as_whole_int(v, label="trials") for v in trials),
            cluster=tuple(cluster) if cluster is not None else None,
        )

    @classmethod
    def from_observations(
        cls,
        axis: Axis,
        *,
        severity: Sequence[float],
        success: Sequence[float],
        outcome: str,
        direction: str,
        cluster: Sequence[str | int] | None = None,
    ) -> Observations:
        """Build ``Observations`` from one row per observation, each stored with ``trials=1``.

        ``success`` entries must be ``0``, ``1``, ``True`` or ``False``; anything else raises.
        """
        successes: list[int] = []
        for value in success:
            if isinstance(value, bool):
                successes.append(1 if value else 0)
            elif value in (0, 1):
                successes.append(int(value))
            else:
                raise ValueError(
                    f"Observations.from_observations: success entries must be 0, 1, True or "
                    f"False, got {value!r}"
                )
        return cls(
            axis=axis,
            outcome=outcome,
            direction=direction,
            severity=tuple(float(v) for v in severity),
            successes=tuple(successes),
            trials=tuple(1 for _ in successes),
            cluster=tuple(cluster) if cluster is not None else None,
        )

    def cells(self) -> tuple[Cell, ...]:
        """Pool rows with exactly equal severity by summing counts, sorted ascending.

        Returns
        -------
        tuple[Cell, ...]
            One :class:`Cell` per distinct severity value, ascending.
        """
        pooled: dict[float, tuple[int, int]] = {}
        for severity, successes, trials in zip(
            self.severity, self.successes, self.trials, strict=True
        ):
            existing_successes, existing_trials = pooled.get(severity, (0, 0))
            pooled[severity] = (existing_successes + successes, existing_trials + trials)
        return tuple(
            Cell(severity=severity, successes=successes, trials=trials)
            for severity, (successes, trials) in sorted(pooled.items())
        )

    @property
    def is_clustered(self) -> bool:
        """Whether this series carries cluster ids."""
        return self.cluster is not None
