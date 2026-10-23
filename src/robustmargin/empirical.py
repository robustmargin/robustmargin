"""The grid break-point rule: the smallest tested severity magnitude at which performance fails.

``grid_break_point`` returns the smallest tested severity magnitude at which performance falls
strictly below a criterion, or an explicit right-censored result when no tested level fails. It
is a statistic over the levels actually tested, not a fitted estimate. Empty input, non-finite
values, and negative severities (unless the caller passes ``signed=True``) raise rather than
returning a silent fallback.

This module holds only the grid rule. Model-based dose-response estimation lives in the other
modules of this package.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import binomtest

from robustmargin.types import Cell, Censoring, _check_finite_number

__all__ = [
    "BaselineRate",
    "ExactRates",
    "GridBreakPoint",
    "grid_break_point",
    "per_cell_clopper_pearson",
]


@dataclass(frozen=True)
class GridBreakPoint:
    """The result of :func:`grid_break_point`: an observed grid statistic, not a threshold.

    Attributes
    ----------
    value
        The smallest *tested* severity magnitude whose performance was strictly below the
        criterion, or ``None`` if no tested level failed. It is always one of the tested
        magnitudes, never a fitted or interpolated estimate of where performance crosses the
        criterion, and it carries no confidence interval.

        Read as a location for the crossing, it is only an **upper bound at grid resolution**.
        If performance were monotone, the crossing would lie somewhere in the half-open gap
        between the largest passing tested magnitude below ``value`` and ``value`` itself. When
        ``value`` equals the smallest tested magnitude (including ``0.0``), no passing level
        was observed below it at all. Monotonicity is neither assumed nor checked.
    max_tested
        The largest tested severity magnitude. Always reported, so a right-censored result can
        be read as "held up to ``max_tested``" rather than as an unqualified pass.
    censoring
        A **point-estimate label for the grid**, not the censoring classification a model-based
        threshold reports. :data:`~robustmargin.types.Censoring.RIGHT` means no tested level's
        performance fell below the criterion (``value is None``).
        :data:`~robustmargin.types.Censoring.NONE` means at least one did. No exact one-sided test
        is applied and no monotonicity is assumed, so ``RIGHT`` here is weaker than a
        statistical lower bound. ``NONE`` does not mean the criterion was bracketed: a zero
        level that already fails gives ``value == 0.0`` with ``NONE``. The grid rule never
        produces ``LEFT``, ``OPEN_UPPER``, or ``OPEN_LOWER``.
    schema_version
        The version of the serialised result schema this object belongs to. Defaults to
        ``"2"``, the version ``to_dict`` writes; ``from_dict`` also reads ``"1"``.
    provenance
        An opaque mapping the caller may attach (for example with :func:`dataclasses.replace`)
        to record where the inputs came from. robustmargin stores it and never interprets it.
        Empty by default.
    """

    value: float | None
    max_tested: float
    censoring: Censoring
    schema_version: str = "2"
    provenance: Mapping[str, object] = field(default_factory=dict)


def grid_break_point(
    severity: ArrayLike,
    performance: ArrayLike,
    *,
    criterion: float,
    signed: bool = False,
) -> GridBreakPoint:
    """Find the smallest tested severity magnitude whose performance fails a criterion.

    A level fails when its performance is strictly below ``criterion``. A level sitting exactly
    on the criterion passes. This is a statistic over the levels actually tested, not a fitted
    or interpolated estimate: the reported ``value`` is always one of the tested magnitudes, or
    ``None`` when the criterion held everywhere. A ``None`` value paired with
    :data:`~robustmargin.types.Censoring.RIGHT` means performance held up to ``max_tested``. It
    does not mean the system cannot be broken, only that breaking it was not observed within
    the range that was tested.

    Parameters
    ----------
    severity
        One-dimensional severity levels tested. Must be non-negative unless ``signed=True``.
    performance
        One-dimensional performance values, one per ``severity`` entry, on whatever scale the
        caller has already put them on (for example, a success rate).
    criterion
        The performance level a passing observation must reach or exceed. The comparison is
        strict: ``performance < criterion`` is a failure, ``performance == criterion`` is a
        pass.
    signed
        If ``False`` (the default), a negative ``severity`` entry is treated as caller error and
        raises. If ``True``, ``severity`` may be signed and the rule breaks on magnitude: the
        side that fails at the smallest ``abs(severity)`` determines ``value``.

    Returns
    -------
    GridBreakPoint
        ``value`` is the smallest tested failing magnitude, or ``None``. ``max_tested`` is the
        largest tested magnitude. ``censoring`` is the grid label described on
        :class:`GridBreakPoint`.

    Raises
    ------
    ValueError
        If ``severity`` or ``performance`` is empty, is not one-dimensional, or the two have
        mismatched lengths. Also if any value in ``severity``, ``performance``, or
        ``criterion`` is non-finite, or if ``severity`` contains a negative value while
        ``signed=False``.

    Notes
    -----
    The caller decides what "severity" and "performance" mean on a given axis, for example
    which nuisance factors were pooled to produce ``performance``. This rule performs no
    pooling and fits no model, and it neither assumes nor checks that performance is monotone
    in severity.

    All inputs are converted to float64 before comparing, so the comparison is exact in
    float64. A float32 performance of ``0.95`` is ``0.949999988...`` and therefore fails a
    criterion of ``0.95``. Integers above ``2**53`` lose precision in that conversion.
    """
    severity_arr = np.asarray(severity, dtype=np.float64)
    performance_arr = np.asarray(performance, dtype=np.float64)

    if severity_arr.ndim != 1:
        raise ValueError(f"severity must be 1-D, got shape {severity_arr.shape}")
    if performance_arr.ndim != 1:
        raise ValueError(f"performance must be 1-D, got shape {performance_arr.shape}")
    if severity_arr.size == 0:
        raise ValueError("severity and performance must not be empty")
    if severity_arr.size != performance_arr.size:
        raise ValueError(
            "severity and performance must have the same length, got "
            f"{severity_arr.size} and {performance_arr.size}"
        )
    if not np.all(np.isfinite(severity_arr)):
        raise ValueError("severity must be finite (no NaN or inf)")
    if not np.all(np.isfinite(performance_arr)):
        raise ValueError("performance must be finite (no NaN or inf)")
    if not math.isfinite(criterion):
        raise ValueError("criterion must be finite (no NaN or inf)")
    if not signed and bool(np.any(severity_arr < 0.0)):
        raise ValueError("severity must be non-negative unless signed=True")

    magnitude = np.abs(severity_arr)
    max_tested = float(np.max(magnitude))

    failing = magnitude[performance_arr < criterion]
    if failing.size == 0:
        return GridBreakPoint(value=None, max_tested=max_tested, censoring=Censoring.RIGHT)

    return GridBreakPoint(
        value=float(np.min(failing)), max_tested=max_tested, censoring=Censoring.NONE
    )


_VALID_RATE_METHOD = "per_cell_clopper_pearson"


def _check_rate(value: float, *, label: str) -> None:
    _check_finite_number(value, label=label)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{label} must lie in [0, 1], got {value!r}")


@dataclass(frozen=True, kw_only=True)
class BaselineRate:
    """The zero-severity control's exact rate, attached to a :class:`~robustmargin.Threshold` whose
    ``status`` is :data:`~robustmargin.Status.FAILS_AT_BASELINE`.

    Serialisable (unlike :class:`ExactRates`): a single control cell's rate, labelled with the
    method it came from, is exactly what a ``FAILS_AT_BASELINE`` result should report, and putting
    it in its own field keeps it out of ``Threshold.lo``/``.hi`` (a per-cell exact rate is never
    attached to a threshold).

    Attributes
    ----------
    severity
        The control's severity. ``0.0`` in every case this package produces.
    successes, trials
        The control cell's observed counts.
    rate
        ``successes / trials``.
    lo, hi
        The **two-sided** exact (Clopper-Pearson) interval at ``level``, as is specified for a
        reported per-cell rate. This is deliberately not the one-sided bound that decided the
        ``FAILS_AT_BASELINE`` classification: that asks "does this cell provably fail the
        target", a one-sided question, while this describes where the control rate lies.
    level
        The confidence of that interval. Carried here so the number is self-describing in a
        stored card rather than requiring a reader to look at the enclosing ``Threshold``.
    method
        Always ``"per_cell_clopper_pearson"``.
    """

    severity: float
    successes: int
    trials: int
    rate: float
    lo: float
    hi: float
    level: float = 0.95
    method: str = _VALID_RATE_METHOD

    def __post_init__(self) -> None:
        _check_finite_number(self.severity, label="BaselineRate.severity")
        if self.severity < 0.0:
            raise ValueError(f"BaselineRate.severity must be non-negative, got {self.severity!r}")
        if isinstance(self.successes, bool) or isinstance(self.trials, bool):
            raise ValueError("BaselineRate.successes and .trials must not be bool")
        if self.trials < 1:
            raise ValueError(f"BaselineRate.trials must be >= 1, got {self.trials!r}")
        if not (0 <= self.successes <= self.trials):
            raise ValueError(
                f"BaselineRate.successes ({self.successes!r}) must satisfy "
                f"0 <= successes <= trials ({self.trials!r})"
            )
        _check_rate(self.rate, label="BaselineRate.rate")
        _check_rate(self.lo, label="BaselineRate.lo")
        _check_rate(self.hi, label="BaselineRate.hi")
        _check_finite_number(self.level, label="BaselineRate.level")
        if not (0.0 < self.level < 1.0):
            raise ValueError(f"BaselineRate.level must satisfy 0 < level < 1, got {self.level!r}")
        if not self.lo <= self.rate <= self.hi:
            raise ValueError(
                f"BaselineRate requires lo <= rate <= hi, got lo={self.lo!r}, "
                f"rate={self.rate!r}, hi={self.hi!r}"
            )
        if self.method != _VALID_RATE_METHOD:
            raise ValueError(
                f"BaselineRate.method must be {_VALID_RATE_METHOD!r}, got {self.method!r}"
            )


@dataclass(frozen=True, kw_only=True)
class ExactRates:
    """Per-cell exact (Clopper-Pearson-style) rate bounds for every cell in a fit, read directly
    from :attr:`robustmargin.Fit.cells`.

    **Deliberately not serialisable.** This type is excluded from ``report.py``'s tagged classes
    and has no schema entry: ``report.to_dict`` raises ``TypeError`` on it. That is what
    structurally enforces the rule that per-cell rates are never attached to a threshold --
    there is no JSON shape for a caller to smuggle one into.

    Attributes
    ----------
    severity, rate, lo, hi
        Parallel tuples, one entry per input cell, in the same order.
    level
        The confidence of the two-sided exact interval each ``lo``/``hi`` pair was computed at.
    method
        Always ``"per_cell_clopper_pearson"``.
    """

    severity: tuple[float, ...]
    rate: tuple[float, ...]
    lo: tuple[float, ...]
    hi: tuple[float, ...]
    level: float = 0.95
    method: str = _VALID_RATE_METHOD

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", tuple(self.severity))
        object.__setattr__(self, "rate", tuple(self.rate))
        object.__setattr__(self, "lo", tuple(self.lo))
        object.__setattr__(self, "hi", tuple(self.hi))
        n = len(self.severity)
        if not (len(self.rate) == len(self.lo) == len(self.hi) == n):
            raise ValueError(
                "ExactRates.severity, .rate, .lo and .hi must have equal length, got "
                f"{n}, {len(self.rate)}, {len(self.lo)}, {len(self.hi)}"
            )
        _check_finite_number(self.level, label="ExactRates.level")
        if not (0.0 < self.level < 1.0):
            raise ValueError(f"ExactRates.level must satisfy 0 < level < 1, got {self.level!r}")
        if self.method != _VALID_RATE_METHOD:
            raise ValueError(
                f"ExactRates.method must be {_VALID_RATE_METHOD!r}, got {self.method!r}"
            )
        for severity, rate, lo, hi in zip(self.severity, self.rate, self.lo, self.hi, strict=True):
            _check_finite_number(severity, label="ExactRates.severity entry")
            _check_rate(rate, label="ExactRates.rate entry")
            _check_rate(lo, label="ExactRates.lo entry")
            _check_rate(hi, label="ExactRates.hi entry")
            if not lo <= rate <= hi:
                raise ValueError(
                    f"ExactRates requires lo <= rate <= hi per cell, got lo={lo!r}, "
                    f"rate={rate!r}, hi={hi!r}"
                )


def per_cell_clopper_pearson(cells: Sequence[Cell], *, level: float = 0.95) -> ExactRates:
    """The exact (Clopper-Pearson) rate and bounds for each cell in ``cells``, independently.

    Each cell's ``lo``/``hi`` are the **two-sided** exact interval at ``level``, putting
    ``(1 - level) / 2`` in each tail -- a reported per-cell rate is defined as
    ``binomtest(k, n).proportion_ci(confidence_level=..., method="exact")``, which is two-sided.

    That is deliberately **not** :func:`robustmargin.censoring.exact_one_sided_bound`, which
    answers the different, one-sided question a censoring classification asks ("does this cell
    provably fail the target?"). The two coexist on purpose and must not be conflated: reporting
    the one-sided pair under this function's name would label a roughly ``2 * level - 1`` region
    as ``level``, which is the mislabelling this separation exists to prevent.

    Parameters
    ----------
    cells
        The cells to compute rates for, typically ``fit.cells`` directly.
    level
        The confidence of the **two-sided** Clopper-Pearson interval, strictly between 0 and 1.
        Defaults to ``0.95``. A reported per-cell rate is defined as
        ``binomtest(k, n).proportion_ci(confidence_level=..., method="exact")``, which is
        two-sided -- *not* the one-sided bound that decides a censoring classification. For
        that, see ``robustmargin.censoring.exact_one_sided_bound``.

    Returns
    -------
    ExactRates
        Parallel arrays, one entry per cell, in the given order. **Never attach this to a
        threshold**; it is deliberately not serialisable (see :class:`ExactRates`).
    """
    severity = tuple(c.severity for c in cells)
    rate = tuple(c.successes / c.trials for c in cells)
    intervals = [
        binomtest(c.successes, c.trials).proportion_ci(confidence_level=level, method="exact")
        for c in cells
    ]
    lo = tuple(float(i.low) for i in intervals)
    hi = tuple(float(i.high) for i in intervals)
    return ExactRates(
        severity=severity, rate=rate, lo=lo, hi=hi, level=level, method=_VALID_RATE_METHOD
    )
