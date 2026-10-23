"""Censoring classification and exact one-sided bounds.

Public: :func:`exact_one_sided_bound`, the one piece of numeric machinery every rule in this
module (and :func:`robustmargin.empirical.per_cell_clopper_pearson`) is built from -- the exact
arithmetic lives in exactly one place, here. Internal: :func:`classify`, which turns a
:class:`~robustmargin.Fit` plus an already-resolved target performance into a
:class:`Classification` describing which of the censoring outcomes applies.

**One confidence for every exact claim.** With ``alpha = 1 - level``, every exact claim
(``FAILS_AT_BASELINE``, ``RIGHT``, ``LEFT``, and a failed fit's bracket or one-sided bound) is read
off one-sided exact (Clopper-Pearson) tests at ``c = 1 - alpha / 4`` (0.9875 at ``level = 0.95``).
``pass(x)`` fires when a severity's one-sided lower bound at ``c`` is at least the target; it
rejects "the true rate at ``x`` is at most the target". ``fail(x)`` fires when the upper bound at
``c`` is below the target; it rejects "the true rate at ``x`` is at least the target". Writing
``x_0 = 0`` for the control and ``x_1 < ... < x_K`` for the nonzero severities, the budget
``alpha`` is spent in four parts of ``alpha / 4`` each:

1. the ascending run ``pass(x_1), pass(x_2), ...``, stopping at the first that does not fire,
   which gives a lower bound;
2. the top cell ``pass(x_K)``, which gives ``RIGHT``;
3. the descending run ``fail(x_K), fail(x_{K-1}), ...`` over the nonzero severities, stopping at
   the first that does not fire, which gives an upper bound;
4. the bottom cell ``fail(x_1)``, which gives an upper bound at ``x_1``, the upper half of
   ``LEFT``, and half of ``FAILS_AT_BASELINE``.

The control's own tests, pass(x_0) and fail(x_0), belong to the same set of hypotheses.
Read as a confidence set of crossing locations (below the control, or between two adjacent tested
severities, or above the highest), each location is rejected by at most four of these tests under
its own null, each of size at most alpha / 4. An exact result can be wrong only if the
location that is true for the curve is rejected, so under a monotone curve the chance that *any*
exact result is returned and wrong is at most alpha, whichever kind the data selects and
however many severities were tested. A run is a fixed-sequence procedure: the hypotheses are nested
under a monotone curve and tested in an order fixed in advance, so a run can only err by wrongly
firing on the first severity past the true crossing (Hsu & Berger, 1999, *Journal of the American
Statistical Association* 94:468-482). The assumptions are a monotone curve, independent trials,
and a target that is fixed or depends only on the control's counts.

An increasing outcome is classified on its complement (performance ``1 - P``, target ``1 - t``),
and :func:`classify` is given the target as the caller stated it so that warnings speak of the
outcome as counted: "passes" then means the rate is confidently below the target.

Classification order (the censoring conditions are a list of conditions, not a sequence; this is
the sequence):

1. ``fit.status is CONTROL_INCOMPATIBLE`` -- terminal mirror.
2. The resolved target is at or below 0, or at or above 1 -- ``UNREACHABLE`` on any fit status,
   with a warning naming the target. No binomial performance curve crosses such a target, and an
   exact test against one is trivially true or false (``fail`` against a target of 1 fires for any
   control with a failure; ``pass`` against a target of 0 always fires). This runs before every
   exact test.
3. ``FAILS_AT_BASELINE`` -- a control exists, ``fail(x_0)`` and ``fail(x_1)`` both fire (the
   second is omitted only when there is no nonzero severity), and no nonzero severity's lower
   bound at ``level`` clears the target. That last guard defers non-monotone data to step 5 and can
   only suppress a result, which is why it stays at ``level``. Short-circuits before step 4.
4. ``UNREACHABLE`` -- the target is not strictly inside ``(l, u)``. Only checked when
   ``fit.status is OK``, since reachability needs a fitted ``l``/``u``.
5. Monotonicity contradiction -- over *all* tested levels at ``level``, the highest confidently
   passing nonzero severity is not below the lowest confidently failing severity, so the exact
   tests contradict a monotone curve. A ``NON_MONOTONE: `` warning names both levels. On a failed
   fit, no bounds are reported and the fit's own status is kept. On a converged fit,
   ``RIGHT``/``LEFT`` are skipped and the solve path (step 8) runs, carrying the warning. This is
   a diagnostic of the monotonicity assumption, not an interval claim, so it deliberately looks
   at every level rather than scanning in sequence, and it stays at ``level``: it can only
   suppress a claim, never add one, so it takes no part in the error budget.
6. ``RIGHT`` -- ``pass(x_K)``: the highest nonzero severity's one-sided lower bound at ``c``
   clears the target. ``lo = x_K``.
7. ``LEFT`` -- a control exists, ``pass(x_0)`` and ``fail(x_1)``. ``hi = x_1``. It claims that the
   control passes and the threshold lies below the lowest nonzero severity. (``RIGHT`` and
   ``LEFT`` are evaluated on the data, so they are reachable on a converged fit too.)
8. ``fit.status is OK`` and no claim above fired -- solve (the caller runs the requested interval
   method); ``Classification.solve`` is ``True``. The value may lie outside the tested range, which
   the caller flags.
9. Otherwise (``SEPARATION``/``NOT_CONVERGED``) -- the exact-bound fallback: attach the grid
   value, then ``lo`` is the last severity in the ascending run of ``pass`` over the nonzero
   severities (``None`` if ``pass(x_1)`` does not fire), and ``hi`` is ``x_1`` if ``fail(x_1)``
   fires, otherwise the last severity in the descending run of ``fail`` over the nonzero
   severities (``None`` if ``fail(x_K)`` does not fire). Both set is a bracket with ``lo < hi``;
   exactly one set is a one-sided bound at ``c``, with a warning naming the side and ``c``;
   neither set reports no bounds. Nothing is rescanned at ``level`` and no bound is ever at the
   control. All of this is against the resolved target: on a failed fit under ``baseline_fraction``
   that target comes from the control's observed rate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from scipy.stats import binomtest

from robustmargin.empirical import BaselineRate, GridBreakPoint, grid_break_point
from robustmargin.models import Fit
from robustmargin.types import Cell, Censoring, Status, _check_finite_number

__all__ = ["Classification", "classify", "exact_one_sided_bound"]

_VALID_SIDES = frozenset({"lower", "upper"})

# The two statuses that mean "the fit failed but the data still supports an exact
# statement" -- the exact-bound fallback, and the only two `Threshold` allows `grid` on.
_NONCONVERGENT_STATUSES = frozenset({Status.SEPARATION, Status.NOT_CONVERGED})


def exact_one_sided_bound(
    successes: int, trials: int, *, level: float, side: Literal["lower", "upper"]
) -> float:
    """The exact (Clopper-Pearson) one-sided confidence bound on a binomial proportion at
    confidence ``level``.

    ``scipy.stats.binomtest(successes, trials).proportion_ci`` is two-sided: at
    ``confidence_level = c`` it puts ``(1 - c) / 2`` in each tail, so each endpoint is itself a
    one-sided bound at confidence ``(1 + c) / 2``. To get a one-sided bound at confidence
    ``level``, this passes ``confidence_level = 2 * level - 1`` and reads the requested side.
    This is the *only* place in robustmargin this conversion is written down; every other exact
    bound in the package (per-cell rates, the censoring rules below) calls this function rather
    than repeating it.

    Parameters
    ----------
    successes, trials
        The observed count and the number of trials, as taken by ``scipy.stats.binomtest``.
    level
        The one-sided confidence level, strictly between 0 and 1.
    side
        ``"lower"`` for a one-sided lower bound (how confident a *lower* limit on the true
        proportion is), ``"upper"`` for a one-sided upper bound.

    Returns
    -------
    float
        The requested bound, in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``side`` is not ``"lower"`` or ``"upper"``, or if ``level`` is not a finite number
        strictly between 0 and 1.
    """
    if side not in _VALID_SIDES:
        raise ValueError(
            f"exact_one_sided_bound: side must be one of {sorted(_VALID_SIDES)}, got {side!r}"
        )
    _check_finite_number(level, label="exact_one_sided_bound level")
    # The `2 * level - 1` conversion below is defined only above 0.5: at or under it scipy is
    # handed a non-positive `confidence_level` and raises naming a number the caller never
    # supplied. A one-sided bound at 0.5 or less is not a meaningful request anyway.
    if not (0.5 < level < 1.0):
        raise ValueError(
            f"exact_one_sided_bound: level must satisfy 0.5 < level < 1, got {level!r}"
        )

    confidence_level = 2.0 * level - 1.0
    result = binomtest(successes, trials).proportion_ci(
        confidence_level=confidence_level, method="exact"
    )
    return float(result.low) if side == "lower" else float(result.high)


@dataclass(frozen=True, kw_only=True)
class Classification:
    """The outcome of :func:`classify`: everything :func:`robustmargin.threshold` needs to build a
    :class:`~robustmargin.Threshold`, other than the solved ``value`` itself (which, when
    ``solve`` is ``True``, is the caller's job -- computed from ``fit.params`` directly, then
    refined by the requested interval method).

    Attributes
    ----------
    status
        The :class:`~robustmargin.Status` the resulting threshold should report.
    censoring
        The :class:`~robustmargin.Censoring` the resulting threshold should report. ``NONE`` unless
        ``RIGHT``/``LEFT`` fired.
    lo, hi
        Severity-scale exact bounds, when this classification determined one directly (``RIGHT``,
        ``LEFT``, or a ``SEPARATION``/``NOT_CONVERGED`` bracket or one-sided bound). ``None``
        otherwise.
    baseline
        Set exactly when ``status is Status.FAILS_AT_BASELINE``.
    grid
        Set exactly on the ``SEPARATION``/``NOT_CONVERGED`` exact-bound fallback (step 9), and
        attached to a ``RIGHT``/``LEFT`` result on such a fit.
    warnings
        Free-text notes to append to the resulting threshold's own ``warnings``.
    solve
        ``True`` only for the ``fit.status is OK``, non-``RIGHT``/``LEFT`` case: the caller must
        solve for ``value`` and run the requested interval method.
    """

    status: Status
    censoring: Censoring
    lo: float | None
    hi: float | None
    baseline: BaselineRate | None
    grid: GridBreakPoint | None
    warnings: tuple[str, ...] = ()
    solve: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))


def _control_cell(cells: tuple[Cell, ...]) -> Cell | None:
    for cell in cells:
        if cell.severity == 0.0:
            return cell
    return None


def _baseline_rate(control: Cell, *, level: float) -> BaselineRate:
    """The control cell's rate with its exact interval, for a `FAILS_AT_BASELINE` result.

    This is a **two-sided** Clopper-Pearson interval at `level`, which is how a reported
    per-cell rate is defined: `binomtest(k, n).proportion_ci(confidence_level=...,
    method='exact')`, labelled `per_cell_clopper_pearson`.

    It is deliberately *not* the one-sided bound that decided the `FAILS_AT_BASELINE`
    classification. Those are two different jobs: the classification asks "does this cell
    provably fail the target", a one-sided question answered by `exact_one_sided_bound` at the
    confidence every exact claim uses; the reported interval describes where the control rate
    lies, a two-sided question. Reporting the one-sided pair under the label
    `per_cell_clopper_pearson` would give a roughly `2*level - 1` region wearing a `level`
    name.
    """
    lo, hi = binomtest(control.successes, control.trials).proportion_ci(
        confidence_level=level, method="exact"
    )
    return BaselineRate(
        severity=control.severity,
        successes=control.successes,
        trials=control.trials,
        rate=control.successes / control.trials,
        lo=float(lo),
        hi=float(hi),
        level=level,
        method="per_cell_clopper_pearson",
    )


def _exact_claim_confidence(level: float) -> float:
    """The confidence of every exact claim: ``1 - (1 - level) / 4``.

    The error budget ``1 - level`` is spent in four equal parts (the module docstring), so each
    one-sided test is run at the confidence that leaves ``(1 - level) / 4`` in its tail.
    """
    return 1.0 - (1.0 - level) / 4.0


def _passes(cell: Cell, *, target: float, confidence: float) -> bool:
    """``pass(x)``: the one-sided lower bound at ``confidence`` is at least ``target``."""
    return (
        exact_one_sided_bound(cell.successes, cell.trials, level=confidence, side="lower") >= target
    )


def _fails(cell: Cell, *, target: float, confidence: float) -> bool:
    """``fail(x)``: the one-sided upper bound at ``confidence`` is below ``target``."""
    return (
        exact_one_sided_bound(cell.successes, cell.trials, level=confidence, side="upper") < target
    )


def _nonzero_cells(cells: tuple[Cell, ...]) -> tuple[Cell, ...]:
    """The cells with positive severity, in the ascending order ``cells`` already has."""
    return tuple(c for c in cells if c.severity > 0.0)


def _target_out_of_range(*, target: float, shown_target: float) -> Classification | None:
    """``UNREACHABLE`` for a resolved target at or below 0, or at or above 1, on any fit status.

    A binomial performance curve takes values in ``[0, 1]``, so it never crosses a target outside
    the open interval ``(0, 1)``, and the exact one-sided tests against one are trivially true or
    false: every ``pass`` fires against a target of 0, and ``fail`` fires for any control with a
    failure against a target of 1. A claim made from them would rest on nothing, so this is decided
    before any exact test. ``shown_target`` is the target as the caller states it, which is what
    the warning names.
    """
    if not (target <= 0.0 or target >= 1.0):
        return None
    return Classification(
        status=Status.UNREACHABLE,
        censoring=Censoring.NONE,
        lo=None,
        hi=None,
        baseline=None,
        grid=None,
        warnings=(
            f"censoring.classify: the resolved target {shown_target!r} is outside the open "
            "interval (0, 1), so no binomial performance curve crosses it and an exact test "
            "against it is trivially true or false; reported as UNREACHABLE without running any "
            "exact test",
        ),
    )


def _fails_at_baseline(
    cells: tuple[Cell, ...], *, target: float, level: float, confidence: float
) -> Classification | None:
    """``FAILS_AT_BASELINE``: a control exists, ``fail(x_0)`` and ``fail(x_1)`` fire (the latter
    only when there is a nonzero severity), and no nonzero severity passes at ``level``."""
    control = _control_cell(cells)
    if control is None:
        return None
    if not _fails(control, target=target, confidence=confidence):
        return None
    nonzero = _nonzero_cells(cells)
    if nonzero and not _fails(nonzero[0], target=target, confidence=confidence):
        return None
    # A control that fails is only a *clean* baseline failure if nothing else tested ever
    # confidently passes the same target -- otherwise this is a monotonicity contradiction (a
    # later level passing while the control fails), which the contradiction check reports
    # instead, naming both levels. This guard stays at `level`: it defers data to that
    # diagnostic and can only suppress a result, never add one, so it is not part of the error
    # budget the four exact tests share.
    other_passes = any(
        exact_one_sided_bound(c.successes, c.trials, level=level, side="lower") >= target
        for c in nonzero
    )
    if other_passes:
        return None
    return Classification(
        status=Status.FAILS_AT_BASELINE,
        censoring=Censoring.NONE,
        lo=None,
        hi=None,
        baseline=_baseline_rate(control, level=level),
        grid=None,
    )


def _unreachable(fit: Fit, *, target: float) -> Classification | None:
    if fit.status is not Status.OK:
        return None
    assert fit.params is not None
    lower = fit.params["lower"].value
    upper = fit.params["upper"].value
    if lower < target < upper:
        return None
    return Classification(
        status=Status.UNREACHABLE,
        censoring=Censoring.NONE,
        lo=None,
        hi=None,
        baseline=None,
        grid=None,
    )


def _right(
    fit: Fit, cells: tuple[Cell, ...], *, target: float, confidence: float
) -> Classification | None:
    """``RIGHT``: ``pass(x_K)``, one cell chosen in advance (not a run). With no nonzero severity
    there is no ``x_K``, and a bound at the control would claim nothing."""
    nonzero = _nonzero_cells(cells)
    if not nonzero:
        return None
    highest = nonzero[-1]
    if not _passes(highest, target=target, confidence=confidence):
        return None
    return Classification(
        status=fit.status,
        censoring=Censoring.RIGHT,
        lo=highest.severity,
        hi=None,
        baseline=None,
        grid=None,
    )


def _left(
    fit: Fit, cells: tuple[Cell, ...], *, target: float, confidence: float
) -> Classification | None:
    """``LEFT``: a control exists, ``pass(x_0)`` and ``fail(x_1)``. It claims that the control
    passes and the threshold lies below the lowest nonzero severity."""
    nonzero = _nonzero_cells(cells)
    if not nonzero:
        return None
    lowest_nonzero = nonzero[0]
    if not _fails(lowest_nonzero, target=target, confidence=confidence):
        return None
    control = _control_cell(cells)
    if control is None:
        return None
    if not _passes(control, target=target, confidence=confidence):
        return None
    return Classification(
        status=fit.status,
        censoring=Censoring.LEFT,
        lo=None,
        hi=lowest_nonzero.severity,
        baseline=None,
        grid=None,
    )


def _grid_for(
    cells: tuple[Cell, ...], *, target: float, original_target: float | None = None
) -> GridBreakPoint:
    """The grid break-point of ``cells`` against ``target``. For the complement of an increasing
    outcome (``original_target`` given) it is read on the counts as supplied: a level breaks when
    its event rate is strictly above ``original_target``, compared as ``events / trials``, so that
    a rate equal to the target does not break whatever rounding ``1 - target`` carries."""
    if original_target is not None:
        # `events / trials > a` is `-events / trials < -a`, and float negation is exact, so the
        # one grid rule decides it with no rounding from `1 - a`.
        return grid_break_point(
            [c.severity for c in cells],
            [-((c.trials - c.successes) / c.trials) for c in cells],
            criterion=-original_target,
            signed=False,
        )
    severity = [c.severity for c in cells]
    performance = [c.successes / c.trials for c in cells]
    return grid_break_point(severity, performance, criterion=target, signed=False)


def _contradiction_warning(
    *,
    lo_severity: float,
    passes_clause: str,
    hi_severity: float,
    fails_clause: str,
    level: float,
    curve_exists: bool,
) -> str:
    """``curve_exists`` picks the one sentence that differs between the two branches of this
    function's one caller, ``_contradiction_or_none``, so the rest of the wording -- and the
    frozen ``NON_MONOTONE: `` token -- is shared.

    ``True`` (``_contradiction_or_none``'s own ``Status.OK`` branch): the fit converged, so
    ``threshold()`` still solves for ``value`` and still runs the requested interval method --
    only the *exact* one-sided bounds (the suppressed ``RIGHT``/``LEFT`` claim) go unreported,
    not the profile/delta ``lo``/``hi`` this ``Threshold`` actually carries.

    ``False`` (``_contradiction_or_none``'s non-``OK`` branch, a failed fit): there is no fitted
    curve at all, and this
    ``Threshold``'s ``value``/``lo``/``hi`` are all ``None`` -- "no bounds are reported" is
    simply true here, with nothing to qualify.
    """
    exact_bounds_clause = (
        "the exact one-sided bounds are not reported here -- any lo/hi this Threshold carries "
        "comes from the fitted curve instead, not from these exact tests"
        if curve_exists
        else "no bounds are reported"
    )
    return (
        # The leading `NON_MONOTONE: ` token is the supported way to detect this condition --
        # distinct from `Fit.monotonicity`'s own structured, data-only adjacent-level check,
        # which compares raw counts against each other rather than exact bounds against the
        # *fitted* curve; the two can disagree, and both are reported when they fire (see
        # `threshold.py`'s `"NON_MONOTONE_DATA: "` token). On a converged fit this warning is the
        # *only* signal of *this* kind of contradiction -- `grid` is legal only for
        # SEPARATION/NOT_CONVERGED, so a consumer filtering on `status is OK and censoring is
        # NONE` would otherwise see a clean bounded interval from data the exact tests just
        # proved inconsistent with any monotone curve. Changing or removing this token is a
        # behaviourally breaking change.
        "NON_MONOTONE: censoring.classify: exact one-sided tests at confidence "
        f"{level} contradict monotonicity: severity {lo_severity} confidently "
        f"passes ({passes_clause}) while the lower severity "
        f"{hi_severity} confidently fails ({fails_clause}); "
        f"{exact_bounds_clause}"
    )


def _scan_bounds(
    cells: tuple[Cell, ...], *, target: float, level: float
) -> tuple[float | None, float, float | None, float]:
    """The highest severity that confidently passes and the lowest that confidently fails.

    The passing scan skips severity 0: "the threshold is above 0" is true of every threshold by
    construction, so a control that passes is not a bound. The failing scan keeps it, because a
    control that *fails* is genuinely informative and is what a monotonicity contradiction is
    usually built from.

    This looks at every cell and is deliberately not a fixed-sequence run (see
    :func:`_ascending_pass_run_end`): it is a diagnostic of the monotonicity assumption, not an
    interval claim, and a scan that stops at the first gap would hide the contradictions it exists
    to find. It is used by the contradiction check alone.
    """
    lo_severity: float | None = None
    lo_bound = 0.0
    hi_severity: float | None = None
    hi_bound = 0.0
    for cell in cells:
        if cell.severity > 0.0:
            bound = exact_one_sided_bound(cell.successes, cell.trials, level=level, side="lower")
            if bound >= target and (lo_severity is None or cell.severity > lo_severity):
                lo_severity, lo_bound = cell.severity, bound
        bound_up = exact_one_sided_bound(cell.successes, cell.trials, level=level, side="upper")
        if bound_up < target and (hi_severity is None or cell.severity < hi_severity):
            hi_severity, hi_bound = cell.severity, bound_up
    return lo_severity, lo_bound, hi_severity, hi_bound


def _ascending_pass_run_end(
    nonzero: tuple[Cell, ...], *, target: float, confidence: float
) -> float | None:
    """The last severity in the ascending run of ``pass`` over the nonzero severities, or ``None``
    if the run is empty.

    Walk the nonzero severities from the lowest upwards while each one's one-sided lower bound at
    ``confidence`` is at least ``target``; the result is the last severity in that run. The scan
    stops at the first severity that does not qualify, so a later severity that would qualify is
    never reached. Severity 0 is not in ``nonzero``: a control that passes says "the threshold is
    above 0", which every threshold satisfies by construction, so it is not a bound (and on a log
    axis ``lo = 0`` is illegal).

    Why a run and not the extreme: under a monotone decreasing true curve the hypotheses
    ``H_i: p(x_i) <= target`` are nested (if ``H_i`` is true so is every ``H_j`` with
    ``x_j > x_i``). The scan rejects ``H_i`` only after rejecting every ``H_j`` with a lower
    severity, so a wrong result needs the first true hypothesis to be rejected, which has
    probability at most ``1 - confidence`` however many levels lie near the threshold. This is
    fixed-sequence testing of a-priori ordered hypotheses (Hsu & Berger, 1999, *Journal of the
    American Statistical Association* 94:468-482).
    """
    end: float | None = None
    for cell in nonzero:
        if not _passes(cell, target=target, confidence=confidence):
            break
        end = cell.severity
    return end


def _descending_fail_run_end(
    nonzero: tuple[Cell, ...], *, target: float, confidence: float
) -> float | None:
    """The last severity in the descending run of ``fail`` over the nonzero severities, or
    ``None`` if the run is empty.

    The mirror image of :func:`_ascending_pass_run_end`: walk the nonzero severities from the
    highest downwards while each one's one-sided upper bound at ``confidence`` is below
    ``target``. The control is not in ``nonzero`` and is never a bound, so an upper bound is always
    the severity of a nonzero cell.
    """
    end: float | None = None
    for cell in reversed(nonzero):
        if not _fails(cell, target=target, confidence=confidence):
            break
        end = cell.severity
    return end


def _contradiction_clauses(
    cells: tuple[Cell, ...],
    *,
    lo_severity: float,
    lo_bound: float,
    hi_severity: float,
    hi_bound: float,
    target: float,
    shown_target: float,
    level: float,
    mirrored: bool,
) -> tuple[str, str]:
    """The two bound statements of a contradiction warning, in the terms of the outcome as the
    caller counted it. On the view itself they are the bounds just computed. For the complement of
    an increasing outcome, "passes" is a confident *upper* bound at or below the stated target and
    "fails" a confident *lower* bound above it, taken from the original counts."""
    if not mirrored:
        return (
            f"lower bound {lo_bound!r} >= target {target!r}",
            f"upper bound {hi_bound!r} < target {target!r}",
        )
    by_severity = {c.severity: c for c in cells}
    passing, failing = by_severity[lo_severity], by_severity[hi_severity]
    upper_bound = exact_one_sided_bound(
        passing.trials - passing.successes, passing.trials, level=level, side="upper"
    )
    lower_bound = exact_one_sided_bound(
        failing.trials - failing.successes, failing.trials, level=level, side="lower"
    )
    return (
        f"upper bound {upper_bound!r} <= target {shown_target!r}",
        f"lower bound {lower_bound!r} > target {shown_target!r}",
    )


def _contradiction_or_none(
    fit: Fit,
    cells: tuple[Cell, ...],
    *,
    target: float,
    shown_target: float,
    level: float,
    mirrored: bool,
) -> Classification | None:
    """The exact-bounds-contradict-monotonicity check, applied *before* the `RIGHT`/`LEFT`
    branches rather than only inside the bracket.

    Both short-circuits return a directional claim about where the threshold lies -- "sigma* >
    x_max", "sigma* < x_min" -- and every such claim rests on the monotonicity assumption stated
    in `Threshold`'s docstring. When the exact tests themselves contradict that assumption, the
    premise of the claim has failed, so the claim is not a weaker true statement: it is a
    statement whose premise is false. Reporting it with no diagnostic is what this check exists
    to prevent.

    The confidence is `level`, not the `1 - (1 - level) / 4` the exact claims are read at,
    because this is a diagnostic that can only suppress a claim, never add one. A check at the
    looser `level` flags a contradiction in at least every case a stricter check would, and
    suppressing a claim never costs coverage, so the check takes no share of the error budget the
    four exact tests spend.
    """
    lo_severity, lo_bound, hi_severity, hi_bound = _scan_bounds(cells, target=target, level=level)
    if lo_severity is None or hi_severity is None or lo_severity < hi_severity:
        return None
    passes_clause, fails_clause = _contradiction_clauses(
        cells,
        lo_severity=lo_severity,
        lo_bound=lo_bound,
        hi_severity=hi_severity,
        hi_bound=hi_bound,
        target=target,
        shown_target=shown_target,
        level=level,
        mirrored=mirrored,
    )
    warning = _contradiction_warning(
        lo_severity=lo_severity,
        passes_clause=passes_clause,
        hi_severity=hi_severity,
        fails_clause=fails_clause,
        level=level,
        curve_exists=fit.status is Status.OK,
    )
    if fit.status is Status.OK:
        # A converged fit still has a model-based estimate, and `Threshold` requires `value`
        # whenever status is OK and censoring is NONE -- the all-`None` encoding of a
        # contradiction (no value, no bounds) is only constructible on a failed fit. So the
        # contradiction suppresses the `RIGHT`/`LEFT` branches (whose directional claim rests on
        # the monotonicity the data just contradicted) and falls through to the solve path,
        # carrying the warning. The caller gets the model's answer plus an explicit statement
        # that the exact tests disagree with the model's monotonicity assumption: a diagnostic
        # that is reported and never acted on.
        return Classification(
            status=Status.OK,
            censoring=Censoring.NONE,
            lo=None,
            hi=None,
            baseline=None,
            grid=None,
            warnings=(warning,),
            solve=True,
        )
    return Classification(
        status=fit.status,
        censoring=Censoring.NONE,
        lo=None,
        hi=None,
        baseline=None,
        grid=_grid_for(cells, target=target, original_target=shown_target if mirrored else None),
        warnings=(warning,),
    )


def _exact_bound_fallback(
    fit: Fit,
    cells: tuple[Cell, ...],
    *,
    target: float,
    shown_target: float,
    level: float,
    mirrored: bool,
) -> Classification:
    """The exact bracket (or one-sided bound) for a failed fit, from fixed-sequence runs.

    Both sides are read at ``c = 1 - (1 - level) / 4``, over the nonzero severities only. ``lo`` is
    the end of the ascending run of ``pass`` (:func:`_ascending_pass_run_end`). ``hi`` is the
    lowest nonzero severity when it fails (the bottom-cell test), and otherwise the end of the
    descending run of ``fail`` (:func:`_descending_fail_run_end`). Under a monotone curve the
    per-level hypotheses are nested, so a run can only err by wrongly passing (or failing) the
    first level past the threshold, which keeps each part's error at most ``1 - c`` whatever the
    number of levels, where taking the extreme qualifying severity anywhere would let every level
    near the threshold add its own chance of error. The four parts the exact claims share
    (ascending run, top cell, descending run, bottom cell) add up to ``1 - level``.

    Both sides set is a bracket, and ``lo < hi`` always: a cell cannot both pass and fail, the
    ascending run is a prefix of the nonzero severities and the descending run a suffix, and a
    bottom cell that fails is not in the passing run. Exactly one side set is a single one-sided
    statement, reported at ``c`` with a warning that names the side and ``c``. Neither set reports
    no bounds. Nothing is rescanned at ``level``: a rescan would test the same data a second time
    at a looser confidence, and the error of such a statement would exceed the share of the budget
    it is allotted. No bound is ever at the control, whose pass is true of every threshold and
    whose failure is the separate ``FAILS_AT_BASELINE`` result.

    No contradiction check here. A cell cannot both pass (lower bound >= target) and fail (upper
    bound < target), so the runs never overlap, and :func:`classify` has already run the all-cells
    contradiction check at ``level`` before reaching this fallback: a diagnostic that can only
    suppress a claim, never add one.
    """
    grid = _grid_for(cells, target=target, original_target=shown_target if mirrored else None)
    confidence = _exact_claim_confidence(level)
    nonzero = _nonzero_cells(cells)

    lo_severity = _ascending_pass_run_end(nonzero, target=target, confidence=confidence)
    if nonzero and _fails(nonzero[0], target=target, confidence=confidence):
        hi_severity: float | None = nonzero[0].severity
    else:
        hi_severity = _descending_fail_run_end(nonzero, target=target, confidence=confidence)

    warnings: tuple[str, ...] = ()
    if lo_severity is None and hi_severity is None:
        warnings = (
            "censoring.classify: there is no run of confidently passing levels from the lowest "
            "nonzero severity, and neither the lowest nor the highest nonzero severity "
            f"confidently fails, at confidence {confidence}, for the target {shown_target!r}; "
            "no bounds are reported",
        )
    elif lo_severity is None or hi_severity is None:
        # Only one side is determined, so this is not a bracket. It is reported alone, at the
        # confidence its test was run at: there is no rescan at `level`, because a rescan would
        # spend more than this side's share of the error budget.
        determined = "lower" if hi_severity is None else "upper"
        missing_test = (
            "neither the lowest nor the highest nonzero severity confidently fails"
            if determined == "lower"
            else "the lowest nonzero severity does not confidently pass"
        )
        warnings = (
            f"censoring.classify: the exact-bound fallback determined only the {determined} "
            f"side: {missing_test} at confidence {confidence}, for the target "
            f"{shown_target!r}. The {determined} side is reported alone, as a one-sided exact "
            f"bound at confidence {confidence}, not a bracket. Censoring stays NONE because "
            "RIGHT and LEFT are reserved for their own end-of-range conditions, which this result "
            "did not meet",
        )
    return Classification(
        status=fit.status,
        censoring=Censoring.NONE,
        lo=lo_severity,
        hi=hi_severity,
        baseline=None,
        grid=grid,
        warnings=warnings,
    )


def _with_grid(
    classification: Classification,
    fit: Fit,
    cells: tuple[Cell, ...],
    *,
    target: float,
    original_target: float | None,
) -> Classification:
    """Attach the grid break-point when, and only when, the fit failed.

    ``Threshold`` permits ``grid`` only for ``SEPARATION`` and ``NOT_CONVERGED``, so a
    converged fit's one-sided bound is returned unchanged.
    """
    if fit.status not in _NONCONVERGENT_STATUSES:
        return classification
    return replace(
        classification, grid=_grid_for(cells, target=target, original_target=original_target)
    )


def classify(
    fit: Fit, *, target: float, level: float, original_target: float | None = None
) -> Classification:
    """Classify a fit against a resolved target performance (order per the module docstring).

    Parameters
    ----------
    fit
        The :class:`~robustmargin.Fit` to classify. Its ``cells`` (sorted ascending by
        :meth:`~robustmargin.Observations.cells`) are what every exact test here is run against.
    target
        The already-resolved target performance ``P*``. Resolving it from a
        :class:`~robustmargin.Definition` -- including which ``u`` a failed-fit fallback uses --
        is the caller's job.
    level
        The confidence level, strictly between 0 and 1.
    original_target
        ``None`` for a fit of the outcome as counted. Otherwise ``fit`` is the complementary
        outcome of an increasing one, ``target`` is already complemented, and this is the target
        as the caller stated it: warnings name it, and bounds they quote are taken from the
        original counts. Every test and every number in the result is unchanged by it.

    Returns
    -------
    Classification
    """
    mirrored = original_target is not None
    shown_target = original_target if original_target is not None else target
    if fit.status is Status.CONTROL_INCOMPATIBLE:
        return Classification(
            status=Status.CONTROL_INCOMPATIBLE,
            censoring=Censoring.NONE,
            lo=None,
            hi=None,
            baseline=None,
            grid=None,
        )

    # A target outside (0, 1) is decided before any exact test: every one of them is trivially
    # true or false against it. It follows `CONTROL_INCOMPATIBLE`, which is terminal.
    out_of_range = _target_out_of_range(target=target, shown_target=shown_target)
    if out_of_range is not None:
        return out_of_range

    cells = fit.cells
    confidence = _exact_claim_confidence(level)

    fails_at_baseline = _fails_at_baseline(cells, target=target, level=level, confidence=confidence)
    if fails_at_baseline is not None:
        return fails_at_baseline

    unreachable = _unreachable(fit, target=target)
    if unreachable is not None:
        return unreachable

    # `RIGHT`/`LEFT` are reached on a converged fit as well as a failed one (they are evaluated
    # on the data), so the grid statistic is attached here rather than inside `_right`/`_left`:
    # the grid rule accompanies the exact bounds for separation and non-convergence, and a
    # one-sided bound on a failed fit is that same fallback. On an `OK` fit `Threshold` forbids
    # `grid`, so the rule is stated once, in one place, and cannot leak into the `OK` branch.
    # The monotonicity-contradiction check runs before the directional branches: a
    # `RIGHT`/`LEFT` bound asserts where the threshold lies, and that assertion rests on
    # monotonicity. If the exact tests contradict monotonicity, no such bound may be reported.
    contradiction = _contradiction_or_none(
        fit, cells, target=target, shown_target=shown_target, level=level, mirrored=mirrored
    )
    if contradiction is not None:
        return contradiction

    right = _right(fit, cells, target=target, confidence=confidence)
    if right is not None:
        return _with_grid(right, fit, cells, target=target, original_target=original_target)

    left = _left(fit, cells, target=target, confidence=confidence)
    if left is not None:
        return _with_grid(left, fit, cells, target=target, original_target=original_target)

    if fit.status is Status.OK:
        return Classification(
            status=Status.OK,
            censoring=Censoring.NONE,
            lo=None,
            hi=None,
            baseline=None,
            grid=None,
            solve=True,
        )

    return _exact_bound_fallback(
        fit, cells, target=target, shown_target=shown_target, level=level, mirrored=mirrored
    )
