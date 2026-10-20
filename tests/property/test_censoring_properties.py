"""Property tests for ``censoring.classify``: the exact claims, all read off tests at ``c4``.

Every exact claim is a one-sided exact test at ``c4 = 1 - (1 - level) / 4`` (the error budget
``1 - level`` split four ways: an ascending run, the top cell, a descending run, the bottom cell).
Four properties, on arbitrary counts rather than hand-picked grids:

* **The run property.** On a failed fit whose result is an exact-bound fallback (censoring
  ``NONE``, not ``FAILS_AT_BASELINE``, no ``NON_MONOTONE: `` contradiction), a reported ``lo`` is
  the end of an unbroken run of confidently passing nonzero severities from the lowest, and the
  next nonzero severity does not pass. ``hi`` is ``x_1`` whenever the lowest nonzero severity
  confidently fails; otherwise it is the end of an unbroken run of confidently failing nonzero
  severities from the highest, and the next lower one does not fail. A side with no run is
  ``None``. All at ``c4``, whether one side is reported or both; both sides set means ``lo < hi``;
  no bound is at the control.
* **Clean data.** When the cells that pass at ``c4`` are a prefix of the nonzero severities and
  the ones that fail at ``c4`` a suffix, the runs equal a max-anywhere rule at ``c4``. A level
  that passes (or fails) only at the looser ``level`` is neither, so it ends a run and is not
  rescanned.
* **The specification table.** For arbitrary cells, targets (inside and outside ``(0, 1)``),
  levels and fit statuses, ``classify`` returns exactly the kind the written rules give: an
  independent transcription of the classification order and of every firing condition.
* **A higher level never narrows a claim.** Raising ``level`` raises ``c4``, which only makes
  each test harder to pass, so a failed fit's fallback interval can only widen.

Examples are small and the search is hypothesis's own (seeded, deterministic) one; the explicit
``@example`` cases keep the properties from being vacuous by guaranteeing the interesting
non-contiguous shapes are always tried.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Iterator

import pytest
from hypothesis import event, example, given, settings
from hypothesis import strategies as st

from robustmargin import Axis, Fit, Status, censoring
from robustmargin.censoring import Classification, classify
from robustmargin.testing import fake_fit
from robustmargin.types import Cell, Censoring

_MAX_EXAMPLES = 200
_FAILED_STATUSES = (Status.SEPARATION, Status.NOT_CONVERGED)

_TRIALS = [20, 50, 100, 400]
# Fractions chosen to land on both sides of, and right at, the targets below, so cells are
# regularly confidently-passing, confidently-failing and undetermined rather than almost always
# one of the first two.
_FRACTIONS = [0.0, 0.1, 0.5, 0.8, 0.84, 0.88, 0.9, 0.92, 0.96, 1.0]
_TARGETS = [0.5, 0.8, 0.9, 0.95]
_LEVELS = [0.9, 0.95, 0.99]


def _c4(level: float) -> float:
    return 1.0 - (1.0 - level) / 4.0


@pytest.fixture(scope="module", autouse=True)
def _memoised_exact_bound() -> Iterator[None]:
    """``exact_one_sided_bound`` is a pure function of ``(successes, trials, level, side)`` and
    the strategies draw from a few dozen distinct count pairs, so memoising it changes no result
    -- only the run time (it is a ``scipy.stats.binomtest`` call, ~10 ms, and ``classify`` makes
    dozens per example). Module-scoped so hypothesis's function-scoped-fixture health check does
    not apply; the helpers below go through ``censoring.exact_one_sided_bound`` so they use the
    same cache."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            censoring,
            "exact_one_sided_bound",
            functools.lru_cache(maxsize=None)(censoring.exact_one_sided_bound),
        )
        yield


def _failed_fit(cells: tuple[Cell, ...], status: Status) -> Fit:
    return dataclasses.replace(fake_fit(status=status, monotonicity=None), cells=cells)


# A converged fit with a failing control and `upper` fixed at 1 is only valid on a linear axis.
_LINEAR_AXIS = Axis(name="severity", unit="unit", scale="linear")


def _fit(cells: tuple[Cell, ...], status: Status) -> Fit:
    if status is Status.OK:
        return dataclasses.replace(
            fake_fit(diagnostics=None, monotonicity=None), axis=_LINEAR_AXIS, cells=cells
        )
    return _failed_fit(cells, status)


def _lower(cell: Cell, confidence: float) -> float:
    return censoring.exact_one_sided_bound(
        cell.successes, cell.trials, level=confidence, side="lower"
    )


def _upper(cell: Cell, confidence: float) -> float:
    return censoring.exact_one_sided_bound(
        cell.successes, cell.trials, level=confidence, side="upper"
    )


@st.composite
def _arbitrary_cells(draw: st.DrawFn) -> tuple[Cell, ...]:
    """1 to 7 cells at severities 0, 1, 2, ... (the control, severity 0, only sometimes)."""
    with_control = draw(st.booleans())
    size = draw(st.integers(min_value=1, max_value=7))
    first = 0 if with_control else 1
    cells = []
    for severity in range(first, first + size):
        trials = draw(st.sampled_from(_TRIALS))
        successes = round(draw(st.sampled_from(_FRACTIONS)) * trials)
        cells.append(Cell(severity=float(severity), successes=successes, trials=trials))
    return tuple(cells)


@st.composite
def _monotone_cells(draw: st.DrawFn) -> tuple[Cell, ...]:
    """As ``_arbitrary_cells`` but with the observed fractions non-increasing in severity, so the
    monotone kinds (a bracket, ``RIGHT``, ``LEFT``, ``FAILS_AT_BASELINE``) are common rather than
    rare."""
    with_control = draw(st.booleans())
    size = draw(st.integers(min_value=1, max_value=7))
    fractions = sorted(
        (draw(st.sampled_from(_FRACTIONS)) for _ in range(size)),
        reverse=True,
    )
    first = 0 if with_control else 1
    cells = []
    for offset, fraction in enumerate(fractions):
        trials = draw(st.sampled_from(_TRIALS))
        cells.append(
            Cell(
                severity=float(first + offset),
                successes=round(fraction * trials),
                trials=trials,
            )
        )
    return tuple(cells)


_CELLS = st.one_of(_arbitrary_cells(), _monotone_cells())


def _is_fallback(result: Classification, status: Status) -> bool:
    return (
        result.status is status
        and result.censoring is Censoring.NONE
        and result.grid is not None
        and not any(w.startswith("NON_MONOTONE: ") for w in result.warnings)
    )


def _assert_run_property(
    cells: tuple[Cell, ...], result: Classification, *, target: float, level: float
) -> None:
    c4 = _c4(level)
    nonzero = [c for c in cells if c.severity > 0.0]
    nonzero_severities = {c.severity for c in nonzero}
    lo, hi = result.lo, result.hi

    # lo: the end of the unbroken ascending run of passes at c4, over nonzero severities.
    if lo is not None:
        assert lo in nonzero_severities, "a bound is never at the control"
        for c in nonzero:
            if c.severity <= lo:
                assert _lower(c, c4) >= target, (
                    f"lo={lo} but nonzero severity {c.severity} at or below it does not "
                    f"confidently pass at {c4}: not an unbroken run"
                )
        above = [c for c in nonzero if c.severity > lo]
        if above:
            assert _lower(above[0], c4) < target, (
                f"lo={lo} stops short: the next nonzero severity {above[0].severity} "
                f"confidently passes at {c4}"
            )
    elif nonzero:
        assert _lower(nonzero[0], c4) < target, (
            "no lo reported, but the lowest nonzero severity confidently passes at c4, so the "
            "run could begin"
        )

    # hi: x_1 when the lowest nonzero severity fails at c4 (the bottom-cell test); otherwise the
    # end of the unbroken descending run of fails at c4, over nonzero severities.
    if nonzero and _upper(nonzero[0], c4) < target:
        assert hi == nonzero[0].severity, (
            f"the lowest nonzero severity {nonzero[0].severity} confidently fails at {c4}, so "
            f"hi must be there whatever the run from the top does; got hi={hi}"
        )
    elif hi is not None:
        assert hi in nonzero_severities, "a bound is never at the control"
        for c in nonzero:
            if c.severity >= hi:
                assert _upper(c, c4) < target, (
                    f"hi={hi} but nonzero severity {c.severity} at or above it does not "
                    f"confidently fail at {c4}: not an unbroken run"
                )
        below = [c for c in nonzero if c.severity < hi]
        assert below, f"hi={hi} is the lowest nonzero severity but that level does not fail at {c4}"
        assert _upper(below[-1], c4) >= target, (
            f"hi={hi} stops short: the next lower nonzero severity {below[-1].severity} "
            f"confidently fails at {c4}"
        )
    elif nonzero:
        assert _upper(nonzero[-1], c4) >= target, (
            "no hi reported, but the highest severity confidently fails at c4, so the run "
            "could begin"
        )

    if lo is not None and hi is not None:
        assert lo < hi, f"a bracket must have lo < hi, got ({lo}, {hi})"

    # One warning for a one-sided bound or no bounds, none for a bracket.
    if lo is not None and hi is not None:
        assert result.warnings == ()
    elif lo is None and hi is None:
        assert len(result.warnings) == 1 and "no bounds are reported" in result.warnings[0]
    else:
        side = "lower" if hi is None else "upper"
        assert len(result.warnings) == 1
        assert f"determined only the {side} side" in result.warnings[0]
        assert "rescanned" not in result.warnings[0]


def _cells_from(counts: list[tuple[int, int]], *, first_severity: int) -> tuple[Cell, ...]:
    return tuple(
        Cell(severity=float(first_severity + i), successes=s, trials=n)
        for i, (s, n) in enumerate(counts)
    )


@given(
    cells=_CELLS,
    status=st.sampled_from(_FAILED_STATUSES),
    target=st.sampled_from(_TARGETS),
    level=st.sampled_from(_LEVELS),
)
# The unit tests' shapes, so the non-contiguous cases are always tried (P = 100/100, G = 90/100,
# F = 0/100 at target 0.9, level 0.95).
@example(
    cells=_cells_from(
        [(100, 100), (100, 100), (100, 100), (90, 100), (100, 100), (0, 100), (0, 100)],
        first_severity=0,
    ),
    status=Status.SEPARATION,
    target=0.9,
    level=0.95,
)
@example(
    cells=_cells_from(
        [(100, 100), (100, 100), (100, 100), (100, 100), (0, 100), (90, 100), (0, 100), (0, 100)],
        first_severity=0,
    ),
    status=Status.NOT_CONVERGED,
    target=0.9,
    level=0.95,
)
@example(
    cells=_cells_from(
        [(100, 100), (100, 100), (90, 100), (100, 100), (90, 100)],
        first_severity=0,
    ),
    status=Status.SEPARATION,
    target=0.9,
    level=0.95,
)
# A broken descending run with the lowest nonzero level failing: hi must be that level
# (G F G F), and the same with a control that cannot take part (no control at all).
@example(
    cells=_cells_from([(90, 100), (0, 100), (90, 100), (0, 100)], first_severity=0),
    status=Status.SEPARATION,
    target=0.9,
    level=0.95,
)
@example(
    cells=_cells_from([(0, 100), (90, 100), (0, 100)], first_severity=1),
    status=Status.NOT_CONVERGED,
    target=0.9,
    level=0.95,
)
# Levels that pass or fail only at `level` (49/50 and 84/100): the run ends there.
@example(
    cells=_cells_from(
        [(100, 100), (100, 100), (49, 50), (90, 100), (100, 100), (90, 100)], first_severity=0
    ),
    status=Status.SEPARATION,
    target=0.9,
    level=0.95,
)
@example(
    cells=_cells_from(
        [(100, 100), (90, 100), (0, 100), (90, 100), (84, 100), (0, 100)], first_severity=0
    ),
    status=Status.NOT_CONVERGED,
    target=0.9,
    level=0.95,
)
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_a_reported_bound_is_the_end_of_an_unbroken_run_from_its_own_end(
    cells: tuple[Cell, ...], status: Status, target: float, level: float
) -> None:
    result = classify(_failed_fit(cells, status), target=target, level=level)
    if not _is_fallback(result, status):
        event("not an exact-bound fallback (RIGHT / LEFT / baseline / contradiction)")
        return

    event("exact-bound fallback")
    _assert_run_property(cells, result, target=target, level=level)


# --------------------------------------------------------------------------------------------
# Clean data: the runs equal a max-anywhere rule at c4
# --------------------------------------------------------------------------------------------

# At target 0.9, level 0.95 (c4 = 0.9875); each role is verified against
# `exact_one_sided_bound` in the unit tests (`tests/unit/test_censoring_fixed_sequence.py`).
_ROLE_COUNTS: dict[str, tuple[int, int]] = {
    "P": (100, 100),  # passes at c4 and at level
    "p": (49, 50),  # passes at level only
    "G": (90, 100),  # undetermined at both
    "f": (84, 100),  # fails at level only
    "F": (0, 100),  # fails at c4 and at level
}
_CLEAN_TARGET = 0.9
_CLEAN_LEVEL = 0.95


def _max_anywhere_rule(
    cells: tuple[Cell, ...], *, target: float, level: float
) -> tuple[float | None, float | None]:
    """The max-anywhere rule at ``c4``, as an oracle for clean data: ``lo`` the highest nonzero
    severity whose lower bound clears the target, ``hi`` the lowest nonzero severity whose upper
    bound is below it, both at ``c4``. No rescan at ``level`` and no distinction between a
    bracket and a one-sided bound."""
    c4 = _c4(level)
    nonzero = [c for c in cells if c.severity > 0.0]
    lo_found = [c.severity for c in nonzero if _lower(c, c4) >= target]
    hi_found = [c.severity for c in nonzero if _upper(c, c4) < target]
    return (max(lo_found) if lo_found else None, min(hi_found) if hi_found else None)


@st.composite
def _clean_roles(draw: st.DrawFn) -> str:
    """``<control> P* p* G* f* F*``: passes at ``c4`` a prefix of the nonzero severities and fails
    at ``c4`` a suffix. The control is ``P`` or ``G`` (a failing control is either
    ``FAILS_AT_BASELINE`` or a contradiction, never an exact bracket)."""
    control = draw(st.sampled_from("PG"))
    counts = {role: draw(st.integers(min_value=0, max_value=3)) for role in "pGf"}
    counts["P"] = draw(st.integers(min_value=0, max_value=4))
    counts["F"] = draw(st.integers(min_value=0, max_value=4))
    body = "".join(role * counts[role] for role in "PpGfF")
    return control + body


@given(roles=_clean_roles(), status=st.sampled_from(_FAILED_STATUSES))
@example(roles="PPPGFF", status=Status.SEPARATION)
@example(roles="PPPpGfFF", status=Status.NOT_CONVERGED)
@example(roles="PPPpG", status=Status.SEPARATION)
@example(roles="PGGfF", status=Status.NOT_CONVERGED)
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_clean_data_gives_the_same_bounds_as_the_max_anywhere_rule_at_c4(
    roles: str, status: Status
) -> None:
    cells = tuple(
        Cell(severity=float(i), successes=_ROLE_COUNTS[r][0], trials=_ROLE_COUNTS[r][1])
        for i, r in enumerate(roles)
    )
    result = classify(_failed_fit(cells, status), target=_CLEAN_TARGET, level=_CLEAN_LEVEL)
    if not _is_fallback(result, status):
        event("not an exact-bound fallback (RIGHT / LEFT / baseline / no cells)")
        return

    event("exact-bound fallback")
    expected = _max_anywhere_rule(cells, target=_CLEAN_TARGET, level=_CLEAN_LEVEL)
    assert (result.lo, result.hi) == expected


# --------------------------------------------------------------------------------------------
# The specification table: an independent transcription of the written rules
# --------------------------------------------------------------------------------------------


def _expected_kind(
    cells: tuple[Cell, ...], status: Status, *, target: float, level: float
) -> tuple[object, ...]:
    """What the written classification rules give, as a comparable tuple.

    Written from the rules, not from ``classify``: every test is a one-sided exact bound at ``c4``
    except the two the rules keep at ``level`` (the ``FAILS_AT_BASELINE`` guard and the
    contradiction check).
    """
    c4 = _c4(level)
    control = next((c for c in cells if c.severity == 0.0), None)
    nonzero = [c for c in cells if c.severity > 0.0]

    def passes(c: Cell, confidence: float) -> bool:
        return _lower(c, confidence) >= target

    def fails(c: Cell, confidence: float) -> bool:
        return _upper(c, confidence) < target

    # 2. A target outside (0, 1) is UNREACHABLE on any fit status.
    if not 0.0 < target < 1.0:
        return ("UNREACHABLE",)

    # 3. FAILS_AT_BASELINE.
    if (
        control is not None
        and fails(control, c4)
        and (not nonzero or fails(nonzero[0], c4))
        and not any(passes(c, level) for c in nonzero)
    ):
        return ("FAILS_AT_BASELINE",)

    # 5. The contradiction check, at `level`, over every cell.
    passing = [c.severity for c in nonzero if passes(c, level)]
    failing = [c.severity for c in cells if fails(c, level)]
    if passing and failing and max(passing) >= min(failing):
        return ("CONTRADICTION", status is Status.OK)

    # 6. RIGHT.
    if nonzero and passes(nonzero[-1], c4):
        return ("RIGHT", nonzero[-1].severity, status)

    # 7. LEFT.
    if control is not None and nonzero and passes(control, c4) and fails(nonzero[0], c4):
        return ("LEFT", nonzero[0].severity, status)

    # 8. Solve, on a converged fit.
    if status is Status.OK:
        return ("SOLVE",)

    # 9. The fallback.
    lo = None
    for c in nonzero:
        if not passes(c, c4):
            break
        lo = c.severity
    hi = None
    if nonzero and fails(nonzero[0], c4):
        hi = nonzero[0].severity
    else:
        for c in reversed(nonzero):
            if not fails(c, c4):
                break
            hi = c.severity
    return ("FALLBACK", lo, hi, status)


def _observed_kind(result: Classification, status: Status) -> tuple[object, ...]:
    if result.status is Status.UNREACHABLE:
        return ("UNREACHABLE",)
    if result.status is Status.FAILS_AT_BASELINE:
        return ("FAILS_AT_BASELINE",)
    if any(w.startswith("NON_MONOTONE: ") for w in result.warnings):
        return ("CONTRADICTION", result.solve)
    if result.censoring is Censoring.RIGHT:
        return ("RIGHT", result.lo, result.status)
    if result.censoring is Censoring.LEFT:
        return ("LEFT", result.hi, result.status)
    if result.solve:
        return ("SOLVE",)
    return ("FALLBACK", result.lo, result.hi, result.status)


@given(
    cells=_CELLS,
    status=st.sampled_from([Status.OK, *_FAILED_STATUSES]),
    # Mostly targets inside (0, 1), so the exact-test kinds are common.
    target=st.sampled_from([*_TARGETS * 6, 0.0, 1.0, 1.5, -0.1]),
    level=st.sampled_from(_LEVELS),
)
# One example per kind, so every branch of the table is always compared.
@example(
    cells=_cells_from([(0, 100), (0, 100), (0, 100)], first_severity=0),
    status=Status.SEPARATION,
    target=0.9,
    level=0.95,
)
@example(
    cells=_cells_from([(0, 100), (0, 100), (0, 100)], first_severity=0),
    status=Status.SEPARATION,
    target=1.0,
    level=0.95,
)
@example(
    cells=_cells_from([(100, 100), (100, 100), (100, 100)], first_severity=0),
    status=Status.NOT_CONVERGED,
    target=0.9,
    level=0.95,
)
@example(
    cells=_cells_from([(100, 100), (0, 100), (90, 100)], first_severity=0),
    status=Status.OK,
    target=0.9,
    level=0.95,
)
@example(
    cells=_cells_from([(100, 100), (84, 100), (0, 100)], first_severity=0),
    status=Status.SEPARATION,
    target=0.9,
    level=0.95,
)
@example(
    cells=_cells_from([(100, 100), (90, 100), (100, 100)], first_severity=0),
    status=Status.OK,
    target=0.9,
    level=0.95,
)
@example(
    cells=_cells_from([(90, 100), (90, 100), (90, 100)], first_severity=0),
    status=Status.OK,
    target=0.9,
    level=0.95,
)
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_classify_returns_the_kind_the_written_rules_give_on_arbitrary_cells(
    cells: tuple[Cell, ...], status: Status, target: float, level: float
) -> None:
    expected = _expected_kind(cells, status, target=target, level=level)
    result = classify(_fit(cells, status), target=target, level=level)

    event(f"kind: {expected[0]}")
    assert _observed_kind(result, status) == expected


# --------------------------------------------------------------------------------------------
# A higher level never narrows a claim
# --------------------------------------------------------------------------------------------


@given(
    cells=_CELLS,
    status=st.sampled_from(_FAILED_STATUSES),
    target=st.sampled_from(_TARGETS),
    levels=st.tuples(st.sampled_from(_LEVELS), st.sampled_from(_LEVELS)).map(sorted),
)
@example(
    cells=_cells_from([(100, 100), (100, 100), (49, 50), (90, 100)], first_severity=0),
    status=Status.SEPARATION,
    target=0.9,
    levels=[0.9, 0.99],
)
@example(
    cells=_cells_from([(90, 100), (0, 100), (0, 100), (84, 100), (0, 100)], first_severity=0),
    status=Status.NOT_CONVERGED,
    target=0.9,
    levels=[0.9, 0.99],
)
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_a_higher_level_never_narrows_a_failed_fits_exact_interval(
    cells: tuple[Cell, ...], status: Status, target: float, levels: list[float]
) -> None:
    low_level, high_level = levels
    fit = _failed_fit(cells, status)
    low = classify(fit, target=target, level=low_level)
    high = classify(fit, target=target, level=high_level)
    if not (_is_fallback(low, status) and _is_fallback(high, status)):
        event("one side is not an exact-bound fallback")
        return

    event("both are exact-bound fallbacks")
    # A missing lower bound is the widest claim (minus infinity), a missing upper bound likewise.
    low_lo = float("-inf") if low.lo is None else low.lo
    high_lo = float("-inf") if high.lo is None else high.lo
    low_hi = float("inf") if low.hi is None else low.hi
    high_hi = float("inf") if high.hi is None else high.hi
    assert high_lo <= low_lo, f"lo rose from {low.lo} to {high.lo} as level rose"
    assert high_hi >= low_hi, f"hi fell from {low.hi} to {high.hi} as level rose"
