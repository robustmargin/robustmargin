"""Property tests for the failed-fit exact bracket.

Two properties of ``censoring.classify`` on a failed fit (``SEPARATION`` / ``NOT_CONVERGED``)
whose result is an exact-bound fallback (censoring ``NONE``, not ``FAILS_AT_BASELINE``, no
``NON_MONOTONE: `` contradiction):

* **The run property (any data).** A reported ``lo`` is the end of an unbroken run: every nonzero
  severity at or below it confidently passes, and the next nonzero severity above it does not.
  A reported ``hi`` is the mirror image from the top, control included. A side that is not
  reported is one whose run cannot begin: the lowest nonzero severity does not confidently pass
  (or the highest severity does not confidently fail) at the bracket confidence. The
  confidence of a reported side is ``(1 + level) / 2`` when both are reported and ``level``
  when only one is. This is the definition of a fixed-sequence scan, checked on arbitrary
  counts rather than on hand-picked grids; it fails on the max-anywhere rule as soon as a
  passing (or failing) cell sits behind a gap.
* **Clean data is unchanged.** When the confidently passing cells form a prefix of the nonzero
  severities and the confidently failing cells a suffix of all severities, at both confidences,
  the result equals what a max-anywhere rule would give: when the confidently passing and
  failing cells are each contiguous from their end, the two rules are identical.

Examples are small and the search is hypothesis's own (seeded, deterministic) one; the explicit
``@example`` cases keep both properties from being vacuous by guaranteeing the interesting
non-contiguous shapes are always tried.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Iterator

import pytest
from hypothesis import event, example, given, settings
from hypothesis import strategies as st

from robustmargin import Fit, Status, censoring
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
    bracket_confidence = (1.0 + level) / 2.0
    nonzero = [c for c in cells if c.severity > 0.0]
    lo, hi = result.lo, result.hi
    confidence = bracket_confidence if (lo is not None and hi is not None) else level

    if lo is not None:
        assert lo in {c.severity for c in nonzero}
        for c in nonzero:
            if c.severity <= lo:
                assert _lower(c, confidence) >= target, (
                    f"lo={lo} but nonzero severity {c.severity} at or below it does not "
                    f"confidently pass at {confidence}: not an unbroken run"
                )
        above = [c for c in nonzero if c.severity > lo]
        if above:
            assert _lower(above[0], confidence) < target, (
                f"lo={lo} stops short: the next nonzero severity {above[0].severity} "
                f"confidently passes at {confidence}"
            )
    elif nonzero:
        assert _lower(nonzero[0], bracket_confidence) < target, (
            "no lo reported, but the lowest nonzero severity confidently passes at the bracket "
            "confidence, so the run could begin"
        )

    if hi is not None:
        assert hi in {c.severity for c in cells}
        for c in cells:
            if c.severity >= hi:
                assert _upper(c, confidence) < target, (
                    f"hi={hi} but severity {c.severity} at or above it does not confidently "
                    f"fail at {confidence}: not an unbroken run"
                )
        below = [c for c in cells if c.severity < hi]
        if below:
            assert _upper(below[-1], confidence) >= target, (
                f"hi={hi} stops short: the next lower severity {below[-1].severity} "
                f"confidently fails at {confidence}"
            )
    else:
        assert _upper(cells[-1], bracket_confidence) >= target, (
            "no hi reported, but the highest severity confidently fails at the bracket "
            "confidence, so the run could begin"
        )


def _cells_from(counts: list[tuple[int, int]], *, first_severity: int) -> tuple[Cell, ...]:
    return tuple(
        Cell(severity=float(first_severity + i), successes=s, trials=n)
        for i, (s, n) in enumerate(counts)
    )


@given(
    cells=_arbitrary_cells(),
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
# Clean data: the new rule equals the old one
# --------------------------------------------------------------------------------------------

# At target 0.9, level 0.95 (bracket confidence 0.975); each role is verified against
# `exact_one_sided_bound` in the unit tests (`tests/unit/test_censoring_fixed_sequence.py`).
_ROLE_COUNTS: dict[str, tuple[int, int]] = {
    "P": (100, 100),  # passes at both confidences
    "p": (49, 50),  # passes at 0.95 only
    "G": (90, 100),  # undetermined at both
    "f": (84, 100),  # fails at 0.95 only
    "F": (0, 100),  # fails at both
}
_CLEAN_TARGET = 0.9
_CLEAN_LEVEL = 0.95


def _max_anywhere_rule(
    cells: tuple[Cell, ...], *, target: float, level: float
) -> tuple[float | None, float | None]:
    """The max-anywhere rule, as an oracle for clean data: ``lo`` the highest nonzero severity
    whose lower bound clears the target, ``hi`` the lowest severity whose upper bound is below
    it, over every cell at the bracket confidence; one determined side is rescanned at ``level``
    and the other forced to ``None``."""

    def lo_at(confidence: float) -> float | None:
        found = [c.severity for c in cells if c.severity > 0.0 and _lower(c, confidence) >= target]
        return max(found) if found else None

    def hi_at(confidence: float) -> float | None:
        found = [c.severity for c in cells if _upper(c, confidence) < target]
        return min(found) if found else None

    bracket_confidence = (1.0 + level) / 2.0
    lo, hi = lo_at(bracket_confidence), hi_at(bracket_confidence)
    if lo is not None and hi is not None:
        return lo, hi
    if lo is None and hi is None:
        return None, None
    return (lo_at(level), None) if hi is None else (None, hi_at(level))


@st.composite
def _clean_roles(draw: st.DrawFn) -> str:
    """``<control> P* p* G* f* F*``: passes a prefix of the nonzero severities and fails a suffix
    at both confidences. The control is ``P`` or ``G`` (a failing control is either
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
def test_clean_data_gives_the_same_bounds_as_the_max_anywhere_rule(
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
