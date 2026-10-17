"""The failed-fit exact bracket takes each side from a fixed-sequence scan.

When a fit fails (``SEPARATION`` / ``NOT_CONVERGED``) and neither ``RIGHT`` nor ``LEFT`` applies,
``censoring.classify`` reports an exact bracket from the tested cells. Its coverage claim: each
side at ``(1 + level) / 2`` so the pair covers at least ``level``; a one-sided bound at
``level``. That argument needs each side to come from **one cell chosen in advance**. A
max-anywhere rule -- the highest confidently-passing nonzero severity and the lowest
confidently-failing severity over *every* cell -- does not satisfy it: with ``k`` cells near the
target each side's error is up to about ``k * (1 - c)`` (the slow simulation in
``tests/coverage/test_exact_bracket_coverage.py`` measures it).

The fixed-sequence scans:

* ``lo`` is the last severity in the run of confidently-passing **nonzero** severities ascending
  from the bottom;
* ``hi`` is the last severity in the run of confidently-failing severities, control included,
  descending from the top;
* each scan stops at the first severity that does not qualify;
* the confidences are unchanged, and the side is still decided at the bracket confidence and
  then rescanned at ``level``, with the same stop-at-the-first-gap rule.

These tests build failed fits from chosen cells and read ``classify`` directly. A grid is written
as a string of roles, one character per severity (``0, 1, 2, ...``; the first is the control),
and every role is **verified against the exact bounds** before a test uses it, so a hand-picked
count that stopped meaning what its test says would fail loudly in setup. At ``target = 0.9``,
``level = 0.95`` (bracket confidence ``0.975``):

====  ===========  ==============================================================
role  counts       meaning
====  ===========  ==============================================================
P     100 / 100    confidently passes at 0.975 and at 0.95
p     49 / 50      confidently passes at 0.95 only; undetermined at 0.975
G     90 / 100     undetermined (neither passes nor fails) at both
f     84 / 100     confidently fails at 0.95 only; undetermined at 0.975
F     0 / 100      confidently fails at 0.975 and at 0.95
====  ===========  ==============================================================

Each test where the two rules differ also asserts what the max-anywhere rule gives on the same
cells (``_max_anywhere_rule``), so none of them can pass on a grid where the two rules happen to
agree.

Pinned at the end and independent of the scans: the monotonicity-contradiction check, which
looks at *every* cell at ``level`` -- a fixed-sequence scan would hide exactly the contradictions
that check exists to find -- and ``RIGHT`` / ``LEFT``.
"""

from __future__ import annotations

import dataclasses

import pytest

from robustmargin import Fit, Status
from robustmargin.censoring import Classification, classify, exact_one_sided_bound
from robustmargin.testing import fake_fit
from robustmargin.types import Cell, Censoring

_TARGET = 0.9
_LEVEL = 0.95
_BRACKET_CONFIDENCE = (1.0 + _LEVEL) / 2.0

_ROLE_COUNTS: dict[str, tuple[int, int]] = {
    "P": (100, 100),
    "p": (49, 50),
    "G": (90, 100),
    "f": (84, 100),
    "F": (0, 100),
}

# role -> (passes at the bracket confidence, passes at `level`,
#          fails at the bracket confidence, fails at `level`)
_ROLE_MEANING: dict[str, tuple[bool, bool, bool, bool]] = {
    "P": (True, True, False, False),
    "p": (False, True, False, False),
    "G": (False, False, False, False),
    "f": (False, False, False, True),
    "F": (False, False, True, True),
}


def _passes(cell: Cell, confidence: float) -> bool:
    return (
        exact_one_sided_bound(cell.successes, cell.trials, level=confidence, side="lower")
        >= _TARGET
    )


def _fails(cell: Cell, confidence: float) -> bool:
    return (
        exact_one_sided_bound(cell.successes, cell.trials, level=confidence, side="upper") < _TARGET
    )


def _grid(roles: str) -> tuple[Cell, ...]:
    """Cells at severities ``0.0, 1.0, 2.0, ...`` from a role string (see the module docstring).

    Whitespace is ignored. Each role's meaning is checked against ``exact_one_sided_bound``
    here, so a test never silently runs on a cell that is not what its name says.
    """
    cells: list[Cell] = []
    for severity, role in enumerate(roles.replace(" ", "")):
        successes, trials = _ROLE_COUNTS[role]
        cell = Cell(severity=float(severity), successes=successes, trials=trials)
        observed = (
            _passes(cell, _BRACKET_CONFIDENCE),
            _passes(cell, _LEVEL),
            _fails(cell, _BRACKET_CONFIDENCE),
            _fails(cell, _LEVEL),
        )
        assert observed == _ROLE_MEANING[role], (
            f"setup: role {role!r} ({successes}/{trials}) has "
            f"(pass@{_BRACKET_CONFIDENCE}, pass@{_LEVEL}, fail@{_BRACKET_CONFIDENCE}, "
            f"fail@{_LEVEL}) = {observed}, expected {_ROLE_MEANING[role]}"
        )
        cells.append(cell)
    return tuple(cells)


def _failed_fit(cells: tuple[Cell, ...], status: Status) -> Fit:
    """A real ``Fit`` of a failed status carrying the chosen cells. ``classify`` reads only
    ``status`` and ``cells``; everything else is the schema-valid filler from ``fake_fit``."""
    return dataclasses.replace(fake_fit(status=status, monotonicity=None), cells=cells)


def _classify(roles: str, status: Status) -> Classification:
    return classify(_failed_fit(_grid(roles), status), target=_TARGET, level=_LEVEL)


def _max_anywhere_rule(roles: str) -> tuple[float | None, float | None]:
    """The max-anywhere rule, as the contrast to the scans: ``lo`` the highest nonzero
    severity whose lower bound clears the target, ``hi`` the lowest severity whose upper bound is
    below it, both over every cell, each side at the bracket confidence; when exactly one side
    is determined, that side is rescanned the same way at ``level`` and the other forced to
    ``None``. (Ignores the contradiction and ``RIGHT``/``LEFT`` pre-checks, which the grids
    that use this are chosen not to trigger.)"""
    cells = _grid(roles)

    def lo_at(confidence: float) -> float | None:
        found = [
            c.severity
            for c in cells
            if c.severity > 0.0
            and exact_one_sided_bound(c.successes, c.trials, level=confidence, side="lower")
            >= _TARGET
        ]
        return max(found) if found else None

    def hi_at(confidence: float) -> float | None:
        found = [
            c.severity
            for c in cells
            if exact_one_sided_bound(c.successes, c.trials, level=confidence, side="upper")
            < _TARGET
        ]
        return min(found) if found else None

    lo, hi = lo_at(_BRACKET_CONFIDENCE), hi_at(_BRACKET_CONFIDENCE)
    if lo is not None and hi is not None:
        return lo, hi
    if lo is None and hi is None:
        return None, None
    return (lo_at(_LEVEL), None) if hi is None else (None, hi_at(_LEVEL))


def _assert_fallback_shape(result: Classification, status: Status) -> None:
    """What every exact-bound fallback result looks like, whatever bounds it carries."""
    assert result.status is status
    assert result.censoring is Censoring.NONE
    assert result.baseline is None
    assert result.grid is not None
    assert result.solve is False


@pytest.fixture(params=[Status.SEPARATION, Status.NOT_CONVERGED], ids=lambda s: s.name)
def failed_status(request: pytest.FixtureRequest) -> Status:
    status: Status = request.param
    return status


class TestEachSideStopsAtTheFirstSeverityThatDoesNotQualify:
    """The run, not the extreme."""

    def test_lower_bound_is_the_end_of_the_bottom_run_not_a_later_confident_pass(
        self, failed_status: Status
    ) -> None:
        """``P P P G P F F``: severities 1 and 2 pass, 3 is undetermined, 4 passes again.

        The max-anywhere rule takes the highest passing severity anywhere, ``lo = 4``. The run from
        the bottom ends at 2. (Not a monotonicity contradiction: the highest passing severity,
        4, is below the lowest failing one, 5.)
        """
        assert _max_anywhere_rule("P P P G P F F") == (4.0, 5.0)

        result = _classify("P P P G P F F", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.lo == 2.0, "lo must end at the first gap, not jump to the later pass at 4"
        assert result.hi == 5.0
        assert result.warnings == ()

    def test_upper_bound_is_the_end_of_the_top_run_not_a_lower_confident_fail(
        self, failed_status: Status
    ) -> None:
        """``P P P P F G F F``: 7 and 6 fail, 5 is undetermined, 4 fails again.

        The max-anywhere rule takes the lowest failing severity anywhere, ``hi = 4``. The run from
        the top ends at 6. The lower side is the ordinary run 1..3 (4 fails, so it stops there).
        """
        assert _max_anywhere_rule("P P P P F G F F") == (3.0, 4.0)

        result = _classify("P P P P F G F F", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.hi == 6.0, "hi must end at the first gap from the top, not at the fail at 4"
        assert result.lo == 3.0
        assert result.warnings == ()

    def test_an_undetermined_control_ends_the_descending_scan_at_the_lowest_nonzero_level(
        self, failed_status: Status
    ) -> None:
        """The upper scan runs through the control, so an undetermined control ends it there and
        ``hi`` is the lowest nonzero severity. (A *failing* control cannot appear in a fallback
        bracket at all: with no nonzero severity passing the result is ``FAILS_AT_BASELINE``,
        and with one passing it is the contradiction pinned below.)"""
        result = _classify("G F F F", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.lo is None
        assert result.hi == 1.0
        assert any("determined only the upper side" in w for w in result.warnings)


class TestAGapAtTheStartLeavesThatSideUndetermined:
    """A run that cannot begin is empty. The old rule reached past the gap to a later pass; the
    fixed-sequence rule has no bound on that side, so the result is one-sided (or empty) and
    carries the existing one-sided / no-bounds warning."""

    def test_gap_at_the_lowest_nonzero_level_leaves_the_lower_side_undetermined(
        self, failed_status: Status
    ) -> None:
        """``P G P P F F``: the lowest nonzero level (1) is undetermined while 2 and 3 pass.
        The max-anywhere rule reports the bracket ``(3, 4)``; there is now no ``lo``, and ``hi`` is
        the upper-side run ``5, 4`` (3 passes, so it stops), reported one-sided at ``level``."""
        assert _max_anywhere_rule("P G P P F F") == (3.0, 4.0)

        result = _classify("P G P P F F", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.lo is None
        assert result.hi == 4.0
        assert any("determined only the upper side" in w for w in result.warnings)
        assert not any(w.startswith("NON_MONOTONE: ") for w in result.warnings)

    def test_gap_at_the_highest_level_leaves_the_upper_side_undetermined(
        self, failed_status: Status
    ) -> None:
        """``P P P F G``: the highest level (4) is undetermined while 3 fails. The max-anywhere
        rule reports ``(2, 3)``; the descending scan cannot start, so only the lower side is
        bound."""
        assert _max_anywhere_rule("P P P F G") == (2.0, 3.0)

        result = _classify("P P P F G", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.lo == 2.0
        assert result.hi is None
        assert any("determined only the lower side" in w for w in result.warnings)
        assert not any(w.startswith("NON_MONOTONE: ") for w in result.warnings)

    def test_gaps_at_both_ends_report_no_bounds_and_no_contradiction(
        self, failed_status: Status
    ) -> None:
        """``P G P F G``: level 2 passes and level 3 fails, but both scans are stopped by an
        undetermined level at their own end. The max-anywhere rule reports ``(2, 3)``. The cells are
        monotone (2 passes below 3 failing), so this is *not* a contradiction -- it is simply no
        bound, with the existing no-bounds warning."""
        assert _max_anywhere_rule("P G P F G") == (2.0, 3.0)

        result = _classify("P G P F G", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.lo is None and result.hi is None
        assert any("no bounds are reported" in w for w in result.warnings)
        assert not any(w.startswith("NON_MONOTONE: ") for w in result.warnings)


class TestTheOneSidedRescanAtLevelAlsoStopsAtTheFirstGap:
    """When only one side is determined at the bracket confidence, that
    side is rescanned at ``level`` -- with the same fixed-sequence rule, not the max-anywhere
    one. Each grid here is built so that the rescan result differs from the max-anywhere answer
    at ``level``."""

    def test_lower_rescan_stops_at_the_first_gap(self, failed_status: Status) -> None:
        """``P P G P G``: 1 passes, 2 is undetermined, 3 passes, 4 is undetermined (so nothing
        fails and only the lower side is determined). Max-anywhere at ``level`` gives ``lo = 3``;
        the run from the bottom ends at 1."""
        assert _max_anywhere_rule("P P G P G") == (3.0, None)

        result = _classify("P P G P G", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.lo == 1.0
        assert result.hi is None
        assert any("determined only the lower side" in w for w in result.warnings)

    def test_lower_rescan_extends_the_run_at_level_then_stops_at_the_next_gap(
        self, failed_status: Status
    ) -> None:
        """``P P p G P G``: at the bracket confidence the run is just level 1 (level 2 passes
        only at ``level``); the rescan at ``level`` extends it to 2, then stops at the gap at 3.
        The answer, 2, is neither the bracket-confidence run's end (1) nor the max-anywhere
        answer (4): it pins both that the rescan happens at ``level`` and that it is a run."""
        assert _max_anywhere_rule("P P p G P G") == (4.0, None)

        result = _classify("P P p G P G", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.lo == 2.0
        assert result.hi is None
        assert any("determined only the lower side" in w for w in result.warnings)

    def test_upper_rescan_stops_at_the_first_gap(self, failed_status: Status) -> None:
        """``P G F G f F``: nothing passes (so only the upper side can be determined), the run
        from the top is 5 then 4 (which fails only at ``level``, so it is reached by the rescan
        alone), and 3 is undetermined. Max-anywhere at ``level`` gives ``hi = 2``."""
        assert _max_anywhere_rule("P G F G f F") == (None, 2.0)

        result = _classify("P G F G f F", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.lo is None
        assert result.hi == 4.0
        assert any("determined only the upper side" in w for w in result.warnings)


@dataclasses.dataclass(frozen=True)
class _CleanCase:
    roles: str
    lo: float | None
    hi: float | None


# Passes contiguous from the bottom and fails contiguous from the top, at both confidences: the
# fixed-sequence rule and the max-anywhere rule must agree on every one of these.
_CLEAN_CASES = [
    _CleanCase("P P P G F F", lo=2.0, hi=4.0),
    _CleanCase("P P F F", lo=1.0, hi=2.0),
    _CleanCase("P P P P P F", lo=4.0, hi=5.0),
    _CleanCase("G P P G F", lo=2.0, hi=4.0),
    _CleanCase("P P P p G f F F", lo=2.0, hi=6.0),
    _CleanCase("P P P P G", lo=3.0, hi=None),
    _CleanCase("P P P p G", lo=3.0, hi=None),
    _CleanCase("P G F F F", lo=None, hi=2.0),
    _CleanCase("P G G f F", lo=None, hi=3.0),
]


class TestCleanDataIsUnchanged:
    """When the confidently passing and failing cells are each contiguous from their end, the
    result is identical to the max-anywhere rule's."""

    @pytest.mark.parametrize("case", _CLEAN_CASES, ids=lambda c: c.roles.replace(" ", ""))
    def test_contiguous_passes_and_fails_give_the_same_bounds_as_the_max_anywhere_rule(
        self, case: _CleanCase, failed_status: Status
    ) -> None:
        # The cases are clean by construction; the oracle confirms the old rule agrees with the
        # explicit expectation, so a typo in `_CLEAN_CASES` cannot make the test vacuous.
        assert _max_anywhere_rule(case.roles) == (case.lo, case.hi)

        result = _classify(case.roles, failed_status)

        _assert_fallback_shape(result, failed_status)
        assert (result.lo, result.hi) == (case.lo, case.hi)

    def test_a_bracket_carries_no_warning_and_a_one_sided_bound_carries_the_existing_one(
        self, failed_status: Status
    ) -> None:
        bracket = _classify("P P P G F F", failed_status)
        one_sided = _classify("P P P P G", failed_status)

        assert bracket.warnings == ()
        assert len(one_sided.warnings) == 1
        # Names the bracket confidence the side was *not* determined at (0.975), and the
        # confidence the reported one-sided bound was rescanned at (0.95); the first must not be
        # confused with the second.
        assert (
            "determined only the lower side at the bracket confidence 0.975"
            in one_sided.warnings[0]
        )
        assert "rescanned at confidence 0.95" in one_sided.warnings[0]


class TestTheContradictionCheckIsUnchanged:
    """``_contradiction_or_none`` compares the highest confidently
    passing nonzero severity with the lowest confidently failing one, over every cell, at
    ``level``. The grids below are ones where the contradicting cells sit *behind* an
    undetermined or failing level that would stop a fixed-sequence scan, so a contradiction
    check that had been folded into the new scans would report 'no bounds' (or a bracket)
    instead of naming the contradiction."""

    # roles, the severity that passes, the lower severity that fails
    _CONTRADICTIONS = [
        pytest.param("P F P", 2.0, 1.0, id="fail-below-a-pass"),
        pytest.param("P G F P G", 3.0, 2.0, id="gaps-at-both-ends-hide-it"),
        pytest.param("F P P", 2.0, 0.0, id="failing-control-below-passes"),
    ]

    @pytest.mark.parametrize(("roles", "passing", "failing"), _CONTRADICTIONS)
    def test_non_monotone_cells_give_the_non_monotone_warning_and_no_bounds(
        self, roles: str, passing: float, failing: float, failed_status: Status
    ) -> None:
        result = _classify(roles, failed_status)

        assert result.status is failed_status
        assert result.censoring is Censoring.NONE
        assert result.lo is None and result.hi is None
        assert result.grid is not None
        assert result.baseline is None
        assert len(result.warnings) == 1
        (warning,) = result.warnings
        assert warning.startswith("NON_MONOTONE: ")
        # Both contradicting severities are named, and the check is at `level`, not at the
        # bracket confidence the fallback scans use.
        assert f"severity {passing} confidently passes" in warning
        assert f"the lower severity {failing} confidently fails" in warning
        assert f"at confidence {_LEVEL} contradict monotonicity" in warning
        assert "no bounds are reported" in warning

    def test_a_converged_fit_still_gets_the_warning_and_solves(self) -> None:
        """The same check on an ``OK`` fit: it suppresses ``RIGHT``/``LEFT`` and falls through to
        the solve path, carrying the warning (the 'fitted curve' wording differs by design)."""
        ok_fit = dataclasses.replace(
            fake_fit(diagnostics=None, monotonicity=None), cells=_grid("P G F P G")
        )

        result = classify(ok_fit, target=_TARGET, level=_LEVEL)

        assert result.status is Status.OK
        assert result.censoring is Censoring.NONE
        assert result.solve is True
        assert result.lo is None and result.hi is None
        assert len(result.warnings) == 1
        assert result.warnings[0].startswith("NON_MONOTONE: ")
        assert "severity 3.0 confidently passes" in result.warnings[0]
        assert "the lower severity 2.0 confidently fails" in result.warnings[0]


class TestRightAndLeftAreUnchanged:
    """``RIGHT`` and ``LEFT`` each test one cell chosen in advance and are not scans. Pinned on
    grids that also contain gaps, where a fixed-sequence rule applied to them would change the
    answer."""

    def test_right_still_fires_on_the_highest_level_alone(self, failed_status: Status) -> None:
        """Level 2 is undetermined, but the highest level (3) confidently passes: ``RIGHT`` at
        ``x_max``, exactly as before. It is the single pre-chosen cell, not the end of a run."""
        result = _classify("P P G P", failed_status)

        assert result.censoring is Censoring.RIGHT
        assert result.lo == 3.0
        assert result.hi is None
        assert result.status is failed_status

    def test_left_still_fires_on_the_lowest_nonzero_level_alone(
        self, failed_status: Status
    ) -> None:
        """The control passes and the lowest nonzero level (1) confidently fails: ``LEFT`` at
        that level, even though level 2 is undetermined."""
        result = _classify("P F G G", failed_status)

        assert result.censoring is Censoring.LEFT
        assert result.hi == 1.0
        assert result.lo is None
        assert result.status is failed_status


class TestUnsortedCellsCannotReachTheExactTestRules:
    """Regression: every exact-test rule reads ``fit.cells`` in order -- ``RIGHT`` the last
    cell, ``LEFT`` the first nonzero cell, the scans from each end -- so if ``Fit`` did not check
    that order, a hand-built (or hand-edited) fit with its cells out of order would make
    ``classify`` test the wrong levels with no error. Out-of-order cells fail at ``Fit``
    construction.

    Each case below is a pair: the unsorted cells (which would give the wrong answer, quoted in
    the docstring) must not build, and the same cells in ascending order give the right one.
    """

    def test_cells_that_would_give_left_at_the_wrong_severity_cannot_be_built(
        self, failed_status: Status
    ) -> None:
        """``P F F F`` with the severity-3 cell listed first among the nonzero ones. In list
        order the "first nonzero cell" would be severity 3, so ``classify`` would return ``LEFT``
        with ``hi = 3.0``; the lowest failing level is 1.0."""
        control, one, two, three = _grid("P F F F")

        with pytest.raises(ValueError):
            _failed_fit((control, three, one, two), failed_status)

    def test_cells_that_would_give_right_at_a_passing_level_below_a_failing_one_cannot_be_built(
        self, failed_status: Status
    ) -> None:
        """``P P F`` with the failing severity-2 cell listed before the passing severity-1 one.
        In list order the "last cell" would be severity 1, which passes, so ``classify`` would
        return ``RIGHT`` with ``lo = 1.0`` -- a claim that no tested level fails, with a
        confidently failing level at 2.0."""
        control, one, two = _grid("P P F")

        with pytest.raises(ValueError):
            _failed_fit((control, two, one), failed_status)

    def test_the_left_cells_in_ascending_order_give_left_at_the_lowest_failing_severity(
        self, failed_status: Status
    ) -> None:
        result = _classify("P F F F", failed_status)

        assert result.censoring is Censoring.LEFT
        assert result.hi == 1.0
        assert result.lo is None
        assert result.status is failed_status

    def test_the_right_cells_in_ascending_order_give_the_bracket_not_right(
        self, failed_status: Status
    ) -> None:
        result = _classify("P P F", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.lo == 1.0
        assert result.hi == 2.0

    def test_replacing_the_cells_of_a_valid_fit_with_unsorted_ones_raises(
        self, failed_status: Status
    ) -> None:
        """The route these tests use to hand-build a fit, ``dataclasses.replace``, runs the same
        check, so no test of ``classify`` can pass unsorted cells in by accident."""
        control, one, two = _grid("P P F")
        valid = _failed_fit((control, one, two), failed_status)

        with pytest.raises(ValueError):
            dataclasses.replace(valid, cells=(control, two, one))
