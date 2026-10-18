"""The failed-fit exact bounds are the ends of fixed-sequence runs, every test at ``c4``.

When a fit fails (``SEPARATION`` / ``NOT_CONVERGED``) and neither ``RIGHT`` nor ``LEFT`` applies,
``censoring.classify`` reports exact bounds from the tested cells. Every exact claim the module
makes shares one error budget of ``1 - level``, split four ways, and each part is a one-sided
exact test at confidence ``c4 = 1 - (1 - level) / 4`` (0.9875 at ``level = 0.95``):

* the ascending run over the **nonzero** severities, which gives ``lo``;
* the top cell, which gives ``RIGHT``;
* the descending run over the **nonzero** severities, which gives ``hi``;
* the bottom cell, which gives ``hi = x1`` whenever it fails (and is also the upper half of
  ``LEFT`` and half of ``FAILS_AT_BASELINE``).

Each run is a fixed-sequence test (Hsu & Berger, 1999, *Journal of the American Statistical
Association* 94:468-482): it stops at the first severity that does not qualify, so under a
monotone curve it can only err by wrongly passing (or failing) the first severity past the
threshold, whatever the number of levels. A max-anywhere rule -- the highest confidently-passing
nonzero severity and the lowest confidently-failing one over *every* cell -- does not have that
property: with ``k`` cells near the target each side's error is up to about ``k`` times its
budget (the slow simulation in ``tests/coverage/test_exact_bracket_coverage.py`` measures it).

What follows from the split, and is pinned here:

* a side that one confidence level determines is reported at **that** level only: a one-sided
  bound is the end of its run at ``c4`` and is never extended by a rescan at the looser
  ``level``;
* there is no bracket-versus-one-sided distinction in the confidence, so a bracket and a
  one-sided bound are read off the same two runs;
* no bound is ever at the control: both runs are over nonzero severities;
* ``hi = x1`` whenever the lowest nonzero level fails at ``c4``, even when the descending run from
  the top is broken before it reaches ``x1``.

These tests build failed fits from chosen cells and read ``classify`` directly. A grid is written
as a string of roles, one character per severity (``0, 1, 2, ...``; the first is the control),
and every role is **verified against the exact bounds** before a test uses it, so a hand-picked
count that stopped meaning what its test says would fail loudly in setup. At ``target = 0.9``,
``level = 0.95`` (``c4 = 0.9875``):

====  ===========  ==============================================================
role  counts       meaning
====  ===========  ==============================================================
P     100 / 100    confidently passes at ``c4`` and at ``level``
p     49 / 50      confidently passes at ``level`` only; undetermined at ``c4``
G     90 / 100     undetermined (neither passes nor fails) at both
f     84 / 100     confidently fails at ``level`` only; undetermined at ``c4``
F     0 / 100      confidently fails at ``c4`` and at ``level``
====  ===========  ==============================================================

Each test where the two rules differ also asserts what the max-anywhere rule gives on the same
cells (``_max_anywhere_rule``) or what a rescan at ``level`` would give (``_runs_at``), so none of
them can pass on a grid where the rules happen to agree.

Pinned at the end and independent of the runs: the monotonicity-contradiction check, which looks
at *every* cell at ``level`` -- a fixed-sequence run would hide exactly the contradictions that
check exists to find -- and ``RIGHT`` / ``LEFT``, which are single-cell tests.
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
_C4 = 1.0 - (1.0 - _LEVEL) / 4.0

_ROLE_COUNTS: dict[str, tuple[int, int]] = {
    "P": (100, 100),
    "p": (49, 50),
    "G": (90, 100),
    "f": (84, 100),
    "F": (0, 100),
}

# role -> (passes at c4, passes at `level`, fails at c4, fails at `level`)
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
            _passes(cell, _C4),
            _passes(cell, _LEVEL),
            _fails(cell, _C4),
            _fails(cell, _LEVEL),
        )
        assert observed == _ROLE_MEANING[role], (
            f"setup: role {role!r} ({successes}/{trials}) has "
            f"(pass@{_C4}, pass@{_LEVEL}, fail@{_C4}, fail@{_LEVEL}) = {observed}, "
            f"expected {_ROLE_MEANING[role]}"
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
    """The max-anywhere rule at ``c4``, as the contrast to the runs: ``lo`` the highest nonzero
    severity whose lower bound clears the target, ``hi`` the lowest nonzero severity whose upper
    bound is below it, both over every nonzero cell and both at ``c4``. (Ignores the
    contradiction and ``RIGHT``/``LEFT`` pre-checks, which the grids that use this are chosen not
    to trigger.)"""
    cells = [c for c in _grid(roles) if c.severity > 0.0]
    lo_found = [c.severity for c in cells if _passes(c, _C4)]
    hi_found = [c.severity for c in cells if _fails(c, _C4)]
    return (max(lo_found) if lo_found else None, min(hi_found) if hi_found else None)


def _runs_at(roles: str, confidence: float) -> tuple[float | None, float | None]:
    """What a fixed-sequence run over the nonzero severities gives at ``confidence``, with no
    bottom-cell rule: the contrast for a side that a rescan at ``level`` would extend."""
    cells = [c for c in _grid(roles) if c.severity > 0.0]
    lo: float | None = None
    for cell in cells:
        if not _passes(cell, confidence):
            break
        lo = cell.severity
    hi: float | None = None
    for cell in reversed(cells):
        if not _fails(cell, confidence):
            break
        hi = cell.severity
    return lo, hi


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

    def test_the_control_is_never_a_bound_so_the_descending_run_ends_at_the_lowest_nonzero_level(
        self, failed_status: Status
    ) -> None:
        """Both runs are over the nonzero severities. With an undetermined control (``G``) and
        every nonzero level failing, ``hi`` is the lowest nonzero severity, never the control, and
        the bound is one-sided: the lower side has nothing to start from. (A *failing* control
        cannot appear in a fallback bracket at all: with the lowest nonzero level failing too the
        result is ``FAILS_AT_BASELINE``, and with a nonzero level passing it is the contradiction
        pinned below.)"""
        result = _classify("G F F F", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.lo is None
        assert result.hi == 1.0
        assert any("determined only the upper side" in w for w in result.warnings)


class TestTheLowestLevelsFailureGivesHiThereEvenWhenTheRunFromTheTopIsBroken:
    """The bottom cell is its own test at ``c4``. Where it fails, ``hi = x1`` is established
    directly, whatever the levels above it do; the descending run only matters when ``x1`` does
    not fail."""

    def test_a_broken_descending_run_still_gives_hi_at_the_lowest_failing_nonzero_level(
        self, failed_status: Status
    ) -> None:
        """``G F G F``: the run from the top is 3, then stops at the undetermined 2 (so a
        run-only rule gives ``hi = 3``). The lowest nonzero level, 1, fails on its own, so
        ``hi = 1``. The control is undetermined, so neither ``LEFT`` nor ``FAILS_AT_BASELINE``
        applies."""
        assert _runs_at("G F G F", _C4) == (None, 3.0)
        assert _max_anywhere_rule("G F G F") == (None, 1.0)

        result = _classify("G F G F", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.hi == 1.0
        assert result.lo is None
        assert any("determined only the upper side" in w for w in result.warnings)

    def test_an_intact_descending_run_gives_the_same_hi(self, failed_status: Status) -> None:
        """When the run from the top reaches ``x1`` the two routes agree."""
        result = _classify("G F F F", failed_status)

        assert result.hi == 1.0

    def test_a_lowest_level_that_fails_only_at_level_does_not_give_hi_there(
        self, failed_status: Status
    ) -> None:
        """``G f F``: level 1 fails at ``level`` but not at ``c4``, so it is not a test that
        fired. The run from the top is 2, then stops at 1: ``hi = 2``."""
        result = _classify("G f F", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.hi == 2.0
        assert result.lo is None


class TestAGapAtTheStartLeavesThatSideUndetermined:
    """A run that cannot begin is empty. A max-anywhere rule reaches past the gap to a later
    pass; the fixed-sequence runs have no bound on that side, so the result is one-sided (or
    empty) and carries the one-sided / no-bounds warning."""

    def test_gap_at_the_lowest_nonzero_level_leaves_the_lower_side_undetermined(
        self, failed_status: Status
    ) -> None:
        """``P G P P F F``: the lowest nonzero level (1) is undetermined while 2 and 3 pass.
        The max-anywhere rule reports ``(3, 4)``; there is no ``lo`` here, and ``hi`` is the
        upper-side run ``5, 4`` (3 passes, so it stops), reported one-sided."""
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
        rule reports ``(2, 3)``; the descending run cannot start, so only the lower side is
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
        """``P G P F G``: level 2 passes and level 3 fails, but both runs are stopped by an
        undetermined level at their own end. The max-anywhere rule reports ``(2, 3)``. The cells
        are monotone (2 passes below 3 failing), so this is *not* a contradiction -- it is simply
        no bound, with the no-bounds warning."""
        assert _max_anywhere_rule("P G P F G") == (2.0, 3.0)

        result = _classify("P G P F G", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.lo is None and result.hi is None
        assert any("no bounds are reported" in w for w in result.warnings)
        assert not any(w.startswith("NON_MONOTONE: ") for w in result.warnings)

    def test_a_missing_upper_side_says_neither_end_nonzero_severity_confidently_fails(
        self, failed_status: Status
    ) -> None:
        """``P P P F G``: ``hi`` is missing because the lowest nonzero level (1, ``P``) does not
        fail and the highest nonzero level (4, ``G``) does not fail, so the descending run cannot
        start. The warning says so of both ends, and calls both of them *nonzero* severities."""
        result = _classify("P P P F G", failed_status)

        (warning,) = result.warnings
        assert "determined only the lower side" in warning
        assert "neither the lowest nor the highest nonzero severity confidently fails" in warning

    def test_a_missing_lower_side_says_the_lowest_nonzero_severity_does_not_confidently_pass(
        self, failed_status: Status
    ) -> None:
        """``P G F F F``: ``lo`` is missing because the lowest nonzero level (1) is undetermined, so
        the ascending run cannot start."""
        result = _classify("P G F F F", failed_status)

        (warning,) = result.warnings
        assert "determined only the upper side" in warning
        assert "the lowest nonzero severity does not confidently pass" in warning

    def test_no_bounds_says_neither_end_nonzero_severity_confidently_fails(
        self, failed_status: Status
    ) -> None:
        """``P G G G``: the ascending run cannot start (level 1 is undetermined) and neither the
        lowest nor the highest nonzero level fails."""
        result = _classify("P G G G", failed_status)

        (warning,) = result.warnings
        assert "no bounds are reported" in warning
        assert "neither the lowest nor the highest nonzero severity confidently fails" in warning


class TestAOneSidedBoundIsTheEndOfItsRunAtC4NeverExtendedAtLevel:
    """When only one side is determined, it is reported as the end of its run **at ``c4``**.
    There is no rescan at the looser ``level``: a level that passes (or fails) only at ``level``
    ends the run. Each grid is built so that a rescan at ``level`` would give a different answer,
    and the contrast is asserted."""

    def test_lower_run_stops_at_the_first_gap(self, failed_status: Status) -> None:
        """``P P G P G``: 1 passes, 2 is undetermined, 3 passes, 4 is undetermined (so nothing
        fails and only the lower side is determined). Max-anywhere gives ``lo = 3``; the run from
        the bottom ends at 1."""
        assert _max_anywhere_rule("P P G P G") == (3.0, None)

        result = _classify("P P G P G", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.lo == 1.0
        assert result.hi is None
        assert any("determined only the lower side" in w for w in result.warnings)

    def test_lower_run_is_not_extended_by_a_level_that_passes_only_at_level(
        self, failed_status: Status
    ) -> None:
        """``P P p G P G``: level 2 passes at ``level`` but not at ``c4``. The run at ``c4`` is
        level 1 alone. A rescan at ``level`` would extend it to 2 before stopping at the gap at 3,
        and max-anywhere would give 4; the answer is 1."""
        assert _runs_at("P P p G P G", _LEVEL) == (2.0, None)
        assert _max_anywhere_rule("P P p G P G") == (4.0, None)

        result = _classify("P P p G P G", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.lo == 1.0
        assert result.hi is None
        assert any("determined only the lower side" in w for w in result.warnings)

    def test_upper_run_stops_at_the_first_gap(self, failed_status: Status) -> None:
        """``P G F G F F``: nothing passes (so only the upper side can be determined), the run
        from the top is 5, 4, then stops at the undetermined 3, and level 2 fails but lies behind
        the gap. Max-anywhere gives ``hi = 2``."""
        assert _max_anywhere_rule("P G F G F F") == (None, 2.0)

        result = _classify("P G F G F F", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.lo is None
        assert result.hi == 4.0
        assert any("determined only the upper side" in w for w in result.warnings)

    def test_upper_run_is_not_extended_by_a_level_that_fails_only_at_level(
        self, failed_status: Status
    ) -> None:
        """``P G F G f F``: the run from the top is 5, then stops at 4, which fails at ``level``
        but not at ``c4``. A rescan at ``level`` would extend it to 4 before stopping at the gap
        at 3; the answer is 5."""
        assert _runs_at("P G F G f F", _LEVEL) == (None, 4.0)

        result = _classify("P G F G f F", failed_status)

        _assert_fallback_shape(result, failed_status)
        assert result.lo is None
        assert result.hi == 5.0
        assert any("determined only the upper side" in w for w in result.warnings)


@dataclasses.dataclass(frozen=True)
class _CleanCase:
    roles: str
    lo: float | None
    hi: float | None


# Passes contiguous from the bottom and fails contiguous from the top, at ``c4``: the fixed-sequence
# runs and the max-anywhere rule at ``c4`` must agree on every one of these.
_CLEAN_CASES = [
    _CleanCase("P P P G F F", lo=2.0, hi=4.0),
    _CleanCase("P P F F", lo=1.0, hi=2.0),
    _CleanCase("P P P P P F", lo=4.0, hi=5.0),
    _CleanCase("G P P G F", lo=2.0, hi=4.0),
    _CleanCase("P P P p G f F F", lo=2.0, hi=6.0),
    _CleanCase("P P P P G", lo=3.0, hi=None),
    _CleanCase("P P P p G", lo=2.0, hi=None),
    _CleanCase("P G F F F", lo=None, hi=2.0),
    _CleanCase("P G G f F", lo=None, hi=4.0),
]


class TestCleanDataAgreesWithTheMaxAnywhereRuleAtC4:
    """When the cells that confidently pass at ``c4`` are contiguous from the bottom and the
    ones that confidently fail at ``c4`` are contiguous from the top, the runs and the
    max-anywhere rule at ``c4`` are identical. (A level that passes or fails only at ``level``
    is neither, so it ends a run: ``P P P p G`` has ``lo = 2``, not 3.)"""

    @pytest.mark.parametrize("case", _CLEAN_CASES, ids=lambda c: c.roles.replace(" ", ""))
    def test_contiguous_passes_and_fails_give_the_same_bounds_as_the_max_anywhere_rule(
        self, case: _CleanCase, failed_status: Status
    ) -> None:
        # The cases are clean by construction; the oracle confirms the max-anywhere rule agrees
        # with the explicit expectation, so a typo in `_CLEAN_CASES` cannot make the test vacuous.
        assert _max_anywhere_rule(case.roles) == (case.lo, case.hi)

        result = _classify(case.roles, failed_status)

        _assert_fallback_shape(result, failed_status)
        assert (result.lo, result.hi) == (case.lo, case.hi)

    def test_a_bracket_carries_no_warning_and_a_one_sided_bound_names_its_side_and_c4(
        self, failed_status: Status
    ) -> None:
        bracket = _classify("P P P G F F", failed_status)
        one_sided = _classify("P P P P G", failed_status)

        assert bracket.warnings == ()
        assert len(one_sided.warnings) == 1
        # Names the side that was determined and the confidence it was determined at (c4, 0.9875
        # at level 0.95). There is no second confidence: nothing is rescanned at `level`, and a
        # Bonferroni split of the budget over two sides (0.975) does not appear.
        (warning,) = one_sided.warnings
        assert "determined only the lower side" in warning
        assert "confidence 0.9875" in warning
        assert "rescanned" not in warning
        assert "0.975" not in warning


class TestTheContradictionCheckIsUnchanged:
    """``_contradiction_or_none`` compares the highest confidently
    passing nonzero severity with the lowest confidently failing one, over every cell, at
    ``level``. The grids below are ones where the contradicting cells sit *behind* an
    undetermined or failing level that would stop a fixed-sequence run, so a contradiction
    check that had been folded into the runs would report 'no bounds' (or a bracket)
    instead of naming the contradiction. It can only suppress a claim, never add one."""

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
        # `c4` the exact claims use.
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


class TestRightAndLeftAreSingleCellTests:
    """``RIGHT`` and ``LEFT`` each test one cell chosen in advance at ``c4`` and are not runs.
    Pinned on grids that also contain gaps, where a fixed-sequence run applied to them would
    change the answer."""

    def test_right_fires_on_the_highest_level_alone(self, failed_status: Status) -> None:
        """Level 2 is undetermined, but the highest level (3) confidently passes: ``RIGHT`` at
        ``x_max``. It is the single pre-chosen cell, not the end of a run (the run from the
        bottom would stop at 2)."""
        result = _classify("P P G P", failed_status)

        assert result.censoring is Censoring.RIGHT
        assert result.lo == 3.0
        assert result.hi is None
        assert result.status is failed_status

    def test_left_fires_on_the_lowest_nonzero_level_alone(self, failed_status: Status) -> None:
        """The control passes and the lowest nonzero level (1) confidently fails: ``LEFT`` at
        that level, even though level 2 is undetermined (the run from the top would stop at
        2)."""
        result = _classify("P F G G", failed_status)

        assert result.censoring is Censoring.LEFT
        assert result.hi == 1.0
        assert result.lo is None
        assert result.status is failed_status


class TestUnsortedCellsCannotReachTheExactTestRules:
    """Regression: every exact-test rule reads ``fit.cells`` in order -- ``RIGHT`` the last
    cell, ``LEFT`` the first nonzero cell, the runs from each end -- so if ``Fit`` did not check
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
