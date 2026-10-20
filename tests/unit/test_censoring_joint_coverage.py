"""Each exact result ``censoring.classify`` returns, and the exact condition that produces it.

Every exact claim -- ``FAILS_AT_BASELINE``, ``RIGHT``, ``LEFT``, and a failed fit's bracket or
one-sided bound -- is read off one-sided exact tests at ``c4 = 1 - (1 - level) / 4`` (0.9875 at
``level = 0.95``). The four uses of the error budget ``alpha = 1 - level`` are each ``alpha / 4``:

1. the ascending run of ``pass`` over the nonzero cells, which gives ``lo``;
2. the top cell, ``pass(x_K)``, which gives ``RIGHT``;
3. the descending run of ``fail`` over the nonzero cells, which gives ``hi``;
4. the bottom cell, ``fail(x_1)``, which gives ``hi = x_1``, the upper half of ``LEFT`` and half
   of ``FAILS_AT_BASELINE``.

``pass(x)`` fires when the one-sided lower bound at ``c4`` is at least the target; ``fail(x)``
fires when the one-sided upper bound at ``c4`` is below it. The control's own ``pass(x_0)`` and
``fail(x_0)`` are also tests at ``c4``. Because the four parts add up to ``alpha``, the chance that
*any* exact result is returned and wrong is at most ``1 - level``, whatever kind the data selects
(the slow simulation in ``tests/coverage/test_exact_bracket_coverage.py`` checks this).

The classification order, pinned here:

1. ``CONTROL_INCOMPATIBLE`` (terminal).
2. A resolved target outside ``(0, 1)`` is ``UNREACHABLE`` on any fit status, before anything
   below. A target of 1 or more makes ``fail(x_0)`` trivially true whenever the control has a
   failure; a target of 0 or less makes every ``pass`` trivially true.
3. ``FAILS_AT_BASELINE``: a control exists, ``fail(x_0)`` and ``fail(x_1)`` at ``c4`` (the latter
   omitted only with no nonzero cell), and no nonzero cell's lower bound at ``level`` clears the
   target. That last guard stays at ``level``: it defers non-monotone data to the contradiction
   check and can only suppress a result.
4. Model-based ``UNREACHABLE`` (converged fits; tested in ``test_threshold_status_coverage.py``).
5. The monotonicity contradiction check, at ``level`` over all cells; it only suppresses.
6. ``RIGHT``: ``pass(x_K)``, giving ``lo = x_K``.
7. ``LEFT``: a control, ``pass(x_0)`` and ``fail(x_1)``, giving ``hi = x_1``.
8. A converged fit with no claim solves.
9. A failed fit's fallback: ``lo`` is the end of the ascending run (or ``None``); ``hi`` is
   ``x_1`` if ``fail(x_1)``, else the end of the descending run (or ``None``). Both set is a
   bracket with ``lo < hi``; exactly one set is a one-sided bound at ``c4`` with a warning; neither
   is no bounds. Nothing is rescanned at ``level`` and no bound is ever at the control.

A grid is written as a string of roles, one character per severity (``0, 1, 2, ...``; the first
is the control), and every role is verified against ``exact_one_sided_bound`` before a test uses
it. At ``target = 0.9``, ``level = 0.95``:

====  ===========  ==============================================================
role  counts       meaning
====  ===========  ==============================================================
P     100 / 100    passes at ``c4`` and at ``level``
p     49 / 50      passes at ``level`` only
G     90 / 100     neither passes nor fails at either confidence
f     84 / 100     fails at ``level`` only
F     0 / 100      fails at ``c4`` and at ``level``
====  ===========  ==============================================================

The boundary tests at the end do not use roles: for each of several levels they find the exact
count at which a test starts to fire at ``c4`` and check one count either side, so the confidence
is pinned to ``c4`` for that level (not ``level``, not a fixed 0.975).
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Iterator, Sequence

import pytest

from robustmargin import Axis, Fit, Status, censoring
from robustmargin.censoring import Classification, classify, exact_one_sided_bound
from robustmargin.testing import fake_fit
from robustmargin.types import Cell, Censoring

_TARGET = 0.9
_LEVEL = 0.95
_C4 = 1.0 - (1.0 - _LEVEL) / 4.0


@pytest.fixture(scope="module", autouse=True)
def _memoised_exact_bound() -> Iterator[None]:
    """``exact_one_sided_bound`` is a pure function of ``(successes, trials, level, side)`` and
    each test reuses a few count pairs, so memoising it changes no result, only the run time (it
    is a ``scipy.stats.binomtest`` call and ``classify`` makes a dozen per call)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            censoring,
            "exact_one_sided_bound",
            functools.lru_cache(maxsize=None)(censoring.exact_one_sided_bound),
        )
        yield


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

_ANY_STATUS = [Status.OK, Status.SEPARATION, Status.NOT_CONVERGED]
_FAILED_STATUSES = [Status.SEPARATION, Status.NOT_CONVERGED]


def _passes(cell: Cell, confidence: float) -> bool:
    return (
        exact_one_sided_bound(cell.successes, cell.trials, level=confidence, side="lower")
        >= _TARGET
    )


def _fails(cell: Cell, confidence: float) -> bool:
    return (
        exact_one_sided_bound(cell.successes, cell.trials, level=confidence, side="upper") < _TARGET
    )


def _grid(
    roles: str, *, control: bool = True, severities: Sequence[float] | None = None
) -> tuple[Cell, ...]:
    """Cells from a role string, severities ``0, 1, 2, ...`` (or ``1, 2, 3, ...`` with no control,
    or the given ones). Whitespace is ignored. Each role's meaning is checked against
    ``exact_one_sided_bound`` here, so a test never silently runs on a cell that is not what its
    name says."""
    letters = roles.replace(" ", "")
    if severities is None:
        first = 0 if control else 1
        severities = [float(first + i) for i in range(len(letters))]
    assert len(severities) == len(letters)
    cells: list[Cell] = []
    for severity, role in zip(severities, letters, strict=True):
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


# A converged fit with a failing control and `upper` fixed at 1 is only valid on a linear axis
# (on a log axis the likelihood is zero), and these tests need failing controls.
_LINEAR_AXIS = Axis(name="severity", unit="unit", scale="linear")


def _fit(cells: tuple[Cell, ...], status: Status) -> Fit:
    """A real ``Fit`` of ``status`` carrying the chosen cells. ``classify`` reads only ``status``
    and ``cells`` (and, for the model-based ``UNREACHABLE`` step, the asymptotes, which the
    filler fixes at 1 and 0)."""
    if status is Status.OK:
        return dataclasses.replace(
            fake_fit(diagnostics=None, monotonicity=None), axis=_LINEAR_AXIS, cells=cells
        )
    return dataclasses.replace(fake_fit(status=status, monotonicity=None), cells=cells)


def _classify(
    roles: str,
    status: Status,
    *,
    control: bool = True,
    severities: Sequence[float] | None = None,
    target: float = _TARGET,
    level: float = _LEVEL,
) -> Classification:
    cells = _grid(roles, control=control, severities=severities)
    return classify(_fit(cells, status), target=target, level=level)


@pytest.fixture(params=_ANY_STATUS, ids=lambda s: s.name)
def any_status(request: pytest.FixtureRequest) -> Status:
    status: Status = request.param
    return status


@pytest.fixture(params=_FAILED_STATUSES, ids=lambda s: s.name)
def failed_status(request: pytest.FixtureRequest) -> Status:
    status: Status = request.param
    return status


def _assert_no_claim(result: Classification) -> None:
    assert result.censoring is Censoring.NONE
    assert result.lo is None and result.hi is None


class TestControlIncompatibleIsTerminal:
    def test_a_control_incompatible_fit_is_mirrored_whatever_the_cells_say(self) -> None:
        cells = _grid("P P P")

        result = classify(_fit(cells, Status.CONTROL_INCOMPATIBLE), target=_TARGET, level=_LEVEL)

        assert result.status is Status.CONTROL_INCOMPATIBLE
        _assert_no_claim(result)
        assert result.baseline is None and result.grid is None and result.solve is False

    @pytest.mark.parametrize("target", [0.0, 1.0, 1.5, -0.1])
    def test_it_outranks_the_target_range_step(self, target: float) -> None:
        """Step 1 comes before step 2: an impossible target does not turn a control the fit
        itself rejected into ``UNREACHABLE``."""
        cells = _grid("P P P")

        result = classify(_fit(cells, Status.CONTROL_INCOMPATIBLE), target=target, level=_LEVEL)

        assert result.status is Status.CONTROL_INCOMPATIBLE


class TestATargetOutsideZeroOneIsUnreachableOnAnyFitStatus:
    """No binomial curve crosses a target at or below 0 or at or above 1, and exact tests against
    such a target are trivially true or false. It is decided before any exact test."""

    @pytest.mark.parametrize("target", [0.0, 1.0, 1.5, -0.1, 7.0])
    def test_the_result_is_unreachable_with_no_numbers_and_a_warning_naming_the_target(
        self, target: float, any_status: Status
    ) -> None:
        result = _classify("P P P", any_status, target=target)

        assert result.status is Status.UNREACHABLE
        _assert_no_claim(result)
        assert result.baseline is None
        assert result.grid is None, "Threshold allows a grid only on SEPARATION / NOT_CONVERGED"
        assert result.solve is False
        assert any(
            f"the resolved target {target!r} is outside the open interval (0, 1)" in w
            for w in result.warnings
        ), f"no warning names the target {target!r}: {result.warnings!r}"

    def test_it_is_decided_before_fails_at_baseline(self, any_status: Status) -> None:
        """A target of 1.0 or more makes ``fail(x_0)`` true whenever the control has a failure,
        so a rule that tested the control first would report ``FAILS_AT_BASELINE``: the control and
        every nonzero cell here fail confidently against 1.0 and 1.5."""
        for target in (1.0, 1.5):
            result = _classify("F F F", any_status, target=target)

            assert result.status is Status.UNREACHABLE, target
            assert result.baseline is None

    def test_it_is_decided_before_right(self, any_status: Status) -> None:
        """Against a target of 0 every ``pass`` is true, including the top cell's, so a rule that
        tested the top cell first would report ``RIGHT`` -- here on cells that all failed."""
        result = _classify("F F F", any_status, target=0.0)

        assert result.status is Status.UNREACHABLE
        assert result.censoring is Censoring.NONE

    def test_the_range_is_the_open_interval_so_targets_just_inside_it_are_classified(
        self, any_status: Status
    ) -> None:
        near_one = _classify("F F F", any_status, target=1.0 - 1e-9)
        near_zero = _classify("P P P", any_status, target=1e-9)

        assert near_one.status is Status.FAILS_AT_BASELINE
        assert near_zero.censoring is Censoring.RIGHT
        assert near_zero.status is any_status


class TestFailsAtBaseline:
    """``fail(x_0)`` and ``fail(x_1)`` at ``c4``, with the unchanged guard at ``level``."""

    def test_the_control_and_the_lowest_nonzero_level_both_failing_gives_the_result(
        self, any_status: Status
    ) -> None:
        cells = _grid("F F F F")

        result = classify(_fit(cells, any_status), target=_TARGET, level=_LEVEL)

        assert result.status is Status.FAILS_AT_BASELINE
        assert result.censoring is Censoring.NONE
        assert result.lo is None and result.hi is None
        assert result.grid is None and result.solve is False
        # The control's own two-sided interval at `level` is what is reported with it; it is not
        # the one-sided pair that decided the classification.
        assert result.baseline is not None
        assert (result.baseline.successes, result.baseline.trials) == (0, 100)
        assert result.baseline.severity == 0.0
        assert result.baseline.level == _LEVEL
        assert result.baseline.method == "per_cell_clopper_pearson"

    def test_only_the_control_and_the_lowest_nonzero_level_are_tested(
        self, any_status: Status
    ) -> None:
        """Levels above ``x_1`` may be anything that is not a confident pass at ``level``."""
        for roles in ("F F G G", "F F f G", "F F G F"):
            assert _classify(roles, any_status).status is Status.FAILS_AT_BASELINE, roles

    def test_a_control_that_fails_only_at_level_does_not_fire(self, failed_status: Status) -> None:
        """``f F F``: the control's upper bound is below the target at ``level`` but not at
        ``c4``. A rule that tested the control at ``level`` reports ``FAILS_AT_BASELINE``; here it
        is a fallback whose ``hi`` is ``x_1``. The control is never a bound."""
        assert _fails(_grid("f F F")[0], _LEVEL) and not _fails(_grid("f F F")[0], _C4)

        result = _classify("f F F", failed_status)

        assert result.status is failed_status
        assert result.baseline is None
        assert result.censoring is Censoring.NONE
        assert result.hi == 1.0
        assert result.lo is None

    @pytest.mark.parametrize("roles", ["F f F", "F G F", "F p F"])
    def test_a_lowest_nonzero_level_that_does_not_fail_at_c4_does_not_fire(
        self, roles: str, failed_status: Status
    ) -> None:
        """The second half of the test is ``fail(x_1)`` at ``c4``. ``f`` fails at ``level`` only,
        ``G`` and ``p`` do not fail."""
        result = _classify(roles, failed_status)

        assert result.status is failed_status
        assert result.baseline is None

    def test_a_lowest_nonzero_level_that_fails_only_at_level_ends_in_the_run_from_the_top(
        self, failed_status: Status
    ) -> None:
        """``F f F``: nothing passes, ``x_1`` does not fail at ``c4`` and the control is not a
        bound, so the result is the run from the top, ``hi = 2``, one-sided."""
        result = _classify("F f F", failed_status)

        assert result.hi == 2.0
        assert result.lo is None
        assert any("determined only the upper side" in w for w in result.warnings)

    @pytest.mark.parametrize("roles", ["F F p", "F F P", "F F F p"])
    def test_a_nonzero_level_that_passes_at_level_stops_it_and_the_contradiction_is_named(
        self, roles: str, failed_status: Status
    ) -> None:
        """The guard is at ``level``, not ``c4``: ``p`` passes at ``level`` only, and still
        defers the data to the contradiction check (a nonzero level passing above a control that
        fails). Nothing is claimed."""
        result = _classify(roles, failed_status)

        assert result.status is failed_status
        assert result.baseline is None
        _assert_no_claim(result)
        assert [w for w in result.warnings if w.startswith("NON_MONOTONE: ")], result.warnings

    def test_a_control_alone_needs_only_fail_x0(self, failed_status: Status) -> None:
        """With no nonzero cell the ``fail(x_1)`` half is omitted."""
        fires = _classify("F", failed_status)
        undetermined = _classify("G", failed_status)
        fails_at_level_only = _classify("f", failed_status)

        assert fires.status is Status.FAILS_AT_BASELINE
        assert fires.baseline is not None
        assert undetermined.status is failed_status
        assert fails_at_level_only.status is failed_status, "fail(x_0) is a test at c4, not level"

    def test_with_no_control_it_cannot_fire(self, failed_status: Status) -> None:
        """Nothing is below ``x_1`` to fail. The lowest level failing still gives ``hi = x_1``."""
        result = _classify("F F F", failed_status, control=False)

        assert result.status is failed_status
        assert result.baseline is None
        assert result.hi == 1.0
        assert result.lo is None


class TestRight:
    """``pass(x_K)`` at ``c4`` gives ``RIGHT`` with ``lo = x_K``: one pre-chosen cell, not a run."""

    def test_the_top_cell_passing_gives_right_at_that_severity(self, any_status: Status) -> None:
        result = _classify("P P G P", any_status, severities=[0.0, 0.5, 2.0, 7.5])

        assert result.censoring is Censoring.RIGHT
        assert result.status is any_status
        assert result.lo == 7.5
        assert result.hi is None
        assert result.baseline is None and result.solve is False
        assert (result.grid is not None) == (any_status is not Status.OK), (
            "the grid accompanies a failed fit's exact bound, and Threshold forbids it on OK"
        )

    def test_a_top_cell_that_passes_only_at_level_does_not_fire(
        self, failed_status: Status
    ) -> None:
        """``P P p``: a rule that tested the top cell at ``level`` reports ``RIGHT`` at 2. At
        ``c4`` the result is the one-sided lower bound ``lo = 1`` (the run stops at ``p``)."""
        assert _passes(_grid("P P p")[2], _LEVEL) and not _passes(_grid("P P p")[2], _C4)

        result = _classify("P P p", failed_status)

        assert result.censoring is Censoring.NONE
        assert result.status is failed_status
        assert result.lo == 1.0
        assert result.hi is None

    def test_a_converged_fit_whose_top_cell_passes_only_at_level_goes_to_the_solve_path(
        self,
    ) -> None:
        result = _classify("P P p", Status.OK)

        assert result.solve is True
        assert result.status is Status.OK
        assert result.censoring is Censoring.NONE
        assert result.lo is None and result.hi is None

    def test_with_no_control_the_top_cell_alone_decides(self, any_status: Status) -> None:
        result = _classify("P P", any_status, control=False)

        assert result.censoring is Censoring.RIGHT
        assert result.lo == 2.0

    def test_a_passing_control_alone_is_not_a_right_bound_at_the_control(
        self, failed_status: Status
    ) -> None:
        """With no nonzero cell there is no ``x_K``; a bound at severity 0 would claim nothing a
        threshold on a non-negative axis does not already satisfy."""
        result = _classify("P", failed_status)

        assert result.censoring is not Censoring.RIGHT
        assert result.lo is None


class TestLeft:
    """``LEFT`` needs a control that passes at ``c4`` and a lowest nonzero cell that fails at
    ``c4``, and gives ``hi = x_1``."""

    def test_a_passing_control_and_a_failing_lowest_nonzero_level_give_left(
        self, any_status: Status
    ) -> None:
        result = _classify("P F G G", any_status, severities=[0.0, 0.25, 1.0, 3.0])

        assert result.censoring is Censoring.LEFT
        assert result.status is any_status
        assert result.hi == 0.25
        assert result.lo is None
        assert result.baseline is None and result.solve is False
        assert (result.grid is not None) == (any_status is not Status.OK)

    def test_a_control_that_passes_only_at_level_does_not_fire(self, failed_status: Status) -> None:
        """``p F F``: a rule that gated on the control at ``level`` reports ``LEFT``. At ``c4`` the
        control is not confidently passing, so there is no claim about the control and the result
        is the fallback: ``hi = x_1`` from the bottom-cell test, one-sided, with no bound at the
        control."""
        assert _passes(_grid("p F F")[0], _LEVEL) and not _passes(_grid("p F F")[0], _C4)

        result = _classify("p F F", failed_status)

        assert result.censoring is Censoring.NONE
        assert result.status is failed_status
        assert result.hi == 1.0
        assert result.lo is None

    def test_a_lowest_nonzero_level_that_fails_only_at_level_does_not_fire(
        self, failed_status: Status
    ) -> None:
        """``P f F``: a rule that tested ``x_1`` at ``level`` reports ``LEFT`` at 1. At ``c4``
        ``x_1`` has not failed; the run from the top gives ``hi = 2``."""
        assert _fails(_grid("P f F")[1], _LEVEL) and not _fails(_grid("P f F")[1], _C4)

        result = _classify("P f F", failed_status)

        assert result.censoring is Censoring.NONE
        assert result.status is failed_status
        assert result.hi == 2.0
        assert result.lo is None

    def test_with_no_control_it_cannot_fire(self, failed_status: Status) -> None:
        result = _classify("F F F", failed_status, control=False)

        assert result.censoring is Censoring.NONE
        assert result.hi == 1.0

    def test_a_failing_control_is_fails_at_baseline_not_left(self, any_status: Status) -> None:
        """The control must pass for ``LEFT`` and fail for ``FAILS_AT_BASELINE``: with the lowest
        level failing in both, the control decides which."""
        passing_control = _classify("P F F", any_status)
        failing_control = _classify("F F F", any_status)

        assert passing_control.censoring is Censoring.LEFT
        assert failing_control.status is Status.FAILS_AT_BASELINE
        assert failing_control.censoring is Censoring.NONE

    def test_an_undetermined_control_is_neither(self, failed_status: Status) -> None:
        result = _classify("G F F", failed_status)

        assert result.censoring is Censoring.NONE
        assert result.status is failed_status
        assert result.hi == 1.0

    def test_a_passing_control_alone_cannot_give_left(self, failed_status: Status) -> None:
        """No nonzero cell, so there is no ``x_1`` to fail."""
        result = _classify("P", failed_status)

        assert result.censoring is not Censoring.LEFT


class TestRightAndLeftCannotBothFire:
    """``RIGHT`` needs the top cell to pass and ``LEFT`` the lowest to fail, both at ``c4``. With
    two or more nonzero cells that is a passing cell above a failing one, which the contradiction
    check (at the looser ``level``) always catches first; with one nonzero cell it is one cell
    passing and failing at once."""

    @pytest.mark.parametrize("roles", ["P F P", "P F G P", "P F F P"])
    def test_a_pass_above_a_fail_is_the_contradiction_not_a_claim(
        self, roles: str, any_status: Status
    ) -> None:
        result = _classify(roles, any_status)

        assert result.censoring is Censoring.NONE
        assert result.lo is None and result.hi is None
        assert [w for w in result.warnings if w.startswith("NON_MONOTONE: ")]


class TestTheFailedFitFallback:
    """Step 9. ``lo`` is the end of the ascending run, ``hi`` is ``x_1`` if ``fail(x_1)`` and
    otherwise the end of the descending run, both over nonzero cells, both at ``c4``."""

    def test_both_sides_set_is_a_bracket_with_lo_below_hi_and_no_warning(
        self, failed_status: Status
    ) -> None:
        result = _classify("P P P G F F", failed_status)

        assert result.censoring is Censoring.NONE
        assert result.lo is not None and result.hi is not None
        assert (result.lo, result.hi) == (2.0, 4.0)
        assert result.lo < result.hi
        assert result.warnings == ()
        assert result.grid is not None

    def test_exactly_one_side_set_is_a_one_sided_bound_at_c4_with_a_warning_naming_it(
        self, failed_status: Status
    ) -> None:
        lower = _classify("P P P P G", failed_status)
        upper = _classify("P G F F F", failed_status)

        assert (lower.lo, lower.hi) == (3.0, None)
        assert (upper.lo, upper.hi) == (None, 2.0)
        for result, side in ((lower, "lower"), (upper, "upper")):
            assert len(result.warnings) == 1
            (warning,) = result.warnings
            assert f"determined only the {side} side" in warning
            assert "confidence 0.9875" in warning
            assert "rescanned" not in warning

    def test_neither_side_set_is_no_bounds_with_a_warning(self, failed_status: Status) -> None:
        result = _classify("P G G G", failed_status)

        _assert_no_claim(result)
        assert result.status is failed_status
        assert len(result.warnings) == 1
        assert "no bounds are reported" in result.warnings[0]
        assert result.grid is not None

    def test_no_bound_is_ever_at_the_control(self, failed_status: Status) -> None:
        """Even where the control is the only confident cell on one side, ``lo`` and ``hi`` are
        severities of nonzero cells."""
        for roles in ("P G G G", "G F F F", "P P G F", "G G F F"):
            result = _classify(roles, failed_status)
            for bound in (result.lo, result.hi):
                assert bound is None or bound > 0.0, (roles, result.lo, result.hi)

    def test_bounds_are_the_severities_not_the_positions(self, failed_status: Status) -> None:
        result = _classify("P P P G F F", failed_status, severities=[0.0, 0.1, 0.4, 0.9, 2.5, 6.0])

        assert (result.lo, result.hi) == (0.4, 2.5)

    def test_a_single_nonzero_cell_can_give_either_side(self, failed_status: Status) -> None:
        passing = _classify("G P", failed_status)
        failing = _classify("G F", failed_status)

        assert passing.censoring is Censoring.RIGHT
        assert failing.censoring is Censoring.NONE and failing.hi == 1.0

    def test_with_no_control_the_runs_are_over_all_cells(self, failed_status: Status) -> None:
        result = _classify("P P G F F", failed_status, control=False)

        assert (result.lo, result.hi) == (2.0, 4.0)


_BOUNDARY_TRIALS = 200


def _c4_of(level: float) -> float:
    return 1.0 - (1.0 - level) / 4.0


@functools.cache
def _pass_boundary(confidence: float) -> int:
    """The smallest success count out of ``_BOUNDARY_TRIALS`` whose one-sided lower bound at
    ``confidence`` reaches the target. (The bound is increasing in the count, so this is a
    bisection.)"""
    low, high = 0, _BOUNDARY_TRIALS
    assert exact_one_sided_bound(high, _BOUNDARY_TRIALS, level=confidence, side="lower") >= _TARGET
    while low < high:
        middle = (low + high) // 2
        lower = exact_one_sided_bound(middle, _BOUNDARY_TRIALS, level=confidence, side="lower")
        if lower >= _TARGET:
            high = middle
        else:
            low = middle + 1
    return low


@functools.cache
def _fail_boundary(confidence: float) -> int:
    """The largest success count out of ``_BOUNDARY_TRIALS`` whose one-sided upper bound at
    ``confidence`` is below the target (a bisection, for the same reason)."""
    low, high = 0, _BOUNDARY_TRIALS
    assert exact_one_sided_bound(low, _BOUNDARY_TRIALS, level=confidence, side="upper") < _TARGET
    while low < high:
        middle = (low + high + 1) // 2
        upper = exact_one_sided_bound(middle, _BOUNDARY_TRIALS, level=confidence, side="upper")
        if upper < _TARGET:
            low = middle
        else:
            high = middle - 1
    return low


class TestTheConfidenceIsC4AtEveryLevel:
    """The confidence of every exact test is ``1 - (1 - level) / 4``, whatever ``level`` is. For
    each level below, find the smallest count whose lower bound at ``c4`` clears the target (or
    the largest whose upper bound is below it) and check that count and its neighbour. The
    neighbour is a count that a test at ``level`` itself would already have fired on (asserted in
    setup), so a rule at ``level`` -- or at any confidence below ``c4`` -- fails these."""

    _LEVELS = [0.8, 0.9, 0.95, 0.99]

    @staticmethod
    def _boundary_pass(level: float) -> tuple[int, int]:
        """``(first count that passes at c4, the count below it)``; the latter passes at
        ``level`` but not at ``c4``."""
        at_c4 = _pass_boundary(_c4_of(level))
        assert _pass_boundary(level) < at_c4, "setup: c4 must be stricter than level here"
        return at_c4, at_c4 - 1

    @staticmethod
    def _boundary_fail(level: float) -> tuple[int, int]:
        """``(last count that fails at c4, the count above it)``; the latter fails at ``level``
        but not at ``c4``."""
        at_c4 = _fail_boundary(_c4_of(level))
        assert _fail_boundary(level) > at_c4, "setup: c4 must be stricter than level here"
        return at_c4, at_c4 + 1

    @staticmethod
    def _cells(*counts: int) -> tuple[Cell, ...]:
        return tuple(
            Cell(severity=float(i), successes=k, trials=_BOUNDARY_TRIALS)
            for i, k in enumerate(counts)
        )

    def _classify(self, counts: Sequence[int], status: Status, level: float) -> Classification:
        return classify(_fit(self._cells(*counts), status), target=_TARGET, level=level)

    # An all-success cell passes and an all-failure cell fails at every level used here; 180 of
    # 200 is exactly the target and neither passes nor fails.
    _ALL = _BOUNDARY_TRIALS
    _NONE = 0
    _MID = 180

    @pytest.mark.parametrize("level", _LEVELS)
    def test_right_fires_from_the_first_count_that_passes_at_c4(
        self, level: float, any_status: Status
    ) -> None:
        fires, just_short = self._boundary_pass(level)

        on = self._classify([self._ALL, fires], any_status, level)
        off = self._classify([self._ALL, just_short], any_status, level)

        assert on.censoring is Censoring.RIGHT and on.lo == 1.0
        assert off.censoring is not Censoring.RIGHT

    @pytest.mark.parametrize("level", _LEVELS)
    def test_left_fires_up_to_the_last_count_that_fails_at_c4(
        self, level: float, any_status: Status
    ) -> None:
        fires, just_over = self._boundary_fail(level)

        on = self._classify([self._ALL, fires], any_status, level)
        off = self._classify([self._ALL, just_over], any_status, level)

        assert on.censoring is Censoring.LEFT and on.hi == 1.0
        assert off.censoring is not Censoring.LEFT

    @pytest.mark.parametrize("level", _LEVELS)
    def test_lefts_control_gate_is_the_first_count_that_passes_at_c4(
        self, level: float, any_status: Status
    ) -> None:
        fires, just_short = self._boundary_pass(level)

        on = self._classify([fires, self._NONE], any_status, level)
        off = self._classify([just_short, self._NONE], any_status, level)

        assert on.censoring is Censoring.LEFT
        assert off.censoring is not Censoring.LEFT

    @pytest.mark.parametrize("level", _LEVELS)
    def test_fails_at_baseline_needs_the_control_to_fail_at_c4(
        self, level: float, any_status: Status
    ) -> None:
        fires, just_over = self._boundary_fail(level)

        on = self._classify([fires, self._NONE], any_status, level)
        off = self._classify([just_over, self._NONE], any_status, level)

        assert on.status is Status.FAILS_AT_BASELINE
        assert off.status is not Status.FAILS_AT_BASELINE

    @pytest.mark.parametrize("level", _LEVELS)
    def test_fails_at_baseline_needs_the_lowest_nonzero_level_to_fail_at_c4(
        self, level: float, any_status: Status
    ) -> None:
        fires, just_over = self._boundary_fail(level)

        on = self._classify([self._NONE, fires], any_status, level)
        off = self._classify([self._NONE, just_over], any_status, level)

        assert on.status is Status.FAILS_AT_BASELINE
        assert off.status is not Status.FAILS_AT_BASELINE

    @pytest.mark.parametrize("level", _LEVELS)
    def test_the_fails_at_baseline_guard_is_at_level_not_c4(
        self, level: float, any_status: Status
    ) -> None:
        """A nonzero cell above the failing ones blocks the result exactly when its lower bound
        at ``level`` clears the target -- the first count that does so at ``level`` is one that
        does not pass at ``c4``, which is what separates the two confidences."""
        blocks_at_level = _pass_boundary(level)
        assert blocks_at_level < _pass_boundary(_c4_of(level)), "setup"

        blocked = self._classify([self._NONE, self._NONE, blocks_at_level], any_status, level)
        allowed = self._classify([self._NONE, self._NONE, blocks_at_level - 1], any_status, level)

        assert blocked.status is not Status.FAILS_AT_BASELINE
        assert allowed.status is Status.FAILS_AT_BASELINE

    @pytest.mark.parametrize("level", _LEVELS)
    def test_the_ascending_run_continues_through_the_first_count_that_passes_at_c4(
        self, level: float, failed_status: Status
    ) -> None:
        """``x_3`` is left undetermined so ``RIGHT`` does not take the result; the run is
        ``x_1`` and, when ``x_2`` passes at ``c4``, ``x_2``."""
        fires, just_short = self._boundary_pass(level)

        on = self._classify([self._ALL, self._ALL, fires, self._MID], failed_status, level)
        off = self._classify([self._ALL, self._ALL, just_short, self._MID], failed_status, level)

        assert on.lo == 2.0
        assert off.lo == 1.0

    @pytest.mark.parametrize("level", _LEVELS)
    def test_the_descending_run_continues_through_the_last_count_that_fails_at_c4(
        self, level: float, failed_status: Status
    ) -> None:
        """An undetermined control and ``x_1`` keep ``LEFT`` and the bottom-cell test out of the
        way; the run from the top is ``x_3`` and, when ``x_2`` fails at ``c4``, ``x_2``."""
        fires, just_over = self._boundary_fail(level)

        on = self._classify([self._MID, self._MID, fires, self._NONE], failed_status, level)
        off = self._classify([self._MID, self._MID, just_over, self._NONE], failed_status, level)

        assert on.hi == 2.0
        assert off.hi == 3.0

    @pytest.mark.parametrize("level", _LEVELS)
    def test_the_contradiction_check_stays_at_level(
        self, level: float, failed_status: Status
    ) -> None:
        """A passing nonzero level above a failing one is a contradiction at ``level`` even when
        neither would fire at ``c4``: the check can suppress a claim, never add one."""
        passes_at_level_only = _pass_boundary(level)
        assert passes_at_level_only < _pass_boundary(_c4_of(level)), "setup"
        fails_at_level_only = _fail_boundary(level)
        assert fails_at_level_only > _fail_boundary(_c4_of(level)), "setup"

        result = self._classify(
            [self._MID, fails_at_level_only, passes_at_level_only], failed_status, level
        )

        assert result.lo is None and result.hi is None
        assert [w for w in result.warnings if w.startswith("NON_MONOTONE: ")]
