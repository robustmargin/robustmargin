"""Censoring/status propagation through ``ratio_interval()``.

**The rule.** A censored or failed input threshold propagates a flag, never a number
(``estimate``/``lo``/``hi``/``shape`` all ``None``, matching :class:`~robustmargin.Ratio`'s own
``__post_init__`` invariant), **and** which specific :class:`~robustmargin.Censoring` the
``Ratio`` carries is direction-aware, per this table:

| ``threshold_a`` | ``threshold_b`` | ``Ratio.censoring`` |
|---|---|---|
| ``RIGHT`` | ``NONE`` | ``RIGHT`` |
| ``LEFT`` | ``NONE`` | ``LEFT`` |
| ``NONE`` | ``RIGHT`` | ``LEFT`` (a larger denominator pushes the ratio **down**) |
| ``NONE`` | ``LEFT`` | ``RIGHT`` (a smaller denominator pushes the ratio **up**) |
| ``OPEN_UPPER`` | ``NONE`` | ``OPEN_UPPER`` |
| ``NONE`` | ``OPEN_UPPER`` | ``OPEN_LOWER`` (flipped) |
| ``OPEN_LOWER`` | ``NONE`` | ``OPEN_LOWER`` |
| ``NONE`` | ``OPEN_LOWER`` | ``OPEN_UPPER`` (flipped) |
| censored | censored | ``threshold_a``'s own flag, plus a ``warnings`` entry (no direction) |

``TestCensoringDirectionTable`` below is the specific version of this, using two ``Status.OK``
thresholds each time so the direction-flip logic is tested in isolation from any fit-level
failure. A ``(Status.OK, Censoring.RIGHT)`` threshold is a *legal, non-failed* result (RIGHT/LEFT
take the fit's status, and ``(OK, RIGHT)``/``(OK, LEFT)`` are both reachable), so the ``Ratio``
built from one carries ``Status.OK`` together with its (possibly flipped) censoring flag -- never
a failure status.

**The status rule is pinned in ``TestStatusPropagationRule``**:
``Ratio.status`` is ``Status.OK`` only when both input thresholds are ``Status.OK``; when exactly
one has failed (``SEPARATION``, ``NOT_CONVERGED``, ``UNREACHABLE``, ``FAILS_AT_BASELINE`` or
``CONTROL_INCOMPATIBLE``), that failure wins, on either side; when both have failed with
*different* statuses, the ``Ratio`` takes ``threshold_a``'s and carries a ``warnings`` entry (the
same "no coherent single reason" rationale as the both-censored row of the table above -- two
failures resolving to one status is itself a judgement call, not a derivation). This is why
``TestNumeratorProblemPropagatesWithACleanDenominator`` and its mirror assert
``result.status is status`` directly (one clean side means "one failure wins" determines the
result exactly), and why ``TestCensoringDirectionAppliesRegardlessOfStatus`` asserts
``.status`` too, not only ``.censoring``.

Building real censored/failed fits for every combination would be disproportionate to what this
file checks, so every threshold here comes from ``robustmargin.testing.fake_threshold``, which
already produces schema-valid objects for each allowed ``(status, censoring)`` combination, on
a shared axis/definition (so the axis/definition guards, tested separately in
``tests/unit/test_ratio_interval_guards.py``, never fire here by accident).
"""

from __future__ import annotations

import pytest

from robustmargin import Censoring, Ratio, Status, ratio_interval
from robustmargin.testing import fake_threshold

# The fourteen (status, censoring) combinations allowed for a Threshold,
# mirroring robustmargin.testing._ALLOWED_THRESHOLD_COMBINATIONS. (Status.OK, Censoring.NONE) is
# the one "clean" combination and is exercised as the control case below, not as a "problem"
# case in this list.
_PROBLEM_COMBINATIONS: tuple[tuple[Status, Censoring], ...] = (
    (Status.OK, Censoring.RIGHT),
    (Status.OK, Censoring.LEFT),
    (Status.OK, Censoring.OPEN_UPPER),
    (Status.OK, Censoring.OPEN_LOWER),
    (Status.SEPARATION, Censoring.NONE),
    (Status.SEPARATION, Censoring.RIGHT),
    (Status.SEPARATION, Censoring.LEFT),
    (Status.NOT_CONVERGED, Censoring.NONE),
    (Status.NOT_CONVERGED, Censoring.RIGHT),
    (Status.NOT_CONVERGED, Censoring.LEFT),
    (Status.UNREACHABLE, Censoring.NONE),
    (Status.FAILS_AT_BASELINE, Censoring.NONE),
    (Status.CONTROL_INCOMPATIBLE, Censoring.NONE),
)

assert len(_PROBLEM_COMBINATIONS) == 13  # the fourteenth is (OK, NONE), the control case


def _assert_flag_only_no_numbers(result: Ratio) -> None:
    assert not (result.status is Status.OK and result.censoring is Censoring.NONE)
    assert result.shape is None
    assert result.estimate is None
    assert result.lo is None
    assert result.hi is None


class TestNumeratorProblemPropagatesWithACleanDenominator:
    """``a`` is every non-``(OK, NONE)`` combination in turn; ``b`` is the clean control.

    ``result.status is status`` holds uniformly across all thirteen combinations: for the four
    ``(Status.OK, censoring)`` rows, both inputs are ``Status.OK`` (only the censoring differs),
    so the status rule's "``Status.OK`` only when both inputs are ``OK``" gives ``Status.OK ==
    status`` trivially; for the other nine, ``b`` is clean, so "one failure wins" gives
    ``status`` directly.
    """

    @pytest.mark.parametrize(("status", "censoring"), _PROBLEM_COMBINATIONS)
    def test_no_numbers_when_a_is_not_ok_and_none(
        self, status: Status, censoring: Censoring
    ) -> None:
        t_a = fake_threshold(status=status, censoring=censoring)
        t_b = fake_threshold(status=Status.OK, censoring=Censoring.NONE)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        _assert_flag_only_no_numbers(result)
        assert result.status is status


class TestDenominatorProblemPropagatesWithACleanNumerator:
    """The mirror of the above: ``b`` carries the problem, ``a`` is clean. Same status-rule
    reasoning as above applies (status propagation is symmetric: "one failure wins" does not
    care which side the failure is on)."""

    @pytest.mark.parametrize(("status", "censoring"), _PROBLEM_COMBINATIONS)
    def test_no_numbers_when_b_is_not_ok_and_none(
        self, status: Status, censoring: Censoring
    ) -> None:
        t_a = fake_threshold(status=Status.OK, censoring=Censoring.NONE)
        t_b = fake_threshold(status=status, censoring=censoring)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        _assert_flag_only_no_numbers(result)
        assert result.status is status


class TestBothSidesProblemPropagates:
    """When *both* inputs are censored or failed, in different ways, numbers are still absent --
    now checked against the exact settled rule (module docstring) for each row, not just the
    generic "no numbers" half:

    - ``(OK, RIGHT) / (OK, LEFT)``: both ``Status.OK`` -> ``Ratio.status is OK``; censoring is
      the "both censored" row of the table (``threshold_a``'s flag, ``RIGHT``),
      identical to ``TestCensoringDirectionTable.test_both_censored_*``'s first case.
    - ``(SEPARATION, NONE) / (OK, OPEN_UPPER)``: one failure (``a``) wins -> ``SEPARATION``;
      censoring has one ``NONE`` side and one ``OPEN_UPPER`` side, so the direction table's
      ``NONE``/``OPEN_UPPER`` row applies (flipped) -> ``OPEN_LOWER``.
    - ``(UNREACHABLE, NONE) / (FAILS_AT_BASELINE, NONE)``: two *differing* failures ->
      ``threshold_a``'s status (``UNREACHABLE``) plus a warning; neither side carries a
      directional censoring flag, so ``Ratio.censoring`` stays ``NONE``.
    - ``(NOT_CONVERGED, RIGHT) / (NOT_CONVERGED, LEFT)``: the *same* failure status on both sides
      -> ``NOT_CONVERGED`` with no special warning required; censoring is again the "both
      censored" row -> ``threshold_a``'s flag, ``RIGHT``.
    """

    @pytest.mark.parametrize(
        (
            "status_a",
            "censoring_a",
            "status_b",
            "censoring_b",
            "expected_status",
            "expected_censoring",
        ),
        [
            (Status.OK, Censoring.RIGHT, Status.OK, Censoring.LEFT, Status.OK, Censoring.RIGHT),
            (
                Status.SEPARATION,
                Censoring.NONE,
                Status.OK,
                Censoring.OPEN_UPPER,
                Status.SEPARATION,
                Censoring.OPEN_LOWER,
            ),
            (
                Status.UNREACHABLE,
                Censoring.NONE,
                Status.FAILS_AT_BASELINE,
                Censoring.NONE,
                Status.UNREACHABLE,
                Censoring.NONE,
            ),
            (
                Status.NOT_CONVERGED,
                Censoring.RIGHT,
                Status.NOT_CONVERGED,
                Censoring.LEFT,
                Status.NOT_CONVERGED,
                Censoring.RIGHT,
            ),
        ],
    )
    def test_no_numbers_when_both_sides_have_a_problem(
        self,
        status_a: Status,
        censoring_a: Censoring,
        status_b: Status,
        censoring_b: Censoring,
        expected_status: Status,
        expected_censoring: Censoring,
    ) -> None:
        t_a = fake_threshold(status=status_a, censoring=censoring_a)
        t_b = fake_threshold(status=status_b, censoring=censoring_b)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        _assert_flag_only_no_numbers(result)
        assert result.status is expected_status
        assert result.censoring is expected_censoring


class TestBothSidesCleanIsTheControlCase:
    """The contrast case: when both inputs are ``(OK, NONE)``, numbers ARE present. This is
    exercised in depth by ``tests/reference/`` and ``tests/property/``; it is included here only
    to show the "problem" cases above are actually being exercised, not vacuously passing
    because *no* combination ever produces numbers.
    """

    def test_numbers_present_when_both_sides_are_ok_and_none(self) -> None:
        t_a = fake_threshold(status=Status.OK, censoring=Censoring.NONE)
        t_b = fake_threshold(status=Status.OK, censoring=Censoring.NONE)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.status is Status.OK
        assert result.censoring is Censoring.NONE
        assert result.shape is not None
        assert result.estimate is not None


# ------------------------------------------------------------------------------------------
# The direction-aware censoring table, and the status/censoring decoupling it implies for a
# threshold that is censored but not otherwise failed.
# ------------------------------------------------------------------------------------------


class TestCensoringDirectionTable:
    """The table, cell by cell. Every threshold here is ``Status.OK`` (a converged fit whose
    one-sided bound or open profile side is the *only* thing wrong with it, e.g. the
    ``system_a_axis_2`` worked case), so ``Ratio.status`` is asserted to be
    ``Status.OK`` throughout -- a censored-but-converged input is not a failed one.
    """

    def test_right_numerator_with_clean_denominator_stays_right(self) -> None:
        t_a = fake_threshold(status=Status.OK, censoring=Censoring.RIGHT)
        t_b = fake_threshold(status=Status.OK, censoring=Censoring.NONE)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.status is Status.OK
        assert result.censoring is Censoring.RIGHT
        assert result.estimate is None and result.lo is None and result.hi is None

    def test_left_numerator_with_clean_denominator_stays_left(self) -> None:
        t_a = fake_threshold(status=Status.OK, censoring=Censoring.LEFT)
        t_b = fake_threshold(status=Status.OK, censoring=Censoring.NONE)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.status is Status.OK
        assert result.censoring is Censoring.LEFT

    def test_right_denominator_with_clean_numerator_flips_to_left(self) -> None:
        """A larger denominator (``sigma*_b`` held only to exceed ``x_max_b``) pushes the ratio
        **down** -- ``RIGHT`` on ``b`` becomes ``LEFT`` on the ratio, the whole point of doing
        this by direction rather than by copying."""
        t_a = fake_threshold(status=Status.OK, censoring=Censoring.NONE)
        t_b = fake_threshold(status=Status.OK, censoring=Censoring.RIGHT)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.status is Status.OK
        assert result.censoring is Censoring.LEFT

    def test_left_denominator_with_clean_numerator_flips_to_right(self) -> None:
        t_a = fake_threshold(status=Status.OK, censoring=Censoring.NONE)
        t_b = fake_threshold(status=Status.OK, censoring=Censoring.LEFT)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.status is Status.OK
        assert result.censoring is Censoring.RIGHT

    def test_open_upper_numerator_with_clean_denominator_stays_open_upper(self) -> None:
        t_a = fake_threshold(status=Status.OK, censoring=Censoring.OPEN_UPPER)
        t_b = fake_threshold(status=Status.OK, censoring=Censoring.NONE)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.status is Status.OK
        assert result.censoring is Censoring.OPEN_UPPER

    def test_open_upper_denominator_with_clean_numerator_flips_to_open_lower(self) -> None:
        t_a = fake_threshold(status=Status.OK, censoring=Censoring.NONE)
        t_b = fake_threshold(status=Status.OK, censoring=Censoring.OPEN_UPPER)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.status is Status.OK
        assert result.censoring is Censoring.OPEN_LOWER

    def test_open_lower_numerator_with_clean_denominator_stays_open_lower(self) -> None:
        t_a = fake_threshold(status=Status.OK, censoring=Censoring.OPEN_LOWER)
        t_b = fake_threshold(status=Status.OK, censoring=Censoring.NONE)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.status is Status.OK
        assert result.censoring is Censoring.OPEN_LOWER

    def test_open_lower_denominator_with_clean_numerator_flips_to_open_upper(self) -> None:
        t_a = fake_threshold(status=Status.OK, censoring=Censoring.NONE)
        t_b = fake_threshold(status=Status.OK, censoring=Censoring.OPEN_LOWER)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.status is Status.OK
        assert result.censoring is Censoring.OPEN_UPPER

    @pytest.mark.parametrize(
        ("censoring_a", "censoring_b"),
        [
            (Censoring.RIGHT, Censoring.LEFT),
            (Censoring.LEFT, Censoring.RIGHT),
            (Censoring.OPEN_UPPER, Censoring.OPEN_LOWER),
            (Censoring.OPEN_LOWER, Censoring.RIGHT),
        ],
    )
    def test_both_censored_carries_threshold_as_flag_plus_a_warning(
        self, censoring_a: Censoring, censoring_b: Censoring
    ) -> None:
        """ "No coherent direction exists": the ``Ratio`` carries
        ``threshold_a``'s own censoring flag, unflipped, together with an explicit warning."""
        t_a = fake_threshold(status=Status.OK, censoring=censoring_a)
        t_b = fake_threshold(status=Status.OK, censoring=censoring_b)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.status is Status.OK
        assert result.censoring is censoring_a
        assert result.estimate is None and result.lo is None and result.hi is None
        assert len(result.warnings) >= 1
        assert any(len(warning.strip()) > 20 for warning in result.warnings)


class TestCensoringDirectionAppliesRegardlessOfStatus:
    """The direction a ``Censoring`` value implies (module docstring) is a
    property of what ``RIGHT``/``LEFT``/``OPEN_*`` themselves mean against a tested range, not of
    why a threshold does not have a point value -- so the same flip applies whether the
    denominator's ``RIGHT`` came from a converged fit or from a ``SEPARATION``/``NOT_CONVERGED``
    exact-bound fallback (a failed fit "falls back to the exact censoring bound, labelled",
    which still classifies as ``RIGHT``/``LEFT``).

    The status rule applies as usual: ``Status.OK`` only when both inputs are ``Status.OK``;
    otherwise one failure wins. Here ``t_a`` is ``OK`` and ``t_b`` is ``NOT_CONVERGED``, so the
    ratio's status is ``NOT_CONVERGED`` -- asserted directly below.
    """

    def test_right_denominator_from_a_not_converged_fallback_still_flips_to_left(self) -> None:
        t_a = fake_threshold(status=Status.OK, censoring=Censoring.NONE)
        t_b = fake_threshold(status=Status.NOT_CONVERGED, censoring=Censoring.RIGHT)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.status is Status.NOT_CONVERGED
        assert result.censoring is Censoring.LEFT
        assert result.estimate is None and result.lo is None and result.hi is None


class TestStatusPropagationRule:
    """Status propagation, pinned directly: ``Status.OK`` only when both
    inputs are ``Status.OK``; when exactly one input has failed, that failure's status wins
    (symmetric -- it does not matter which side); when both inputs have failed with *different*
    statuses, the ``Ratio`` takes ``threshold_a``'s status and carries a ``warnings`` entry (no
    coherent single "reason" exists, the same rationale as the both-censored row of the
    censoring table above); two failures with the *same* status resolve to that status with no
    special warning required (there is nothing indeterminate to flag -- both inputs agree).
    """

    def test_one_failure_on_the_numerator_wins(self) -> None:
        t_a = fake_threshold(status=Status.SEPARATION, censoring=Censoring.NONE)
        t_b = fake_threshold(status=Status.OK, censoring=Censoring.NONE)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.status is Status.SEPARATION

    def test_one_failure_on_the_denominator_wins(self) -> None:
        t_a = fake_threshold(status=Status.OK, censoring=Censoring.NONE)
        t_b = fake_threshold(status=Status.UNREACHABLE, censoring=Censoring.NONE)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.status is Status.UNREACHABLE

    @pytest.mark.parametrize(
        ("status_a", "status_b"),
        [
            (Status.SEPARATION, Status.NOT_CONVERGED),
            (Status.UNREACHABLE, Status.FAILS_AT_BASELINE),
            (Status.CONTROL_INCOMPATIBLE, Status.SEPARATION),
        ],
    )
    def test_two_differing_failures_take_threshold_as_status_and_warn(
        self, status_a: Status, status_b: Status
    ) -> None:
        t_a = fake_threshold(status=status_a, censoring=Censoring.NONE)
        t_b = fake_threshold(status=status_b, censoring=Censoring.NONE)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.status is status_a
        assert len(result.warnings) >= 1
        assert any(len(warning.strip()) > 20 for warning in result.warnings)

    def test_two_failures_with_the_same_status_resolve_to_that_status(self) -> None:
        t_a = fake_threshold(status=Status.SEPARATION, censoring=Censoring.NONE)
        t_b = fake_threshold(status=Status.SEPARATION, censoring=Censoring.NONE)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.status is Status.SEPARATION
