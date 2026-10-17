"""Property tests for ``robustmargin.empirical.grid_break_point`` (hypothesis).

Invariants that must hold for any valid input, not just the hand-picked cases in
``tests/unit/test_empirical.py``: permutation invariance, ``value`` is always one of the
tested magnitudes (or ``None``), ``value is None`` if and only if the criterion held at every
level if and only if ``censoring is Censoring.RIGHT``, ``max_tested`` is the max magnitude, and
agreement with a direct, independent re-statement of the two-line rule. Example counts are
kept modest so the suite stays fast.
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from robustmargin import Censoring, grid_break_point

_MAX_EXAMPLES = 100


@st.composite
def _curves(draw: st.DrawFn) -> tuple[list[float], list[float], float, bool]:
    """Draw a valid ``(severity, performance, criterion, signed)`` call."""
    signed = draw(st.booleans())
    n = draw(st.integers(min_value=1, max_value=8))

    severity_strategy = st.floats(
        min_value=-1_000.0 if signed else 0.0,
        max_value=1_000.0,
        allow_nan=False,
        allow_infinity=False,
    )
    performance_strategy = st.floats(
        min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
    )

    severity = draw(st.lists(severity_strategy, min_size=n, max_size=n))
    performance = draw(st.lists(performance_strategy, min_size=n, max_size=n))
    criterion = draw(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    return severity, performance, criterion, signed


@st.composite
def _curves_with_tied_criterion(draw: st.DrawFn) -> tuple[list[float], list[float], float, bool]:
    """Like :func:`_curves`, but draws ``criterion`` so it sometimes lands exactly on a
    drawn ``performance`` value.

    With ``criterion`` drawn independently (as in :func:`_curves`), an exact tie between
    ``criterion`` and some ``performance`` entry is a probability-zero event, so a strict
    ``<`` requirement would never actually be exercised: a ``<=`` implementation would pass
    every example just as often as the correct one. Sampling from the drawn performance
    values forces ties to occur.
    """
    severity, performance, _criterion, signed = draw(_curves())
    criterion = draw(
        st.one_of(
            st.sampled_from(performance),
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        )
    )
    return severity, performance, criterion, signed


@given(_curves(), st.data())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_row_permutation_does_not_change_result(
    curve: tuple[list[float], list[float], float, bool],
    data: st.DataObject,
) -> None:
    """Shuffling ``(severity, performance)`` pairs together must not change the result.

    Draws an arbitrary permutation via ``st.permutations`` rather than a single fixed
    reversal, so hypothesis can explore shufflings beyond "reverse the list" (including the
    identity permutation and permutations that only move a subset of rows).
    """
    severity, performance, criterion, signed = curve
    n = len(severity)
    order = data.draw(st.permutations(range(n)))
    shuffled_severity = [severity[i] for i in order]
    shuffled_performance = [performance[i] for i in order]

    original = grid_break_point(severity, performance, criterion=criterion, signed=signed)
    shuffled = grid_break_point(
        shuffled_severity, shuffled_performance, criterion=criterion, signed=signed
    )

    assert original == shuffled


@given(_curves())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_value_is_a_tested_magnitude_or_none(
    curve: tuple[list[float], list[float], float, bool],
) -> None:
    severity, performance, criterion, signed = curve

    result = grid_break_point(severity, performance, criterion=criterion, signed=signed)

    if result.value is not None:
        tested_magnitudes = {abs(s) for s in severity}
        assert result.value in tested_magnitudes


@given(_curves_with_tied_criterion())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_value_none_iff_no_level_fails_iff_right_censored(
    curve: tuple[list[float], list[float], float, bool],
) -> None:
    severity, performance, criterion, signed = curve

    result = grid_break_point(severity, performance, criterion=criterion, signed=signed)

    no_level_fails = all(p >= criterion for p in performance)

    assert (result.value is None) == no_level_fails
    assert (result.value is None) == (result.censoring is Censoring.RIGHT)
    # The grid rule only ever reports NONE or RIGHT, never LEFT or OPEN_*: it is a two-outcome
    # rule, and the full Censoring enum is for model-based results.
    assert result.censoring in (Censoring.NONE, Censoring.RIGHT)


@given(_curves())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_max_tested_equals_max_absolute_severity(
    curve: tuple[list[float], list[float], float, bool],
) -> None:
    severity, performance, criterion, signed = curve

    result = grid_break_point(severity, performance, criterion=criterion, signed=signed)

    assert result.max_tested == max(abs(s) for s in severity)


@given(_curves())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_value_never_exceeds_max_tested(
    curve: tuple[list[float], list[float], float, bool],
) -> None:
    severity, performance, criterion, signed = curve

    result = grid_break_point(severity, performance, criterion=criterion, signed=signed)

    if result.value is not None:
        assert result.value <= result.max_tested


@given(_curves_with_tied_criterion())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_agrees_with_the_two_line_rule(
    curve: tuple[list[float], list[float], float, bool],
) -> None:
    """Independent re-statement of the grid rule's two lines.

    ``max_tested = max(abs(sev) for sev, _ in curve)``;
    ``failing = sorted(abs(sev) for sev, success in curve if success < gate)``;
    ``return (failing[0] if failing else None), max_tested``.
    This is a second, independently written implementation of the same rule, so it catches
    transcription bugs that a shared helper would hide. Drawing from
    ``_curves_with_tied_criterion`` (rather than ``_curves``) means ``criterion`` sometimes
    equals a tested ``performance`` value exactly, so the ``<`` in both this re-statement and
    the implementation actually gets exercised at the boundary, not just away from it.
    """
    severity, performance, criterion, _signed = curve

    expected_max_tested = max(abs(s) for s in severity)
    failing = sorted(abs(s) for s, p in zip(severity, performance, strict=True) if p < criterion)
    expected_value = failing[0] if failing else None

    # signed=True always, since the reference rule always breaks on magnitude (cards.py never
    # validates sign — that check is robustmargin's own addition).
    result = grid_break_point(severity, performance, criterion=criterion, signed=True)

    assert result.value == expected_value
    assert result.max_tested == expected_max_tested


@given(_curves(), st.integers(min_value=-4, max_value=4))
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_scaling_severity_scales_value_and_max_tested(
    curve: tuple[list[float], list[float], float, bool],
    exponent: int,
) -> None:
    """Rescaling severity by ``c > 0`` scales ``value`` and ``max_tested`` by ``c`` and leaves
    ``censoring`` unchanged (the units property: unit correctness is a correctness property).

    ``c`` is restricted to a power of two: multiplying an IEEE-754 float by a power of two only
    ever moves its exponent, never rounds its mantissa, so the scaled result matches ``c *
    original`` without accumulating float noise from the multiplication itself. ``math.isclose``
    is still used for the comparison, as a second line of defence rather than the only one.
    """
    severity, performance, criterion, signed = curve
    c = 2.0**exponent

    original = grid_break_point(severity, performance, criterion=criterion, signed=signed)
    scaled_severity = [c * s for s in severity]
    scaled = grid_break_point(scaled_severity, performance, criterion=criterion, signed=signed)

    assert scaled.censoring is original.censoring
    assert math.isclose(scaled.max_tested, c * original.max_tested, rel_tol=1e-12)
    if original.value is None:
        assert scaled.value is None
    else:
        assert scaled.value is not None
        assert math.isclose(scaled.value, c * original.value, rel_tol=1e-12)
