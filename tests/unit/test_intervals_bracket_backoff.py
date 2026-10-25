"""``_bracket_root`` and ``_solve_profile_bracket`` where the inner fit fails partway out.

The profile search steps outward from ``theta_hat`` in strides that grow by a factor of 1.7. Where
the inner fit fails (``g`` returns ``None``) a long stride can jump from a good ``theta`` straight
past a real sign change into the failing region. The search then shortens the stride: it bisects
between the last good ``theta`` and the failure, at most ``_MAX_BACKOFF_HALVINGS`` times, and
returns a bracket as soon as one is found. If none is, the side is open at the last good ``theta``.

The synthetic ``g`` here is written in the distance ``x = direction * (theta - theta_hat)`` from the
estimate, so the same case runs outward in either direction. It is negative at ``x = 0``, crosses
zero at ``x = root``, and is ``None`` from ``x = frontier`` on.

With ``theta_hat = 0`` and a limit of 100 the strides are ``0.01 * 1.7**n``: 3.43 (``n = 11``),
5.83 (``n = 12``), 9.91 (``n = 13``). A frontier at 5.0 therefore sits between the last good point
(3.43) and the first failing one (5.83).
"""

from __future__ import annotations

import re
from collections.abc import Callable

import pytest

from robustmargin import Censoring, intervals
from robustmargin.intervals import _bracket_root, _solve_profile_bracket

_HALVINGS = 30
_FIRST_STRIDE_FRACTION = 1.0e-4
_GROWTH = 1.7
_LIMIT = 100.0
_THETA_HAT = 2.0

_DIRECTIONS = pytest.mark.parametrize("direction", [1.0, -1.0])


def _stride_positions(span: float) -> list[float]:
    """Distances from the estimate at which the stride rule evaluates ``g``, unclamped."""
    step = max(span * _FIRST_STRIDE_FRACTION, 1.0e-8)
    out: list[float] = []
    for _ in range(200):
        step *= _GROWTH
        out.append(step)
    return out


def _g(direction: float, *, root: float, frontier: float | None) -> Callable[[float], float | None]:
    def g(theta: float) -> float | None:
        x = direction * (theta - _THETA_HAT)
        if frontier is not None and x >= frontier:
            return None
        return x - root

    return g


def _spy(g: Callable[[float], float | None], calls: list[float]) -> Callable[[float], float | None]:
    def spy(theta: float) -> float | None:
        calls.append(theta)
        return g(theta)

    return spy


def _expected_evaluations(direction: float, root: float) -> list[float]:
    """The thetas the stride rule evaluates for ``g = x - root`` with no failure, in order."""
    span = _LIMIT
    limit = _THETA_HAT + direction * _LIMIT
    seen = [_THETA_HAT]
    step = max(span * _FIRST_STRIDE_FRACTION, 1.0e-8)
    for _ in range(200):
        step *= _GROWTH
        theta = _THETA_HAT + direction * step
        reached = (direction > 0.0 and theta >= limit) or (direction < 0.0 and theta <= limit)
        if reached:
            theta = limit
        seen.append(theta)
        if direction * (theta - _THETA_HAT) - root >= 0.0 or reached:
            break
    return seen


def _call(
    g: Callable[[float], float | None], direction: float
) -> tuple[str, float] | tuple[str, float, float]:
    return _bracket_root(g, _THETA_HAT, _THETA_HAT + direction * _LIMIT, direction)


class TestARootBeforeTheFailureFrontier:
    @_DIRECTIONS
    def test_a_stride_that_overshoots_the_frontier_backs_off_to_a_bracket(
        self, direction: float
    ) -> None:
        g = _g(direction, root=4.5, frontier=5.0)

        outcome = _call(g, direction)

        assert outcome[0] == "bracket" and len(outcome) == 3
        _, a, b = outcome
        ga, gb = g(a), g(b)
        assert ga is not None and gb is not None
        assert ga < 0.0 <= gb

    @_DIRECTIONS
    def test_the_bracket_contains_the_root(self, direction: float) -> None:
        outcome = _call(_g(direction, root=4.5, frontier=5.0), direction)

        assert outcome[0] == "bracket"
        xs = sorted(direction * (t - _THETA_HAT) for t in outcome[1:])
        assert xs[0] < 4.5 <= xs[1]

    @_DIRECTIONS
    def test_a_root_just_inside_the_frontier_is_still_bracketed(self, direction: float) -> None:
        """Several midpoints fail before one lands between the root and the frontier."""
        root = 5.0 - 1.0e-4
        g = _g(direction, root=root, frontier=5.0)

        outcome = _call(g, direction)

        assert outcome[0] == "bracket"
        xs = sorted(direction * (t - _THETA_HAT) for t in outcome[1:])
        assert xs[0] < root <= xs[1] < 5.0

    @_DIRECTIONS
    def test_the_returned_bracket_ends_are_finite(self, direction: float) -> None:
        outcome = _call(_g(direction, root=4.5, frontier=5.0), direction)

        assert outcome[0] == "bracket"
        assert all(t == t and abs(t) < float("inf") for t in outcome[1:])


class TestNoRootBeforeTheFailureFrontier:
    @_DIRECTIONS
    def test_the_side_is_open_at_a_point_next_to_the_frontier_on_the_good_side(
        self, direction: float
    ) -> None:
        frontier = 5.0
        g = _g(direction, root=8.0, frontier=frontier)
        positions = _stride_positions(_LIMIT)
        good0 = max(p for p in positions if p < frontier)
        failed0 = min(p for p in positions if p >= frontier)

        outcome = _call(g, direction)

        assert outcome[0] == "open_failed"
        x = direction * (outcome[1] - _THETA_HAT)
        assert good0 <= x < frontier
        # Each halving halves the gap between the last good point and the first failing one,
        # which starts as one stride wide and always contains the frontier.
        assert frontier - x <= (failed0 - good0) / 2**_HALVINGS

    @_DIRECTIONS
    def test_g_is_negative_at_the_point_reported_open(self, direction: float) -> None:
        g = _g(direction, root=8.0, frontier=5.0)

        outcome = _call(g, direction)

        assert outcome[0] == "open_failed"
        value = g(outcome[1])
        assert value is not None and value < 0.0

    @_DIRECTIONS
    def test_failure_at_the_first_stride_backs_off_toward_the_estimate(
        self, direction: float
    ) -> None:
        g = _g(direction, root=8.0, frontier=1.0e-9)
        first = _stride_positions(_LIMIT)[0]

        outcome = _call(g, direction)

        assert outcome[0] == "open_failed"
        x = direction * (outcome[1] - _THETA_HAT)
        assert 0.0 <= x < 1.0e-9 + first / 2**_HALVINGS

    @_DIRECTIONS
    def test_the_estimate_itself_failing_is_reported_unchanged(self, direction: float) -> None:
        def g(theta: float) -> float | None:
            return None

        assert _call(g, direction) == ("open_failed", _THETA_HAT)


class TestEvaluationCounts:
    @_DIRECTIONS
    def test_the_evaluations_after_a_failure_are_at_most_the_halving_limit(
        self, direction: float
    ) -> None:
        calls: list[float] = []
        g = _spy(_g(direction, root=8.0, frontier=5.0), calls)

        _call(g, direction)

        first_failure = next(i for i, t in enumerate(calls) if direction * (t - _THETA_HAT) >= 5.0)
        assert 0 < len(calls) - 1 - first_failure <= _HALVINGS

    @_DIRECTIONS
    def test_a_search_with_no_failure_evaluates_the_same_thetas_as_the_stride_rule(
        self, direction: float
    ) -> None:
        calls: list[float] = []
        g = _spy(_g(direction, root=50.0, frontier=None), calls)

        outcome = _call(g, direction)

        assert outcome[0] == "bracket"
        assert calls == _expected_evaluations(direction, 50.0)
        assert outcome[1:] == (calls[-2], calls[-1])

    @_DIRECTIONS
    def test_a_search_that_reaches_the_limit_with_no_failure_is_unchanged(
        self, direction: float
    ) -> None:
        calls: list[float] = []
        g = _spy(_g(direction, root=1.0e6, frontier=None), calls)

        outcome = _call(g, direction)

        limit = _THETA_HAT + direction * _LIMIT
        assert outcome == ("open_limit", limit)
        assert calls == _expected_evaluations(direction, 1.0e6)

    def test_the_halving_limit_is_thirty(self) -> None:
        assert getattr(intervals, "_MAX_BACKOFF_HALVINGS", None) == _HALVINGS


class TestAnInteriorFailureInsideAFoundBracket:
    """The strides bracket the root cleanly (``g`` is valid at both ends), but the root finder's
    first interior point falls in a stretch where the inner fit fails."""

    @staticmethod
    def _g(direction: float) -> Callable[[float], float | None]:
        def g(theta: float) -> float | None:
            x = direction * theta
            if 3.5 < x < 5.8:
                return None
            return x - 3.9

        return g

    @_DIRECTIONS
    def test_the_side_is_open_with_the_inner_fit_warning_and_no_exception(
        self, direction: float
    ) -> None:
        limits = (0.0, 100.0) if direction > 0.0 else (-100.0, 0.0)

        result = _solve_profile_bracket(
            self._g(direction),
            theta_hat=0.0,
            lower_limit=limits[0],
            upper_limit=limits[1],
            scale="linear",
        )

        label = "upper" if direction > 0.0 else "lower"
        assert (result.lo, result.hi) == (None, None)
        assert result.censoring is Censoring.NONE
        g = self._g(direction)
        pattern = re.compile(
            rf"inner fit failed inside the {label} bracket "
            r"\[(?P<a>[^,\]]+), (?P<b>[^\]]+)\] while solving for the root "
            r"\(theta=(?P<theta>[^)]+)\); " + f"{label} side left open$"
        )
        matches = [m for w in result.warnings if (m := pattern.search(w))]
        assert len(matches) == 1, result.warnings
        a, b, theta = (float(matches[0][k]) for k in ("a", "b", "theta"))
        ga, gb = g(a), g(b)
        assert ga is not None and gb is not None and ga * gb < 0.0  # bracket holds the root
        assert g(theta) is None
        assert min(a, b) < theta < max(a, b)
        assert not any("while stepping toward" in w for w in result.warnings), result.warnings
