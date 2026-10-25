"""The mirrored wording of the fit-time boundary and separation diagnostics.

An increasing outcome is fit on its complement, so the optimizer's asymptotes are the complement's:
``upper' = 1 - lower`` and ``lower' = 1 - upper``. A message about an asymptote that reached a
boundary must name the asymptote of the outcome the caller supplied, and report its probability as
``1 - p``; separation must speak of the counted outcome. These call the private helpers directly
with the mirrored flag, because forcing the optimizer onto a boundary on demand is not
reproducible with small data, and compare each against the unmirrored wording.
"""

from __future__ import annotations

import numpy as np
import pytest

from robustmargin import Axis, Cell
from robustmargin.models import _boundary_warning, _check_separation

_EPS = 1e-6
_LOG = Axis(name="severity", unit="unit", scale="log")
_LINEAR = Axis(name="severity", unit="unit", scale="linear")


def _params(*extra: float) -> np.ndarray:
    return np.array([0.0, 0.0, *extra])


class TestBoundaryWarning:
    def test_a_logit_parameter_on_the_complements_lower_names_the_callers_upper(self) -> None:
        message = _boundary_warning(
            _params(20.0),
            estimate_lower=True,
            estimate_upper=False,
            lower=0.2,
            upper=0.9,
            mirrored=True,
        )

        assert message is not None
        assert "estimated upper asymptote" in message
        assert "logit-scale a=" not in message  # the optimizer's own value is not the caller's

    def test_a_logit_parameter_on_the_complements_upper_names_the_callers_lower(self) -> None:
        message = _boundary_warning(
            _params(-20.0),
            estimate_lower=False,
            estimate_upper=True,
            lower=0.1,
            upper=0.8,
            mirrored=True,
        )

        assert message is not None
        assert "estimated lower asymptote" in message
        assert "logit-scale b=" not in message

    def test_the_unmirrored_message_keeps_the_optimizers_value(self) -> None:
        message = _boundary_warning(
            _params(20.0), estimate_lower=True, estimate_upper=False, lower=0.2, upper=0.9
        )

        assert message is not None
        assert "estimated lower asymptote" in message and "logit-scale a=20.0" in message

    def test_both_estimated_with_the_complements_lower_at_zero_is_the_callers_upper_at_one(
        self,
    ) -> None:
        message = _boundary_warning(
            _params(0.0, 0.0),
            estimate_lower=True,
            estimate_upper=True,
            lower=1e-9,
            upper=0.9,
            mirrored=True,
        )

        assert message is not None
        assert "estimated upper asymptote" in message
        assert f"> 1 - {_EPS}" in message
        assert repr(1.0 - 1e-9) in message

    def test_a_lower_estimated_next_to_a_fixed_upper_swaps_both_names_when_mirrored(self) -> None:
        kwargs = dict(estimate_lower=True, estimate_upper=False, lower=0.5 - 1e-9, upper=0.5)

        mirrored = _boundary_warning(_params(0.0), mirrored=True, **kwargs)  # type: ignore[arg-type]
        plain = _boundary_warning(_params(0.0), **kwargs)  # type: ignore[arg-type]

        assert mirrored is not None and plain is not None
        assert "estimated upper asymptote" in mirrored and "fixed lower asymptote" in mirrored
        assert "estimated lower asymptote" in plain and "fixed upper asymptote" in plain

    def test_an_upper_estimated_at_one_is_the_callers_lower_at_zero_when_mirrored(self) -> None:
        mirrored = _boundary_warning(
            _params(0.0),
            estimate_lower=False,
            estimate_upper=True,
            lower=0.1,
            upper=1.0 - 1e-9,
            mirrored=True,
        )

        assert mirrored is not None
        assert "estimated lower asymptote" in mirrored
        assert f"< {_EPS}" in mirrored

    def test_an_interior_fit_has_no_boundary_warning_either_way(self) -> None:
        for mirrored in (False, True):
            assert (
                _boundary_warning(
                    _params(0.0, 0.0),
                    estimate_lower=True,
                    estimate_upper=True,
                    lower=0.1,
                    upper=0.9,
                    mirrored=mirrored,
                )
                is None
            )


class TestSeparationWording:
    def _cells(self, successes: list[int]) -> tuple[Cell, ...]:
        severities = [0.0, 1.0, 2.0, 3.0][: len(successes)]
        return tuple(
            Cell(severity=s, successes=k, trials=100)
            for s, k in zip(severities, successes, strict=True)
        )

    def test_no_success_in_the_complement_reads_as_no_failure_in_the_counted_outcome(self) -> None:
        cells = self._cells([100, 0, 0, 0])

        assert "no success" in (_check_separation(cells, axis=_LOG) or "")
        assert "no failure" in (_check_separation(cells, axis=_LOG, mirrored=True) or "")

    def test_no_failure_in_the_complement_reads_as_no_success_in_the_counted_outcome(self) -> None:
        cells = self._cells([100, 100, 100, 100])

        assert "no failure" in (_check_separation(cells, axis=_LOG) or "")
        assert "no success" in (_check_separation(cells, axis=_LOG, mirrored=True) or "")

    def test_quasi_complete_separation_swaps_the_two_nouns_when_mirrored(self) -> None:
        cells = self._cells([0, 100, 100, 0])

        plain = _check_separation(cells, axis=_LOG)
        mirrored = _check_separation(cells, axis=_LOG, mirrored=True)

        assert plain is not None and mirrored is not None
        assert "with a success is at or below the lowest severity with a failure" in plain
        assert "with a failure is at or below the lowest severity with a success" in mirrored

    def test_a_log_axis_with_only_the_control_has_nothing_to_check(self) -> None:
        cells = (Cell(severity=0.0, successes=0, trials=100),)

        assert _check_separation(cells, axis=_LOG, mirrored=True) is None

    def test_a_linear_axis_checks_the_zero_severity_level_too(self) -> None:
        cells = (Cell(severity=0.0, successes=0, trials=100),)

        assert _check_separation(cells, axis=_LINEAR, mirrored=True) is not None

    def test_overlapping_levels_do_not_separate(self) -> None:
        cells = self._cells([95, 60, 20, 5])

        assert _check_separation(cells, axis=_LOG, mirrored=True) is None


@pytest.mark.parametrize("mirrored", [False, True])
def test_wording_never_mentions_the_other_orientations_asymptote_for_a_fixed_upper(
    mirrored: bool,
) -> None:
    message = _boundary_warning(
        _params(0.0),
        estimate_lower=True,
        estimate_upper=False,
        lower=0.5 - 1e-9,
        upper=0.5,
        mirrored=mirrored,
    )

    assert message is not None
    estimated, fixed = ("upper", "lower") if mirrored else ("lower", "upper")
    assert f"estimated {estimated} asymptote" in message
    assert f"fixed {fixed} asymptote" in message
