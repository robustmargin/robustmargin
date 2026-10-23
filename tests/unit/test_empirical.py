"""Unit tests for ``robustmargin.empirical.grid_break_point``.

The first test class, ``TestGridRuleBasicCases``, covers the grid rule itself: the first
failing magnitude, right-censoring, and a signed axis. Case names are domain-neutral; the
signed-axis case uses literal curve values rather than deriving them from pooled counts.

Everything after it covers the API's deliberate choices (``ValueError`` instead of a silent
``(None, None)``, keyword-only ``criterion``, a frozen result, built-in ``float`` fields) and
the real-data regression over ``tests/data/pooled_counts.csv``.
"""

from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pytest

from robustmargin import Censoring, GridBreakPoint, grid_break_point

_DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "pooled_counts.csv"


class TestGridRuleBasicCases:
    """The grid rule's basic cases: first failing magnitude, signed axis, right-censoring."""

    def test_first_failing_magnitude(self) -> None:
        """Smallest severity whose performance is strictly below the criterion wins.

        Ports ``test_first_failing_magnitude``: the point at 5.0 sits exactly on the
        criterion (0.95) and must pass, since the rule is strict ``<``.
        """
        result = grid_break_point([0.0, 5.0, 10.0], [1.0, 0.95, 0.4], criterion=0.95)

        assert result.value == 10.0
        assert result.max_tested == 10.0
        assert result.censoring is Censoring.NONE

    def test_signed_axis_breaks_on_failing_side(self) -> None:
        """A signed axis breaks on whichever side fails first, by magnitude.

        Ports ``test_signed_family_breaks_on_failing_side``, using the literal curve from
        ``_synthetic_rows``'s ``mass`` fixture: the -0.1 side fails (0.9 < 0.95) before the
        +/-0.2 points do.
        """
        severity = [-0.2, -0.1, 0.0, 0.1, 0.2]
        performance = [0.2, 0.9, 1.0, 1.0, 1.0]

        result = grid_break_point(severity, performance, criterion=0.95, signed=True)

        assert result.value == 0.1
        assert result.max_tested == 0.2
        assert result.censoring is Censoring.NONE

    def test_right_censored_when_criterion_holds_everywhere(self) -> None:
        """Ports ``test_censored_when_gate_holds_everywhere``: no failing level at all."""
        result = grid_break_point([0.0, 5.0, 10.0], [1.0, 1.0, 0.99], criterion=0.95)

        assert result.value is None
        assert result.max_tested == 10.0
        assert result.censoring is Censoring.RIGHT

    def test_empty_input_raises(self) -> None:
        """Empty input is an error, not a result such as ``(None, None)``.

        robustmargin has no silent fallback: empty input raises rather than returning a
        plausible-looking result.
        """
        with pytest.raises(ValueError):
            grid_break_point([], [], criterion=0.95)


class TestValidation:
    """Every failure path the API contract lists, one behaviour per test."""

    def test_mismatched_lengths_raise_value_error(self) -> None:
        with pytest.raises(ValueError):
            grid_break_point([0.0, 1.0, 2.0], [1.0, 0.5], criterion=0.95)

    def test_non_1d_severity_raises_value_error(self) -> None:
        severity = [[0.0, 1.0], [2.0, 3.0]]
        performance = [1.0, 0.5]

        with pytest.raises(ValueError):
            grid_break_point(severity, performance, criterion=0.95)

    def test_non_1d_performance_raises_value_error(self) -> None:
        severity = [0.0, 2.0]
        performance = [[1.0, 1.0], [0.5, 0.5]]

        with pytest.raises(ValueError):
            grid_break_point(severity, performance, criterion=0.95)

    @pytest.mark.parametrize(
        ("severity", "performance", "criterion"),
        [
            pytest.param([0.0, float("nan"), 5.0], [1.0, 0.9, 0.4], 0.95, id="nan-severity"),
            pytest.param([0.0, 2.0, 5.0], [1.0, float("nan"), 0.4], 0.95, id="nan-performance"),
            pytest.param([0.0, 2.0, 5.0], [1.0, 0.9, 0.4], float("nan"), id="nan-criterion"),
            pytest.param([0.0, float("inf"), 5.0], [1.0, 0.9, 0.4], 0.95, id="inf-severity"),
            pytest.param([0.0, -float("inf"), 5.0], [1.0, 0.9, 0.4], 0.95, id="neg-inf-severity"),
            pytest.param([0.0, 2.0, 5.0], [1.0, float("inf"), 0.4], 0.95, id="inf-performance"),
            pytest.param([0.0, 2.0, 5.0], [1.0, 0.9, 0.4], float("inf"), id="inf-criterion"),
        ],
    )
    def test_non_finite_values_raise_value_error(
        self, severity: list[float], performance: list[float], criterion: float
    ) -> None:
        with pytest.raises(ValueError):
            grid_break_point(severity, performance, criterion=criterion)

    def test_negative_severity_unsigned_raises_value_error(self) -> None:
        """``signed=False`` is the default: a negative severity is a caller bug, not data."""
        with pytest.raises(ValueError):
            grid_break_point([-1.0, 0.0, 5.0], [1.0, 1.0, 0.4], criterion=0.95)

    def test_negative_severity_signed_is_accepted(self) -> None:
        """The mirror of the previous case: ``signed=True`` allows negative severities."""
        result = grid_break_point([-1.0, 0.0, 5.0], [0.4, 1.0, 1.0], criterion=0.95, signed=True)

        assert result.value == 1.0

    def test_criterion_is_keyword_only(self) -> None:
        """Passing ``criterion`` positionally is a ``TypeError``, not a lenient accept."""
        bad_call = cast(Callable[..., GridBreakPoint], grid_break_point)
        with pytest.raises(TypeError):
            bad_call([0.0, 5.0], [1.0, 0.4], 0.95)

    def test_result_is_frozen(self) -> None:
        result = GridBreakPoint(value=5.0, max_tested=10.0, censoring=Censoring.NONE)

        with pytest.raises(dataclasses.FrozenInstanceError):
            cast(Any, result).value = 1.0


class TestStrictInequalityAndTypes:
    def test_performance_equal_to_criterion_does_not_fail(self) -> None:
        """Equality passes: the rule is strict ``<``, not ``<=``."""
        result = grid_break_point([0.0, 5.0], [1.0, 0.95], criterion=0.95)

        assert result.value is None
        assert result.censoring is Censoring.RIGHT

    def test_value_and_max_tested_are_builtin_float(self) -> None:
        """Fields must be ``float``, not ``np.float64``, even when inputs are numpy arrays."""
        severity = np.array([0.0, 5.0, 10.0], dtype=np.float64)
        performance = np.array([1.0, 0.95, 0.4], dtype=np.float64)

        result = grid_break_point(severity, performance, criterion=0.95)

        assert type(result.value) is float
        assert type(result.max_tested) is float

    def test_max_tested_is_builtin_float_when_right_censored(self) -> None:
        """``value`` is ``None`` in the right-censored case; ``max_tested`` is still a float."""
        severity = np.array([0.0, 5.0, 10.0])
        performance = np.array([1.0, 1.0, 0.99])

        result = grid_break_point(severity, performance, criterion=0.95)

        assert result.value is None
        assert type(result.max_tested) is float

    def test_accepts_numpy_array_input(self) -> None:
        """Numpy arrays and plain lists must agree, since both are documented input shapes."""
        severity = [0.0, 5.0, 10.0]
        performance = [1.0, 0.95, 0.4]

        from_lists = grid_break_point(severity, performance, criterion=0.95)
        from_arrays = grid_break_point(np.array(severity), np.array(performance), criterion=0.95)

        assert from_lists == from_arrays

    def test_float32_performance_at_criterion_fails_on_precision(self) -> None:
        """Everything is cast up to float64, and the comparison is then exact in float64 --
        there is no tolerance.

        ``np.float32(0.95)`` is not exactly ``0.95``; its nearest float32 value widens to
        ``0.949999988079071`` once cast to float64, which is strictly below a float64
        ``criterion=0.95``. A comparison that narrowed the Python-float criterion down to
        float32 would find the two equal and pass the level. This test asserts robustmargin's
        float64 behaviour is deliberate, not a bug to fix.
        """
        severity = [0.0, 5.0]
        performance = np.array([1.0, 0.95], dtype=np.float32)

        result = grid_break_point(severity, performance, criterion=0.95)

        assert result.value == 5.0
        assert result.censoring is Censoring.NONE

    def test_unsigned_negative_zero_severity_is_accepted(self) -> None:
        """``-0.0`` is not ``< 0.0``, so ``signed=False``'s non-negativity check accepts it.

        A failing level at ``-0.0`` reports ``value == 0.0`` with the sign cleared, matching
        ``abs()``'s convention (used on ``severity`` in the source) rather than propagating the
        input's negative sign bit.
        """
        result = grid_break_point([-0.0, 1.0], [0.4, 1.0], criterion=0.95, signed=False)

        assert result.value == 0.0
        assert math.copysign(1.0, result.value) == 1.0


def _load_matrix_series() -> dict[str, tuple[list[float], list[float]]]:
    """Read ``tests/data/pooled_counts.csv`` into per-series (severity, rate) arrays."""
    series: dict[str, tuple[list[float], list[float]]] = {}
    with _DATA_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["series"]
            severities, rates = series.setdefault(name, ([], []))
            severities.append(float(row["severity"]))
            successes = float(row["successes"])
            trials = float(row["trials"])
            rates.append(successes / trials)
    return series


# (series, signed, expected value, expected max_tested, expected censoring). The expected values
# are literals, checked by hand against the pooled counts in tests/data/pooled_counts.csv.
_EXPECTED_MATRIX_RESULTS: tuple[tuple[str, bool, float | None, float, Censoring], ...] = (
    ("system_a_axis_1", False, 0.05, 0.1, Censoring.NONE),
    ("system_a_axis_2", False, None, 10.0, Censoring.RIGHT),
    ("system_b_axis_1", False, 0.0, 0.1, Censoring.NONE),
    ("system_c_axis_1", False, 0.0, 0.1, Censoring.NONE),
    ("system_c_axis_3", True, 0.1, 0.2, Censoring.NONE),
)


@pytest.mark.parametrize(
    ("series_name", "signed", "expected_value", "expected_max_tested", "expected_censoring"),
    _EXPECTED_MATRIX_RESULTS,
    ids=[case[0] for case in _EXPECTED_MATRIX_RESULTS],
)
def test_matrix_series_break_point_at_criterion_0_95(
    series_name: str,
    signed: bool,
    expected_value: float | None,
    expected_max_tested: float,
    expected_censoring: Censoring,
) -> None:
    """Real-data regression: every series in the pooled-counts fixture at the 0.95 gate."""
    matrix = _load_matrix_series()
    severity, rate = matrix[series_name]

    result = grid_break_point(severity, rate, criterion=0.95, signed=signed)

    assert result.value == expected_value
    assert result.max_tested == pytest.approx(expected_max_tested)
    assert result.censoring is expected_censoring


def test_matrix_series_names_match_fixture() -> None:
    """The parametrised cases above must cover every series the fixture actually has."""
    matrix = _load_matrix_series()
    expected_names = {case[0] for case in _EXPECTED_MATRIX_RESULTS}

    assert set(matrix) == expected_names


class TestProvenanceAndSchemaVersion:
    """``GridBreakPoint`` carries ``schema_version`` and ``provenance``: both come after the
    first three fields and both have defaults, so the three-field positional form stays valid.
    """

    def test_field_order_is_value_max_tested_censoring_schema_version_provenance(self) -> None:
        field_names = tuple(f.name for f in dataclasses.fields(GridBreakPoint))

        assert field_names == (
            "value",
            "max_tested",
            "censoring",
            "schema_version",
            "provenance",
        )

    def test_default_schema_version_is_the_current_package_version(self) -> None:
        """``GridBreakPoint``'s default tracks the single, package-wide ``SCHEMA_VERSION``
        (``"2"``), since ``report.py`` versions the whole card format, not each type
        independently. This test name does not hardcode which version that is."""
        result = grid_break_point([0.0, 5.0, 10.0], [1.0, 0.95, 0.4], criterion=0.95)

        assert result.schema_version == "2"

    def test_default_provenance_is_an_empty_mapping(self) -> None:
        import collections.abc

        result = grid_break_point([0.0, 5.0, 10.0], [1.0, 0.95, 0.4], criterion=0.95)

        assert isinstance(result.provenance, collections.abc.Mapping)
        assert len(result.provenance) == 0

    def test_replace_sets_provenance_without_disturbing_other_fields(self) -> None:
        result = grid_break_point([0.0, 5.0, 10.0], [1.0, 0.95, 0.4], criterion=0.95)
        provenance = {"run": "abc", "nested": {"k": [1, 2]}}

        replaced = dataclasses.replace(result, provenance=provenance)

        assert replaced.provenance == provenance
        assert replaced.value == result.value
        assert replaced.max_tested == result.max_tested
        assert replaced.censoring == result.censoring
        assert replaced.schema_version == result.schema_version

    def test_positional_three_field_construction_fills_in_defaults(self) -> None:
        """The three-field positional form must keep working (backward compat)."""
        result = GridBreakPoint(0.1, 0.2, Censoring.NONE)

        assert result.value == 0.1
        assert result.max_tested == 0.2
        assert result.censoring is Censoring.NONE
        # package-wide SCHEMA_VERSION
        assert result.schema_version == "2"
        assert len(result.provenance) == 0
