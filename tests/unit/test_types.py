"""Unit tests for ``robustmargin.types``: ``Axis``, ``Observations``, ``Definition``, ``Cell``,
and the ``Status`` / ``IntervalShape`` enums.

Every validation rule is exercised as its own test, one behaviour per test, so a loosened check
shows up as one specific failing test rather than a vague one.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import pytest

from robustmargin import Axis, Cell, Definition, IntervalShape, Observations, Status

_AXIS = Axis(name="noise", unit="m", scale="log")


class TestStatusEnum:
    """``Status`` is a ``StrEnum`` whose value equals its member name, following ``Censoring``."""

    def test_members_and_values(self) -> None:
        members = {member.name: member.value for member in Status}

        assert members == {
            "OK": "OK",
            "SEPARATION": "SEPARATION",
            "NOT_CONVERGED": "NOT_CONVERGED",
            "UNREACHABLE": "UNREACHABLE",
            "FAILS_AT_BASELINE": "FAILS_AT_BASELINE",
            "CONTROL_INCOMPATIBLE": "CONTROL_INCOMPATIBLE",
        }

    def test_is_a_str_subclass_whose_serialised_form_is_the_bare_name(self) -> None:
        assert isinstance(Status.OK, str)
        assert Status.OK == "OK"


class TestIntervalShapeEnum:
    """``IntervalShape`` is a ``StrEnum`` whose value equals its member name.

    ``HALF_OPEN`` is a member because a Fieller set can be bounded on one side only. This
    test's job is to catch an *undecided* change to the set: a deliberate change updates the
    pinned set, which is exactly what keeps the test meaningful for the next one.
    """

    def test_members_and_values(self) -> None:
        members = {member.name: member.value for member in IntervalShape}

        assert members == {
            "BOUNDED": "BOUNDED",
            "UNBOUNDED": "UNBOUNDED",
            "EXCLUSIVE": "EXCLUSIVE",
            "HALF_OPEN": "HALF_OPEN",
        }

    def test_is_a_str_subclass_whose_serialised_form_is_the_bare_name(self) -> None:
        assert isinstance(IntervalShape.BOUNDED, str)
        assert IntervalShape.BOUNDED == "BOUNDED"


class TestAxis:
    def test_valid_construction_stores_every_field(self) -> None:
        axis = Axis(name="axis_1", unit="m", scale="log", citation="ISO/TS 22133")

        assert axis.name == "axis_1"
        assert axis.unit == "m"
        assert axis.scale == "log"
        assert axis.citation == "ISO/TS 22133"

    def test_citation_defaults_to_none(self) -> None:
        axis = Axis(name="axis_2", unit="m/s", scale="linear")

        assert axis.citation is None

    @pytest.mark.parametrize("name", ["", "   "])
    def test_empty_or_whitespace_only_name_raises_value_error(self, name: str) -> None:
        with pytest.raises(ValueError):
            Axis(name=name, unit="m", scale="log")

    @pytest.mark.parametrize("unit", ["", "   "])
    def test_empty_or_whitespace_only_unit_raises_value_error(self, unit: str) -> None:
        with pytest.raises(ValueError):
            Axis(name="noise", unit=unit, scale="log")

    def test_linear_scale_is_accepted(self) -> None:
        axis = Axis(name="staleness", unit="s", scale="linear")

        assert axis.scale == "linear"

    def test_scale_other_than_log_or_linear_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Axis(name="noise", unit="m", scale="quadratic")

    def test_empty_citation_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Axis(name="noise", unit="m", scale="log", citation="")

    def test_is_frozen(self) -> None:
        axis = Axis(name="noise", unit="m", scale="log")

        with pytest.raises(dataclasses.FrozenInstanceError):
            cast(Any, axis).name = "other"


class TestDefinition:
    def test_absolute_stores_kind_and_value(self) -> None:
        definition = Definition.absolute(0.95)

        assert definition.kind == "absolute"
        assert definition.value == 0.95

    def test_absolute_accepts_any_finite_value_including_above_one(self) -> None:
        """Whether ``a`` is reachable is the ``UNREACHABLE`` status of a threshold, not a
        construction error, so an out-of-range-looking value must still construct."""
        definition = Definition.absolute(1.5)

        assert definition.value == 1.5

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_absolute_rejects_non_finite_values(self, value: float) -> None:
        with pytest.raises(ValueError):
            Definition.absolute(value)

    def test_absolute_rejects_a_bool(self) -> None:
        """A bool satisfies ``math.isfinite`` (``isfinite(True)`` is
        ``True``), so the existing finiteness check alone does not reject it."""
        with pytest.raises(ValueError):
            Definition.absolute(True)

    def test_direct_construction_rejects_a_bool_value(self) -> None:
        with pytest.raises(ValueError):
            Definition(kind="absolute", value=True)

    @pytest.mark.parametrize("constructor_name", ["baseline_fraction", "relative"])
    def test_fraction_constructors_accept_an_interior_value(self, constructor_name: str) -> None:
        constructor = getattr(Definition, constructor_name)

        definition = constructor(0.5)

        assert definition.kind == constructor_name
        assert definition.value == 0.5

    @pytest.mark.parametrize("constructor_name", ["baseline_fraction", "relative"])
    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_fraction_constructors_reject_the_endpoints(
        self, constructor_name: str, value: float
    ) -> None:
        constructor = getattr(Definition, constructor_name)

        with pytest.raises(ValueError):
            constructor(value)

    def test_direct_construction_applies_the_same_checks_as_the_named_constructor(self) -> None:
        with pytest.raises(ValueError):
            Definition(kind="baseline_fraction", value=0.0)

    def test_direct_construction_of_a_valid_definition_matches_the_named_constructor(self) -> None:
        assert Definition(kind="absolute", value=0.95) == Definition.absolute(0.95)

    def test_is_frozen(self) -> None:
        definition = Definition.absolute(0.95)

        with pytest.raises(dataclasses.FrozenInstanceError):
            cast(Any, definition).value = 0.5


class TestCell:
    """``Cell`` follows the same count rules as ``Observations``, since it is a single pooled
    row of the same data."""

    def test_valid_construction(self) -> None:
        cell = Cell(severity=0.05, successes=122, trials=300)

        assert cell.severity == 0.05
        assert cell.successes == 122
        assert cell.trials == 300

    def test_negative_severity_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Cell(severity=-1.0, successes=0, trials=4)

    def test_successes_greater_than_trials_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Cell(severity=0.0, successes=5, trials=4)

    def test_negative_successes_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Cell(severity=0.0, successes=-1, trials=4)

    def test_trials_less_than_one_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Cell(severity=0.0, successes=0, trials=0)

    def test_is_frozen(self) -> None:
        cell = Cell(severity=0.0, successes=1, trials=2)

        with pytest.raises(dataclasses.FrozenInstanceError):
            cast(Any, cell).successes = 2

    def test_bool_successes_raises_value_error(self) -> None:
        """``True``/``False`` satisfy the numeric ``0 <= successes <= trials`` checks
        by coercion, so a dedicated bool check is needed."""
        with pytest.raises(ValueError):
            Cell(severity=0.0, successes=True, trials=2)

    def test_non_whole_trials_raises_value_error(self) -> None:
        """``from_counts`` already rejects a non-whole ``trials`` via
        ``_as_whole_int``; direct ``Cell`` construction bypasses that classmethod entirely."""
        with pytest.raises(ValueError):
            Cell(severity=0.0, successes=1, trials=2.5)

    def test_bool_severity_raises_value_error(self) -> None:
        """``True < 0.0`` is ``False``, so the existing "severity must
        be non-negative" check alone does not reject a bool severity."""
        with pytest.raises(ValueError):
            Cell(severity=True, successes=1, trials=2)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_severity_raises_value_error(self, value: float) -> None:
        """Neither ``nan < 0.0`` nor ``inf < 0.0`` is ``True``, so the
        existing non-negativity check alone does not reject either."""
        with pytest.raises(ValueError):
            Cell(severity=value, successes=1, trials=2)

    def test_whole_number_float_counts_are_accepted_and_stored_as_int(self) -> None:
        """A whole-number float such as ``200.0`` is accepted and stored as ``int``, because JSON
        Schema cannot tell ``200.0`` from ``200`` (a real round-trip through ``json.loads``
        would produce exactly this)."""
        cell = Cell(severity=0.0, successes=200.0, trials=200.0)

        assert cell.successes == 200
        assert type(cell.successes) is int
        assert cell.trials == 200
        assert type(cell.trials) is int

    def test_non_whole_float_successes_still_raises_value_error(self) -> None:
        """Only a *whole-number* float is accepted; ``2.5`` stays rejected."""
        with pytest.raises(ValueError):
            Cell(severity=0.0, successes=2.5, trials=10)


class TestObservationsRequiredKeywords:
    """``outcome`` and ``direction`` are required keywords with no default:
    a wrong default would silently flip a threshold, so omitting either is a ``TypeError``, not
    a lenient accept."""

    def test_from_counts_without_outcome_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            cast(Any, Observations.from_counts)(
                _AXIS, severity=[0.0], successes=[1], trials=[1], direction="decreasing"
            )

    def test_from_counts_without_direction_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            cast(Any, Observations.from_counts)(
                _AXIS, severity=[0.0], successes=[1], trials=[1], outcome="success"
            )

    def test_from_observations_without_outcome_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            cast(Any, Observations.from_observations)(
                _AXIS, severity=[0.0], success=[1], direction="decreasing"
            )

    def test_from_observations_without_direction_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            cast(Any, Observations.from_observations)(
                _AXIS, severity=[0.0], success=[1], outcome="success"
            )


class TestDirectionAndOutcome:
    def test_direction_other_than_decreasing_or_increasing_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Observations.from_counts(
                _AXIS,
                severity=[0.0],
                successes=[1],
                trials=[1],
                outcome="success",
                direction="sideways",
            )

    def test_increasing_direction_is_accepted(self) -> None:
        obs = Observations.from_counts(
            _AXIS,
            severity=[0.0],
            successes=[1],
            trials=[1],
            outcome="success",
            direction="increasing",
        )

        assert obs.direction == "increasing"

    def test_empty_outcome_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Observations.from_counts(
                _AXIS,
                severity=[0.0],
                successes=[1],
                trials=[1],
                outcome="",
                direction="decreasing",
            )


class TestFromCountsStoredFields:
    def test_fields_are_tuples_of_the_documented_element_types(self) -> None:
        obs = Observations.from_counts(
            _AXIS,
            severity=[0.0, 0.01, 0.05],
            successes=[200, 299, 122],
            trials=[200, 300, 300],
            outcome="success",
            direction="decreasing",
        )

        assert obs.severity == (0.0, 0.01, 0.05)
        assert obs.successes == (200, 299, 122)
        assert obs.trials == (200, 300, 300)
        assert all(type(v) is float for v in obs.severity)
        assert all(type(v) is int for v in obs.successes)
        assert all(type(v) is int for v in obs.trials)
        assert obs.cluster is None

    def test_whole_number_float_counts_are_accepted_and_stored_as_int(self) -> None:
        obs = Observations.from_counts(
            _AXIS,
            severity=[0.0],
            successes=[200.0],
            trials=[300.0],
            outcome="success",
            direction="decreasing",
        )

        assert obs.successes == (200,)
        assert type(obs.successes[0]) is int
        assert obs.trials == (300,)
        assert type(obs.trials[0]) is int

    def test_non_whole_number_count_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Observations.from_counts(
                _AXIS,
                severity=[0.0],
                successes=[2.5],
                trials=[10],
                outcome="success",
                direction="decreasing",
            )


class TestFromObservationsStoredFields:
    def test_each_row_is_stored_with_trials_equal_to_one(self) -> None:
        obs = Observations.from_observations(
            _AXIS,
            severity=[0.0, 0.0, 1.0],
            success=[1, 0, 1],
            outcome="success",
            direction="decreasing",
        )

        assert obs.trials == (1, 1, 1)
        assert obs.successes == (1, 0, 1)

    @pytest.mark.parametrize("value", [2, -1, 0.5])
    def test_success_value_outside_zero_one_true_false_raises_value_error(
        self, value: float
    ) -> None:
        with pytest.raises(ValueError):
            Observations.from_observations(
                _AXIS,
                severity=[0.0],
                success=[value],
                outcome="success",
                direction="decreasing",
            )

    def test_boolean_success_values_are_accepted_and_stored_as_zero_or_one(self) -> None:
        obs = Observations.from_observations(
            _AXIS,
            severity=[0.0, 1.0],
            success=[True, False],
            outcome="success",
            direction="decreasing",
        )

        assert obs.successes == (1, 0)


class TestValidationErrors:
    """Every failure path ``Observations`` documents, one behaviour per test."""

    def test_empty_input_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Observations.from_counts(
                _AXIS,
                severity=[],
                successes=[],
                trials=[],
                outcome="success",
                direction="decreasing",
            )

    def test_mismatched_lengths_raise_value_error(self) -> None:
        with pytest.raises(ValueError):
            Observations.from_counts(
                _AXIS,
                severity=[0.0, 1.0],
                successes=[1],
                trials=[1, 1],
                outcome="success",
                direction="decreasing",
            )

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_severity_raises_value_error(self, value: float) -> None:
        with pytest.raises(ValueError):
            Observations.from_counts(
                _AXIS,
                severity=[0.0, value],
                successes=[1, 1],
                trials=[1, 1],
                outcome="success",
                direction="decreasing",
            )

    def test_negative_severity_raises_value_error(self) -> None:
        """Signed axes are split by the caller, so ``Observations`` itself never accepts a
        negative severity."""
        with pytest.raises(ValueError):
            Observations.from_counts(
                _AXIS,
                severity=[-1.0],
                successes=[1],
                trials=[1],
                outcome="success",
                direction="decreasing",
            )

    def test_successes_greater_than_trials_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Observations.from_counts(
                _AXIS,
                severity=[0.0],
                successes=[5],
                trials=[4],
                outcome="success",
                direction="decreasing",
            )

    def test_negative_successes_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Observations.from_counts(
                _AXIS,
                severity=[0.0],
                successes=[-1],
                trials=[4],
                outcome="success",
                direction="decreasing",
            )

    def test_trials_less_than_one_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Observations.from_counts(
                _AXIS,
                severity=[0.0],
                successes=[0],
                trials=[0],
                outcome="success",
                direction="decreasing",
            )

    def test_cluster_length_mismatch_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Observations.from_counts(
                _AXIS,
                severity=[0.0, 1.0],
                successes=[1, 1],
                trials=[1, 1],
                outcome="success",
                direction="decreasing",
                cluster=["a"],
            )

    def test_cluster_id_none_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Observations.from_counts(
                _AXIS,
                severity=[0.0, 1.0],
                successes=[1, 1],
                trials=[1, 1],
                outcome="success",
                direction="decreasing",
                cluster=["a", None],
            )

    def test_cluster_id_bool_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Observations.from_counts(
                _AXIS,
                severity=[0.0, 1.0],
                successes=[1, 1],
                trials=[1, 1],
                outcome="success",
                direction="decreasing",
                cluster=["a", True],
            )

    def test_integer_cluster_ids_are_accepted(self) -> None:
        obs = Observations.from_counts(
            _AXIS,
            severity=[0.0, 1.0],
            successes=[1, 1],
            trials=[1, 1],
            outcome="success",
            direction="decreasing",
            cluster=[1, 2],
        )

        assert obs.cluster == (1, 2)


class TestCells:
    def test_cells_are_sorted_by_ascending_severity(self) -> None:
        obs = Observations.from_counts(
            _AXIS,
            severity=[0.1, 0.0, 0.05],
            successes=[0, 200, 122],
            trials=[300, 200, 300],
            outcome="success",
            direction="decreasing",
        )

        assert [cell.severity for cell in obs.cells()] == [0.0, 0.05, 0.1]

    def test_rows_at_the_same_severity_are_pooled_by_summing_counts(self) -> None:
        obs = Observations.from_counts(
            _AXIS,
            severity=[0.0, 0.0],
            successes=[50, 40],
            trials=[100, 90],
            outcome="success",
            direction="decreasing",
        )

        cells = obs.cells()

        assert len(cells) == 1
        assert cells[0].successes == 90
        assert cells[0].trials == 190

    def test_cells_returns_a_tuple_of_cell(self) -> None:
        obs = Observations.from_counts(
            _AXIS,
            severity=[0.0],
            successes=[1],
            trials=[1],
            outcome="success",
            direction="decreasing",
        )

        cells = obs.cells()

        assert isinstance(cells, tuple)
        assert all(isinstance(cell, Cell) for cell in cells)


class TestIsClustered:
    def test_false_without_cluster_ids(self) -> None:
        obs = Observations.from_counts(
            _AXIS,
            severity=[0.0],
            successes=[1],
            trials=[1],
            outcome="success",
            direction="decreasing",
        )

        assert obs.is_clustered is False

    def test_true_with_cluster_ids(self) -> None:
        obs = Observations.from_counts(
            _AXIS,
            severity=[0.0, 1.0],
            successes=[1, 1],
            trials=[1, 1],
            outcome="success",
            direction="decreasing",
            cluster=["a", "b"],
        )

        assert obs.is_clustered is True


class TestObservationsIsFrozen:
    def test_is_frozen(self) -> None:
        obs = Observations.from_counts(
            _AXIS,
            severity=[0.0],
            successes=[1],
            trials=[1],
            outcome="success",
            direction="decreasing",
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            cast(Any, obs).outcome = "other"


class TestObservationsDirectConstructionTypeStrictness:
    """The same whole-number/bool strictness ``from_counts`` already enforces via
    ``_as_whole_int`` must also hold for direct ``Observations(...)`` construction, which
    bypasses that classmethod and its conversion entirely."""

    def test_bool_successes_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Observations(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                severity=(0.0,),
                successes=(True,),
                trials=(1,),
            )

    def test_non_whole_trials_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Observations(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                severity=(0.0,),
                successes=(1,),
                trials=(2.5,),
            )

    def test_bool_severity_raises_value_error(self) -> None:
        """``Observations.severity`` rejects a bool, including through
        direct construction -- ``from_counts`` cannot exercise this case at all, since its own
        ``float(v)`` conversion already turns a bool into a plain ``1.0``/``0.0`` before
        ``__post_init__`` ever sees it."""
        with pytest.raises(ValueError):
            Observations(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                severity=(True,),
                successes=(1,),
                trials=(1,),
            )

    def test_whole_number_float_counts_are_accepted_and_stored_as_int(self) -> None:
        """The same whole-number-float acceptance ``Cell`` gets also holds for direct
        ``Observations`` construction."""
        obs = Observations(
            axis=_AXIS,
            outcome="success",
            direction="decreasing",
            severity=(0.0,),
            successes=(200.0,),
            trials=(200.0,),
        )

        assert obs.successes == (200,)
        assert type(obs.successes[0]) is int
        assert obs.trials == (200,)
        assert type(obs.trials[0]) is int
