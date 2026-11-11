"""``ContinuousObservations``: constructor validation and the pooling of rows into cells.

A continuous series is one real-valued response per observation (``from_observations``) or already
aggregated cells of ``(mean, sd, n)`` (``from_summaries``, with ``sd`` the sample SD, ``ddof=1``).
Either way ``cells()`` returns one :class:`ContinuousCell` per distinct severity, ascending.
Rows at one severity are pooled with the parallel update of Chan, Golub and LeVeque (1983), which
stays accurate where the textbook sum-of-squares formula loses every digit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robustmargin import Axis, ContinuousCell, ContinuousObservations

_AXIS = Axis(name="severity", unit="unit", scale="log")


def _values(**overrides: object) -> ContinuousObservations:
    kwargs: dict[str, object] = {
        "severity": [0.0, 1.0, 1.0, 2.0],
        "value": [5.0, 4.0, 4.5, 2.0],
        "outcome": "response",
        "direction": "decreasing",
    }
    kwargs.update(overrides)
    return ContinuousObservations.from_observations(_AXIS, **kwargs)  # type: ignore[arg-type]


def _summaries(**overrides: object) -> ContinuousObservations:
    kwargs: dict[str, object] = {
        "severity": [0.0, 1.0, 2.0],
        "mean": [5.0, 4.25, 2.0],
        "sd": [0.5, 0.25, 0.0],
        "n": [3, 2, 1],
        "outcome": "response",
        "direction": "decreasing",
    }
    kwargs.update(overrides)
    return ContinuousObservations.from_summaries(_AXIS, **kwargs)  # type: ignore[arg-type]


class TestCells:
    def test_rows_at_one_severity_pool_into_one_cell_sorted_ascending(self) -> None:
        obs = _values(severity=[2.0, 1.0, 0.0, 1.0], value=[2.0, 4.0, 5.0, 4.5])

        cells = obs.cells()

        assert [c.severity for c in cells] == [0.0, 1.0, 2.0]
        assert all(isinstance(c, ContinuousCell) for c in cells)
        assert [c.n for c in cells] == [1, 2, 1]

    def test_a_pooled_cell_has_the_mean_and_sample_sd_of_its_rows(self) -> None:
        rng = np.random.default_rng(3)
        sample = rng.normal(7.0, 2.0, size=9)

        (cell,) = _values(severity=[1.0] * 9, value=sample.tolist()).cells()

        assert cell.n == 9
        assert cell.mean == pytest.approx(sample.mean(), rel=1e-12)
        assert cell.sd == pytest.approx(sample.std(ddof=1), rel=1e-12)

    def test_a_single_row_cell_has_zero_sd(self) -> None:
        (cell,) = _values(severity=[1.0], value=[3.0]).cells()

        assert (cell.n, cell.mean, cell.sd) == (1, 3.0, 0.0)

    def test_pooling_is_accurate_when_the_mean_dwarfs_the_spread(self) -> None:
        # The sum-of-squares shortcut loses all of the spread at a mean of 1e9.
        sample = 1.0e9 + np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        (cell,) = _values(severity=[1.0] * 6, value=sample.tolist()).cells()

        assert cell.sd == pytest.approx(
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).std(ddof=1), rel=1e-6
        )

    def test_summaries_given_as_separate_rows_at_one_severity_pool_like_the_raw_values(
        self,
    ) -> None:
        a, b = [1.0, 2.0, 4.0], [3.0, 3.5, 8.0, 9.0, 10.0]
        pooled = ContinuousObservations.from_summaries(
            _AXIS,
            severity=[1.0, 1.0],
            mean=[np.mean(a), np.mean(b)],
            sd=[np.std(a, ddof=1), np.std(b, ddof=1)],
            n=[3, 5],
            outcome="response",
            direction="decreasing",
        )

        (cell,) = pooled.cells()
        both = np.array(a + b)

        assert cell.n == 8
        assert cell.mean == pytest.approx(both.mean(), rel=1e-12)
        assert cell.sd == pytest.approx(both.std(ddof=1), rel=1e-12)

    def test_values_and_summaries_of_the_same_data_give_the_same_cells(self) -> None:
        from_rows = _values(severity=[1.0, 1.0, 1.0], value=[2.0, 3.0, 7.0])
        from_cells = _summaries(severity=[1.0], mean=[4.0], sd=[math.sqrt(7.0)], n=[3])

        (a,), (b,) = from_rows.cells(), from_cells.cells()

        assert (a.severity, a.n) == (b.severity, b.n)
        assert a.mean == pytest.approx(b.mean, rel=1e-12)
        assert a.sd == pytest.approx(b.sd, rel=1e-12)

    def test_cells_accept_whole_number_float_counts(self) -> None:
        (cell,) = _summaries(severity=[1.0], mean=[4.0], sd=[1.0], n=[3.0]).cells()

        assert cell.n == 3 and isinstance(cell.n, int)


class TestFromValuesValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"severity": [], "value": []},
            {"severity": [0.0, 1.0], "value": [1.0]},
            {"severity": [-1.0, 1.0, 1.0, 2.0]},
            {"severity": [float("nan"), 1.0, 1.0, 2.0]},
            {"severity": [float("inf"), 1.0, 1.0, 2.0]},
            {"severity": [True, 1.0, 1.0, 2.0]},
            {"value": [float("nan"), 4.0, 4.5, 2.0]},
            {"value": [float("inf"), 4.0, 4.5, 2.0]},
            {"value": [True, 4.0, 4.5, 2.0]},
            {"direction": "sideways"},
            {"outcome": ""},
            {"cluster": ["a", "b"]},
            {"cluster": [None, "b", "c", "d"]},
            {"cluster": [True, 1, 2, 3]},
        ],
        ids=[
            "empty",
            "length_mismatch",
            "negative_severity",
            "nan_severity",
            "inf_severity",
            "bool_severity",
            "nan_value",
            "inf_value",
            "bool_value",
            "bad_direction",
            "empty_outcome",
            "cluster_length",
            "cluster_none",
            "cluster_bool",
        ],
    )
    def test_invalid_input_raises(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            _values(**overrides)

    def test_negative_and_zero_values_are_valid_responses(self) -> None:
        obs = _values(value=[-5.0, 0.0, -4.5, -2.0])

        assert [c.mean for c in obs.cells()] == pytest.approx([-5.0, -2.25, -2.0])

    @pytest.mark.parametrize("direction", ["decreasing", "increasing"])
    def test_both_directions_are_accepted_and_recorded(self, direction: str) -> None:
        obs = _values(direction=direction)

        assert obs.direction == direction
        assert obs.outcome == "response"
        assert obs.axis == _AXIS

    def test_cluster_ids_are_kept(self) -> None:
        obs = _values(cluster=["a", "a", "b", "b"])

        assert obs.is_clustered is True
        assert _values().is_clustered is False

    def test_outcome_and_direction_have_no_default(self) -> None:
        with pytest.raises(TypeError):
            ContinuousObservations.from_observations(  # type: ignore[call-arg]
                _AXIS, severity=[1.0], value=[1.0], direction="decreasing"
            )
        with pytest.raises(TypeError):
            ContinuousObservations.from_observations(  # type: ignore[call-arg]
                _AXIS, severity=[1.0], value=[1.0], outcome="response"
            )


class TestFromSummariesValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"severity": []},
            {"mean": [5.0, 4.0]},
            {"sd": [0.5, 0.25]},
            {"n": [3, 2]},
            {"severity": [-1.0, 1.0, 2.0]},
            {"mean": [float("nan"), 4.0, 2.0]},
            {"sd": [-0.5, 0.25, 0.0]},
            {"sd": [float("nan"), 0.25, 0.0]},
            {"sd": [float("inf"), 0.25, 0.0]},
            {"n": [0, 2, 1]},
            {"n": [3, 2.5, 1]},
            {"n": [True, 2, 1]},
            {"n": [3, 2, 1], "sd": [0.5, 0.25, 0.1]},
            {"direction": "sideways"},
            {"outcome": ""},
        ],
        ids=[
            "empty",
            "mean_length",
            "sd_length",
            "n_length",
            "negative_severity",
            "nan_mean",
            "negative_sd",
            "nan_sd",
            "inf_sd",
            "zero_n",
            "fractional_n",
            "bool_n",
            "one_observation_with_spread",
            "bad_direction",
            "empty_outcome",
        ],
    )
    def test_invalid_input_raises(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            _summaries(**overrides)

    def test_a_single_observation_cell_needs_an_sd_of_exactly_zero(self) -> None:
        (cell,) = _summaries(severity=[1.0], mean=[2.0], sd=[0.0], n=[1]).cells()

        assert cell.sd == 0.0

    def test_a_cell_of_several_observations_may_have_zero_sd(self) -> None:
        (cell,) = _summaries(severity=[1.0], mean=[2.0], sd=[0.0], n=[4]).cells()

        assert (cell.n, cell.sd) == (4, 0.0)
