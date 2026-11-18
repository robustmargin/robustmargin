"""``ClusterCell``, ``ClusterAdjacentPair``, ``Observations.cluster_cells()`` and the new exports.

Names are imported inside each test so that a missing name fails only the tests that use it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from _cluster_fixtures import AXIS, clustered_counts

from robustmargin import Observations


def _cell(**overrides: Any) -> Any:
    from robustmargin import ClusterCell

    kwargs: dict[str, Any] = {"severity": 1.0, "cluster": "c0", "successes": 3, "trials": 10}
    kwargs.update(overrides)
    return ClusterCell(**kwargs)


def _pair(**overrides: Any) -> Any:
    from robustmargin import ClusterAdjacentPair

    kwargs: dict[str, Any] = {
        "severity_low": 1.0,
        "severity_high": 2.0,
        "clusters_reversed": 1,
        "clusters_ordered": 4,
        "clusters_tied": 1,
        "clusters_unbalanced": 0,
        "p_value": 0.5,
        "holm_adjusted_p": 0.5,
        "flagged": False,
    }
    kwargs.update(overrides)
    return ClusterAdjacentPair(**kwargs)


class TestExports:
    @pytest.mark.parametrize("name", ["ClusterCell", "ClusterAdjacentPair"])
    def test_new_type_is_exported_in_all_and_importable(self, name: str) -> None:
        import robustmargin

        assert name in robustmargin.__all__
        assert getattr(robustmargin, name) is not None


class TestClusterCell:
    def test_valid_cell_stores_its_fields(self) -> None:
        cell = _cell()

        assert (cell.severity, cell.cluster, cell.successes, cell.trials) == (1.0, "c0", 3, 10)

    @pytest.mark.parametrize("cluster", ["scenario", 7])
    def test_cluster_id_may_be_a_str_or_an_int(self, cluster: str | int) -> None:
        assert _cell(cluster=cluster).cluster == cluster

    @pytest.mark.parametrize("cluster", [None, True, 1.5, b"x", ("a",)])
    def test_cluster_id_other_than_str_or_int_raises(self, cluster: object) -> None:
        with pytest.raises(ValueError):
            _cell(cluster=cluster)

    def test_whole_number_floats_are_stored_as_genuine_ints(self) -> None:
        cell = _cell(successes=3.0, trials=10.0)

        assert type(cell.successes) is int and type(cell.trials) is int

    @pytest.mark.parametrize(
        "overrides",
        [
            {"successes": 2.5},
            {"trials": 10.5},
            {"successes": True},
            {"trials": 0},
            {"successes": -1},
            {"successes": 11, "trials": 10},
            {"severity": -0.1},
            {"severity": float("nan")},
            {"severity": float("inf")},
        ],
    )
    def test_invalid_counts_or_severity_raise_as_a_cell_would(
        self, overrides: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError):
            _cell(**overrides)

    def test_cell_is_frozen(self) -> None:
        cell = _cell()

        with pytest.raises(dataclasses.FrozenInstanceError):
            cell.successes = 4

    def test_fields_are_keyword_only(self) -> None:
        from robustmargin import ClusterCell

        with pytest.raises(TypeError):
            ClusterCell(1.0, "c0", 3, 10)  # type: ignore[call-arg]


class TestClusterAdjacentPair:
    def test_valid_pair_stores_its_fields(self) -> None:
        pair = _pair()

        assert pair.clusters_reversed == 1
        assert pair.clusters_ordered == 4
        assert pair.clusters_tied == 1
        assert pair.clusters_unbalanced == 0
        assert pair.flagged is False

    def test_unbalanced_count_is_a_required_field(self) -> None:
        from robustmargin import ClusterAdjacentPair

        with pytest.raises(TypeError):
            ClusterAdjacentPair(  # type: ignore[call-arg]
                severity_low=1.0,
                severity_high=2.0,
                clusters_reversed=1,
                clusters_ordered=4,
                clusters_tied=1,
                p_value=0.5,
                holm_adjusted_p=0.5,
                flagged=False,
            )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"severity_low": 2.0, "severity_high": 2.0},
            {"severity_low": 3.0, "severity_high": 2.0},
            {"severity_low": float("nan")},
            {"clusters_reversed": -1},
            {"clusters_ordered": -1},
            {"clusters_tied": -1},
            {"clusters_unbalanced": -1},
            {"clusters_reversed": 1.5},
            {"clusters_ordered": True},
            {"clusters_unbalanced": 0.5},
            {"p_value": -0.1},
            {"p_value": 1.1},
            {"p_value": float("nan")},
            {"holm_adjusted_p": -0.1},
            {"holm_adjusted_p": 1.1},
            {"flagged": 1},
        ],
    )
    def test_invalid_fields_raise(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            _pair(**overrides)

    def test_pair_is_frozen(self) -> None:
        pair = _pair()

        with pytest.raises(dataclasses.FrozenInstanceError):
            pair.flagged = True


class TestObservationsClusterCells:
    def test_unclustered_observations_return_none(self) -> None:
        obs = Observations.from_counts(
            AXIS,
            severity=[0.0, 1.0],
            successes=[10, 4],
            trials=[10, 10],
            outcome="success",
            direction="decreasing",
        )

        assert obs.cluster_cells() is None

    def test_counts_are_pooled_per_severity_and_cluster(self) -> None:
        obs = Observations.from_observations(
            AXIS,
            severity=[1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
            success=[1, 0, 1, 1, 0, 1],
            outcome="success",
            direction="decreasing",
            cluster=["x", "x", "y", "y", "x", "y"],
        )

        cells = obs.cluster_cells()

        assert cells is not None
        got = [(c.severity, c.cluster, c.successes, c.trials) for c in cells]
        assert got == [(1.0, "x", 1, 2), (1.0, "y", 2, 2), (2.0, "x", 0, 1), (2.0, "y", 1, 1)]

    def test_order_is_severity_ascending_then_first_seen_cluster_order(self) -> None:
        obs = Observations.from_counts(
            AXIS,
            severity=[2.0, 2.0, 1.0, 1.0],
            successes=[1, 2, 3, 4],
            trials=[5, 5, 5, 5],
            outcome="success",
            direction="decreasing",
            cluster=["zeta", "alpha", "alpha", "zeta"],
        )

        cells = obs.cluster_cells()

        assert cells is not None
        # First seen in the data is "zeta", so it leads at every level, even though "alpha" sorts
        # first alphabetically and 2.0 appears first in the rows.
        assert [(c.severity, c.cluster) for c in cells] == [
            (1.0, "zeta"),
            (1.0, "alpha"),
            (2.0, "zeta"),
            (2.0, "alpha"),
        ]

    def test_a_cluster_absent_at_a_level_has_no_entry_there(self) -> None:
        obs = Observations.from_counts(
            AXIS,
            severity=[1.0, 1.0, 2.0],
            successes=[1, 1, 1],
            trials=[2, 2, 2],
            outcome="success",
            direction="decreasing",
            cluster=["x", "y", "x"],
        )

        cells = obs.cluster_cells()

        assert cells is not None
        assert [(c.severity, c.cluster) for c in cells] == [(1.0, "x"), (1.0, "y"), (2.0, "x")]

    def test_per_severity_sums_equal_cells(self) -> None:
        obs = clustered_counts(5)
        cluster_cells = obs.cluster_cells()
        assert cluster_cells is not None

        for cell in obs.cells():
            at_level = [c for c in cluster_cells if c.severity == cell.severity]
            assert sum(c.successes for c in at_level) == cell.successes
            assert sum(c.trials for c in at_level) == cell.trials

    def test_aggregated_and_per_trial_input_give_the_same_cluster_cells(self) -> None:
        aggregated = Observations.from_counts(
            AXIS,
            severity=[1.0, 1.0, 2.0, 2.0],
            successes=[2, 1, 0, 2],
            trials=[3, 3, 3, 3],
            outcome="success",
            direction="decreasing",
            cluster=["x", "y", "x", "y"],
        )
        per_trial = Observations.from_observations(
            AXIS,
            severity=[1.0] * 6 + [2.0] * 6,
            success=[1, 1, 0, 1, 0, 0] + [0, 0, 0, 1, 1, 0],
            outcome="success",
            direction="decreasing",
            cluster=["x", "x", "x", "y", "y", "y"] * 2,
        )

        assert aggregated.cluster_cells() == per_trial.cluster_cells()

    def test_cluster_order_matches_fit_cluster_ids(self) -> None:
        from robustmargin import fit_dose_response

        obs = clustered_counts(4)
        fit = fit_dose_response(
            obs,
            model="binomial",
            link="probit",
            upper=1.0,
            lower=0.0,
            dependence="independent",
        )
        cells = obs.cluster_cells()
        assert cells is not None and fit.cluster_ids is not None

        first_level = [c.cluster for c in cells if c.severity == cells[0].severity]
        assert tuple(first_level) == fit.cluster_ids
