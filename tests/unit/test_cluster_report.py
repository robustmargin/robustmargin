"""Serialisation of the clustered and paired fields: round trips, the v1/v2 reader and writer
rules, and schema validation of the new JSON.

Field names are API: a rename must break a test here. Schema ``"3"`` is unreleased, so the new
fields extend it; a ``"1"`` or ``"2"`` payload carrying any of them is rejected, while a card in
the shape the previous release wrote still reads and is written back unchanged.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any

import jsonschema
import pytest
from _cluster_fixtures import clustered_fit, clustered_threshold, paired_fits

from robustmargin import Fit, Ratio, Scorecard, ratio_interval
from robustmargin.report import from_dict, load_schema, to_dict
from robustmargin.testing import fake_fit, fake_threshold


def _json(obj: object) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(json.dumps(to_dict(obj), allow_nan=False))
    return payload


def _relabel(obj: Any, version: str) -> Any:
    if isinstance(obj, dict):
        return {
            k: (version if k == "schema_version" else _relabel(v, version)) for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_relabel(item, version) for item in obj]
    return obj


def _validator(type_name: str, version: str = "3") -> jsonschema.protocols.Validator:
    schema = load_schema(version)
    return jsonschema.Draft202012Validator(
        {"$defs": schema["$defs"], "$ref": f"#/$defs/{type_name}"}
    )


def _demote_fit(payload: dict[str, Any]) -> None:
    """Rewrite a clustered Fit payload, in place, to what an independent fit of the same data
    carries, so that a test can leave exactly one clustered field behind."""
    payload["dependence"] = "independent"
    payload["cluster_cells"] = None
    payload["monotonicity"] = _json(clustered_fit(dependence="independent"))["monotonicity"]


def _demote_threshold(payload: dict[str, Any]) -> None:
    payload["dependence"] = "independent"
    payload["interval_method"] = "delta"
    payload["bootstrap"] = None
    _demote_fit(payload["fit"])


def _paired_ratio(method: str = "fieller") -> Ratio:
    fit_a, fit_b = paired_fits()
    extra: dict[str, Any] = {"n_resamples": 19, "seed": 3} if method == "cluster_bootstrap" else {}
    return ratio_interval(
        clustered_threshold(fit_a),
        clustered_threshold(fit_b),
        method=method,
        dependence="paired",
        **extra,
    )


def _round_trip(obj: Any) -> Any:
    return from_dict(json.loads(json.dumps(to_dict(obj), allow_nan=False)))


_CLUSTER_CELL_KEYS = {"severity", "cluster", "successes", "trials"}
_PAIR_KEYS = {
    "severity_low",
    "severity_high",
    "clusters_reversed",
    "clusters_ordered",
    "clusters_tied",
    "clusters_unbalanced",
    "p_value",
    "holm_adjusted_p",
    "flagged",
}


class TestRoundTrip:
    def test_clustered_fit_round_trips_with_dependence_and_cluster_cells(self) -> None:
        fit = clustered_fit()

        back = _round_trip(fit)

        assert back == fit
        assert back.dependence == "clustered"
        assert back.cluster_cells == fit.cluster_cells
        assert back.monotonicity == fit.monotonicity

    def test_independent_fit_of_clustered_data_round_trips_without_cluster_cells(self) -> None:
        fit = clustered_fit(dependence="independent")

        back = _round_trip(fit)

        assert back == fit
        assert back.dependence == "independent" and back.cluster_cells is None

    @pytest.mark.parametrize("method", ["delta", "cluster_bootstrap"])
    def test_clustered_threshold_round_trips_with_method_level_and_bootstrap(
        self, method: str
    ) -> None:
        extra: dict[str, Any] = (
            {"n_resamples": 19, "seed": 7} if method == "cluster_bootstrap" else {}
        )
        t = clustered_threshold(interval_method=method, **extra)

        back = _round_trip(t)

        assert back == t
        assert back.dependence == "clustered"
        assert back.interval_method == method
        assert back.level == t.level
        assert back.bootstrap == t.bootstrap

    @pytest.mark.parametrize("method", ["fieller", "log_delta", "cluster_bootstrap"])
    def test_paired_ratio_round_trips_with_method_level_dependence_and_bootstrap(
        self, method: str
    ) -> None:
        ratio = _paired_ratio(method)

        back = _round_trip(ratio)

        assert back == ratio
        assert back.dependence == "paired"
        assert back.method == method
        assert back.level == ratio.level
        assert back.bootstrap == ratio.bootstrap
        assert back.warnings == ratio.warnings

    def test_scorecard_of_clustered_results_round_trips(self) -> None:
        fit_a, fit_b = paired_fits()
        card = Scorecard(results=(fit_a, _paired_ratio("log_delta")), provenance={"run": "r1"})

        assert _round_trip(card) == card


class TestFieldNamesAreApi:
    def test_a_clustered_fit_writes_dependence_and_cluster_cells_keys(self) -> None:
        payload = _json(clustered_fit())

        assert payload["dependence"] == "clustered"
        assert {tuple(sorted(c)) for c in payload["cluster_cells"]} == {
            tuple(sorted(_CLUSTER_CELL_KEYS))
        }
        assert payload["monotonicity"]["dependence"] == "clustered"
        assert payload["monotonicity"]["method"] == "cluster_sign_test_one_sided_holm"

    def test_cluster_cells_values_survive_as_counted(self) -> None:
        fit = clustered_fit()
        assert fit.cluster_cells is not None

        written = _json(fit)["cluster_cells"]

        assert [(c["severity"], c["cluster"], c["successes"], c["trials"]) for c in written] == [
            (c.severity, c.cluster, c.successes, c.trials) for c in fit.cluster_cells
        ]

    def test_a_cluster_sign_test_pair_writes_exactly_these_keys(self) -> None:
        payload = _json(clustered_fit())

        for pair in payload["monotonicity"]["pairs"]:
            assert set(pair) == _PAIR_KEYS

    def test_a_current_independent_fit_writes_dependence_and_a_null_cluster_cells(self) -> None:
        payload = _json(fake_fit())

        assert payload["dependence"] == "independent"
        assert payload["cluster_cells"] is None

    def test_a_ratio_writes_a_bootstrap_key_null_unless_the_bootstrap_ran(self) -> None:
        assert _json(_paired_ratio("fieller"))["bootstrap"] is None
        boot = _json(_paired_ratio("cluster_bootstrap"))["bootstrap"]

        assert set(boot) == {"n_resamples", "failed", "unreachable_low", "unreachable_high", "seed"}

    def test_a_ratio_writes_paired_as_its_dependence(self) -> None:
        assert _json(_paired_ratio("log_delta"))["dependence"] == "paired"


class TestVersionOneAndTwoWriter:
    def test_a_version_2_fit_leaves_out_an_independent_dependence_and_null_cluster_cells(
        self,
    ) -> None:
        fit = dataclasses.replace(clustered_fit(dependence="independent"), schema_version="2")

        payload = to_dict(fit)

        assert "dependence" not in payload
        assert "cluster_cells" not in payload
        assert payload["cluster_ids"] is not None

    def test_a_version_1_fit_leaves_them_out_too(self) -> None:
        fit = dataclasses.replace(fake_fit(), schema_version="1")

        payload = to_dict(fit)

        assert "dependence" not in payload and "cluster_cells" not in payload

    def test_a_version_2_threshold_still_writes_its_required_dependence(self) -> None:
        fit = dataclasses.replace(fake_fit(), schema_version="2")
        t = dataclasses.replace(fake_threshold(), fit=fit, schema_version="2")

        payload: Any = to_dict(t)

        assert payload["dependence"] == "independent"
        assert "dependence" not in payload["fit"]

    def test_a_version_2_monotonicity_check_still_writes_its_required_dependence(self) -> None:
        fit = dataclasses.replace(fake_fit(), schema_version="2")

        payload: Any = to_dict(fit)

        assert payload["monotonicity"]["dependence"] == "independent"

    def test_a_version_3_fit_always_writes_dependence(self) -> None:
        assert to_dict(fake_fit())["dependence"] == "independent"


class TestVersionOneAndTwoReader:
    @staticmethod
    def _previous_release_fit_payload(version: str) -> dict[str, Any]:
        fit = dataclasses.replace(clustered_fit(dependence="independent"), schema_version=version)
        payload = _json(fit)
        assert "dependence" not in payload
        return payload

    @pytest.mark.parametrize("version", ["1", "2"])
    def test_a_previous_release_fit_without_dependence_reads_as_independent(
        self, version: str
    ) -> None:
        payload = self._previous_release_fit_payload(version)

        fit = from_dict(payload)

        assert isinstance(fit, Fit)
        assert fit.dependence == "independent"
        assert fit.cluster_cells is None
        assert fit.cluster_ids is not None
        assert fit.schema_version == version

    @pytest.mark.parametrize("version", ["1", "2"])
    def test_a_previous_release_fit_is_written_back_byte_for_byte(self, version: str) -> None:
        payload = self._previous_release_fit_payload(version)

        fit = from_dict(payload)

        assert json.dumps(to_dict(fit), sort_keys=True) == json.dumps(payload, sort_keys=True)

    def test_explicit_independent_dependence_and_null_cluster_cells_are_accepted_on_a_v2_fit(
        self,
    ) -> None:
        payload = self._previous_release_fit_payload("2")
        payload["dependence"] = "independent"
        payload["cluster_cells"] = None

        fit = from_dict(payload)

        assert isinstance(fit, Fit) and fit.dependence == "independent"

    def test_a_clustered_fit_tagged_2_is_rejected(self) -> None:
        payload = _relabel(_json(clustered_fit()), "2")

        with pytest.raises(ValueError):
            from_dict(payload)

    def test_non_null_cluster_cells_alone_on_a_v2_fit_are_rejected(self) -> None:
        payload = _relabel(_json(clustered_fit()), "2")
        payload["dependence"] = "independent"
        payload["monotonicity"] = _json(clustered_fit(dependence="independent"))["monotonicity"]

        with pytest.raises(ValueError):
            from_dict(payload)

    def test_a_nested_clustered_monotonicity_check_alone_on_a_v2_fit_is_rejected(self) -> None:
        payload = _relabel(_json(clustered_fit()), "2")
        payload["dependence"] = "independent"
        payload["cluster_cells"] = None

        with pytest.raises(ValueError):
            from_dict(payload)

    def test_a_clustered_threshold_tagged_2_is_rejected(self) -> None:
        payload = _relabel(_json(clustered_threshold()), "2")

        with pytest.raises(ValueError):
            from_dict(payload)

    def test_the_cluster_bootstrap_interval_method_tagged_2_is_rejected(self) -> None:
        payload = _relabel(
            _json(clustered_threshold(interval_method="cluster_bootstrap", n_resamples=19, seed=1)),
            "2",
        )
        payload["dependence"] = "independent"
        payload["bootstrap"] = None
        _demote_fit(payload["fit"])

        with pytest.raises(ValueError):
            from_dict(payload)

    @pytest.mark.parametrize("version", ["1", "2"])
    def test_a_paired_ratio_tagged_below_3_is_rejected(self, version: str) -> None:
        payload = _relabel(_json(_paired_ratio("fieller")), version)

        with pytest.raises(ValueError):
            from_dict(payload)

    def test_a_cluster_bootstrap_ratio_tagged_2_is_rejected_for_its_method_alone(self) -> None:
        # The nested thresholds and fits are demoted to independent shapes so that only the
        # ratio's own fields can be the cause.
        payload = _relabel(_json(_paired_ratio("cluster_bootstrap")), "2")
        payload["dependence"] = "independent"
        payload["bootstrap"] = None
        _demote_threshold(payload["threshold_a"])
        _demote_threshold(payload["threshold_b"])

        with pytest.raises(ValueError):
            from_dict(payload)

    def test_a_non_null_ratio_bootstrap_tagged_2_is_rejected(self) -> None:
        payload = _relabel(_json(_paired_ratio("cluster_bootstrap")), "2")
        payload["dependence"] = "independent"
        payload["method"] = "fieller"
        _demote_threshold(payload["threshold_a"])
        _demote_threshold(payload["threshold_b"])

        with pytest.raises(ValueError):
            from_dict(payload)

    def test_an_independent_ratio_tagged_2_still_reads(self) -> None:
        from robustmargin.testing import fake_ratio

        payload = _relabel(_json(fake_ratio()), "2")
        payload.pop("bootstrap", None)
        for side in ("threshold_a", "threshold_b"):
            payload[side].pop("bootstrap", None)
            payload[side]["fit"].pop("dependence", None)
            payload[side]["fit"].pop("cluster_cells", None)
            payload[side]["fit"].pop("cell_estimates", None)

        ratio = from_dict(payload)

        assert isinstance(ratio, Ratio)
        assert ratio.dependence == "independent"


class TestSchemaValidation:
    def test_clustered_fit_validates_against_the_current_schema(self) -> None:
        _validator("Fit").validate(_json(clustered_fit()))

    def test_clustered_threshold_validates_against_the_current_schema(self) -> None:
        _validator("Threshold").validate(_json(clustered_threshold()))
        _validator("Threshold").validate(
            _json(clustered_threshold(interval_method="cluster_bootstrap", n_resamples=19, seed=2))
        )

    @pytest.mark.parametrize("method", ["fieller", "log_delta", "cluster_bootstrap"])
    def test_paired_ratio_validates_against_the_current_schema(self, method: str) -> None:
        _validator("Ratio").validate(_json(_paired_ratio(method)))

    def test_scorecard_of_clustered_results_validates_against_the_root(self) -> None:
        card = Scorecard(results=(clustered_fit(), _paired_ratio("fieller")), provenance={})

        jsonschema.Draft202012Validator(load_schema()).validate(_json(card))

    def test_no_new_key_is_required_on_any_result_type(self) -> None:
        defs: Any = load_schema()["$defs"]

        assert "dependence" not in defs["Fit"]["required"]
        assert "cluster_cells" not in defs["Fit"]["required"]
        assert "bootstrap" not in defs["Ratio"]["required"]

    def test_the_schema_accepts_the_cluster_sign_test_pair_and_nothing_extra(self) -> None:
        payload = _json(clustered_fit())
        _validator("MonotonicityCheck").validate(payload["monotonicity"])
        extra = copy.deepcopy(payload["monotonicity"])
        extra["pairs"][0]["unexpected"] = 1

        with pytest.raises(jsonschema.ValidationError):
            _validator("MonotonicityCheck").validate(extra)

    def test_the_schema_rejects_an_unknown_fit_dependence(self) -> None:
        payload = _json(clustered_fit())
        payload["dependence"] = "paired"

        with pytest.raises(jsonschema.ValidationError):
            _validator("Fit").validate(payload)

    def test_the_schema_rejects_an_unknown_ratio_dependence(self) -> None:
        payload = _json(_paired_ratio("fieller"))
        payload["dependence"] = "clustered"

        with pytest.raises(jsonschema.ValidationError):
            _validator("Ratio").validate(payload)

    def test_the_schema_rejects_a_malformed_cluster_cell(self) -> None:
        payload = _json(clustered_fit())
        payload["cluster_cells"][0].pop("cluster")

        with pytest.raises(jsonschema.ValidationError):
            _validator("Fit").validate(payload)

    def test_the_schema_accepts_the_new_threshold_interval_method(self) -> None:
        payload = _json(
            clustered_threshold(interval_method="cluster_bootstrap", n_resamples=19, seed=2)
        )

        assert payload["interval_method"] == "cluster_bootstrap"
        _validator("Threshold").validate(payload)

    def test_version_2_schema_does_not_admit_the_new_interval_method(self) -> None:
        defs: Any = load_schema("2")["$defs"]
        enum = defs["Threshold"]["properties"]["interval_method"]["enum"]

        assert "cluster_bootstrap" not in enum
