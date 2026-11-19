"""Score-card schema ``"3"``: what it adds to ``"2"``, and what reading older cards still means.

Schema ``"3"`` is ``"2"`` plus exactly these edits, and nothing else:

* ``Fit.model`` accepts ``"isotonic"``;
* ``Fit.link`` accepts ``"none"``;
* ``Threshold.interval_method`` accepts ``"parametric_bootstrap"`` and ``"cell_bootstrap"``;
* ``Fit.cell_estimates`` (null or an array of numbers) is a new property;
* ``Threshold.bootstrap`` (null or an object with the five integer fields, no others) is a new
  property;
* ``Fit.dependence`` (``"independent"`` or ``"clustered"``) and ``Fit.cluster_cells`` (null or an
  array of ``ClusterCell``) are new properties, and ``ClusterCell`` and ``ClusterAdjacentPair``
  are new definitions;
* ``Threshold.interval_method`` also accepts ``"cluster_bootstrap"``, and ``Threshold.dependence``,
  ``Ratio.dependence`` and ``MonotonicityCheck.dependence`` accept ``"clustered"`` or
  ``"paired"`` as applicable;
* ``Ratio.method`` accepts ``"cluster_bootstrap"`` and ``Ratio.bootstrap`` (null or a
  ``BootstrapSummary``) is a new property;
* ``MonotonicityCheck.method`` accepts ``"cluster_sign_test_one_sided_holm"`` and its pairs may be
  ``ClusterAdjacentPair``;
* the ``Fit`` conditionals: an isotonic fit with status ``OK`` may have null ``params``,
  ``covariance`` and ``diagnostics``, and ``cell_estimates`` is non-null if and only if the model
  is isotonic.

``scorecard-v1.json`` and ``scorecard-v2.json`` stay byte-for-byte what they were. ``from_dict``
reads all three; a payload tagged ``"1"`` or ``"2"`` that carries a value only ``"3"`` has is
rejected, because such a writer could never have produced it. A card written by an earlier
release (none of the keys above) still reads and is written back with
the tag it had, byte for byte (``test_schema_read_keeps_tag.py``). A result constructed now writes
``"3"``.

``test_schema_v2.py`` pins what stays of ``"2"``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from typing import Any

import jsonschema
import pytest

from robustmargin import (
    Axis,
    BootstrapSummary,
    Censoring,
    ContinuousObservations,
    Definition,
    Fit,
    Observations,
    Scorecard,
    Status,
    Threshold,
    fit_dose_response,
    threshold,
)
from robustmargin import report as report_module
from robustmargin.report import SCHEMA_VERSION, from_dict, load_schema, to_dict
from robustmargin.testing import fake_fit, fake_ratio, fake_threshold

_LEVEL = 0.95
_LOG = Axis(name="severity", unit="unit", scale="log")


def _strip(node: Any, *, root: bool = True) -> Any:
    """The validation content of a schema: prose and ``$id`` are left out, and so is the order of
    every ``required`` list and of every ``enum`` list."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if root and key == "$id":
                continue
            if key in ("title", "description") and isinstance(value, str):
                continue
            if key in ("required", "enum") and isinstance(value, list):
                out[key] = sorted(value, key=repr)
                continue
            out[key] = _strip(value, root=False)
        return out
    if isinstance(node, list):
        return [_strip(item, root=False) for item in node]
    return node


def _digest(schema: dict[str, Any]) -> str:
    canonical = json.dumps(_strip(schema), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _schema(version: str) -> dict[str, Any]:
    schema: dict[str, Any] = load_schema(version)
    return schema


def _validator(type_name: str, version: str = "3") -> jsonschema.protocols.Validator:
    schema = _schema(version)
    return jsonschema.Draft202012Validator(
        {"$defs": schema["$defs"], "$ref": f"#/$defs/{type_name}"}
    )


def _dump(obj: object) -> dict[str, Any]:
    payload: dict[str, Any] = to_dict(obj)
    return payload


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


def _without_v3_only_keys(node: Any) -> Any:
    """What a ``"2"`` writer emitted: no ``cell_estimates``, ``bootstrap`` or ``cluster_cells``
    key, and no Fit ``dependence``."""
    if isinstance(node, dict):
        return {
            k: _without_v3_only_keys(v)
            for k, v in node.items()
            if k not in ("cell_estimates", "bootstrap", "cluster_cells")
            and not (k == "dependence" and v == "independent" and "cluster_cells" in node)
        }
    if isinstance(node, list):
        return [_without_v3_only_keys(item) for item in node]
    return node


def _versions_in(obj: Any) -> list[Any]:
    found: list[Any] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "schema_version":
                found.append(value)
            else:
                found.extend(_versions_in(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_versions_in(item))
    return found


def _counts_fit() -> Fit:
    obs = Observations.from_counts(
        _LOG,
        severity=[0.0, 1.0, 2.0, 4.0, 8.0],
        successes=[100, 95, 70, 30, 5],
        trials=[100] * 5,
        outcome="outcome",
        direction="decreasing",
    )
    return fit_dose_response(obs, model="isotonic")


def _continuous_fit() -> Fit:
    obs = ContinuousObservations.from_summaries(
        _LOG,
        severity=[0.0, 0.5, 1.0, 2.0, 4.0],
        mean=[10.0, 8.0, 5.0, 2.0, 1.0],
        sd=[0.5] * 5,
        n=[4] * 5,
        outcome="response",
        direction="decreasing",
    )
    return fit_dose_response(obs, model="isotonic")


def _counts_threshold() -> Threshold:
    return threshold(
        _counts_fit(),
        definition=Definition.absolute(0.5),
        interval_method="cell_bootstrap",
        level=_LEVEL,
        n_resamples=49,
        seed=11,
    )


def _continuous_threshold() -> Threshold:
    return threshold(
        _continuous_fit(),
        definition=Definition.absolute(6.5),
        interval_method="cell_bootstrap",
        level=_LEVEL,
        n_resamples=49,
        seed=11,
    )


_ISOTONIC_BUILDERS: list[tuple[str, Callable[[], Any]]] = [
    ("counts_fit", _counts_fit),
    ("continuous_fit", _continuous_fit),
    ("counts_threshold", _counts_threshold),
    ("continuous_threshold", _continuous_threshold),
    ("fake_fit", lambda: fake_fit(model="isotonic")),
    ("fake_threshold", lambda: fake_threshold(model="isotonic")),
]
_ISOTONIC_IDS = [name for name, _ in _ISOTONIC_BUILDERS]


class TestVersionConstants:
    def test_the_current_schema_version_is_3(self) -> None:
        assert SCHEMA_VERSION == "3"

    def test_the_reader_supports_versions_1_2_and_3(self) -> None:
        assert report_module._SUPPORTED_SCHEMA_VERSIONS == frozenset({"1", "2", "3"})

    def test_every_result_defaults_to_schema_version_3(self) -> None:
        assert fake_fit().schema_version == "3"
        assert fake_threshold().schema_version == "3"
        assert fake_ratio().schema_version == "3"
        assert fake_fit(model="isotonic").schema_version == "3"

    def test_to_dict_writes_3_at_every_nesting_level(self) -> None:
        card = Scorecard(
            results=(fake_fit(), fake_threshold(model="isotonic"), fake_ratio()), provenance={}
        )

        versions = _versions_in(_dump(card))

        assert versions
        assert set(versions) == {"3"}

    @pytest.mark.parametrize("version", ["0", "4", "5", ""])
    def test_other_versions_are_rejected_naming_all_three_supported_ones(
        self, version: str
    ) -> None:
        payload = _relabel(_dump(fake_threshold()), version)

        with pytest.raises(ValueError) as excinfo:
            from_dict(payload)

        message = str(excinfo.value)
        assert "unknown schema_version" in message
        assert all(f"'{v}'" in message for v in ("1", "2", "3"))


class TestSchemaFiles:
    def test_the_three_packaged_schemas_exist_and_nothing_else(self) -> None:
        from pathlib import Path

        names = sorted(
            p.name
            for p in Path(report_module.__file__).resolve().parent.glob("schema/scorecard-v*.json")
        )

        assert names == ["scorecard-v1.json", "scorecard-v2.json", "scorecard-v3.json"]

    def test_version_3_is_a_valid_json_schema(self) -> None:
        jsonschema.Draft202012Validator.check_schema(_schema("3"))

    def test_load_schema_defaults_to_version_3(self) -> None:
        assert load_schema() == load_schema("3")
        assert load_schema("2") != load_schema("3")

    def test_the_documents_have_different_ids(self) -> None:
        ids = {_schema(v)["$id"] for v in ("1", "2", "3")}

        assert len(ids) == 3

    def test_version_2_is_unchanged_in_what_it_admits(self) -> None:
        d2 = _schema("2")["$defs"]

        assert d2["Fit"]["properties"]["model"]["enum"] == ["binomial", "gaussian"]
        assert d2["Fit"]["properties"]["link"]["enum"] == ["probit", "logit", "cloglog"]
        assert d2["Threshold"]["properties"]["interval_method"]["enum"] == [
            "profile",
            "delta",
            "exact_bound",
        ]
        assert "cell_estimates" not in d2["Fit"]["properties"]
        assert "bootstrap" not in d2["Threshold"]["properties"]

    def test_the_enum_additions_are_exactly_the_named_ones(self) -> None:
        d2 = _schema("2")["$defs"]
        d3 = _schema("3")["$defs"]

        def added(name: str, *path: str) -> list[str]:
            old = d2[name]
            new = d3[name]
            for step in path:
                old, new = old[step], new[step]
            assert set(old["enum"]) <= set(new["enum"])
            return sorted(set(new["enum"]) - set(old["enum"]))

        assert added("Fit", "properties", "model") == ["isotonic"]
        assert added("Fit", "properties", "link") == ["none"]
        assert sorted(d3["Fit"]["properties"]["dependence"]["enum"]) == ["clustered", "independent"]
        assert added("Threshold", "properties", "interval_method") == [
            "cell_bootstrap",
            "cluster_bootstrap",
            "parametric_bootstrap",
        ]
        assert added("Threshold", "properties", "dependence") == ["clustered"]
        assert added("Ratio", "properties", "method") == ["cluster_bootstrap"]
        assert added("Ratio", "properties", "dependence") == ["paired"]
        assert added("MonotonicityCheck", "properties", "dependence") == ["clustered"]
        assert added("MonotonicityCheck", "properties", "method") == [
            "cluster_sign_test_one_sided_holm"
        ]
        assert sorted(d3["Fit"]["properties"]["model"]["enum"]) == [
            "binomial",
            "gaussian",
            "isotonic",
        ]
        assert sorted(d3["Fit"]["properties"]["link"]["enum"]) == [
            "cloglog",
            "logit",
            "none",
            "probit",
        ]
        assert sorted(d3["Threshold"]["properties"]["interval_method"]["enum"]) == [
            "cell_bootstrap",
            "cluster_bootstrap",
            "delta",
            "exact_bound",
            "parametric_bootstrap",
            "profile",
        ]

    def test_version_3_is_version_2_plus_exactly_the_listed_edits(self) -> None:
        v2 = _schema("2")
        v3 = _schema("3")
        d2, d3 = v2["$defs"], v3["$defs"]

        # Definitions: three are added, none removed.
        assert set(d3) - set(d2) == {"BootstrapSummary", "ClusterCell", "ClusterAdjacentPair"}
        assert set(d2) - set(d3) == set()
        # No `required` list of an existing definition changes: every new property is optional
        # on the wire, so every card written by an earlier release still validates.
        for name in d2:
            assert sorted(d3[name].get("required", [])) == sorted(d2[name].get("required", [])), (
                name
            )
        # Properties: exactly these are new, on exactly these definitions.
        new_properties = {
            "Fit": {"cell_estimates", "dependence", "cluster_cells"},
            "Threshold": {"bootstrap"},
            "Ratio": {"bootstrap"},
        }
        for name in d2:
            added = set(d3[name].get("properties", {})) - set(d2[name].get("properties", {}))
            assert added == new_properties.get(name, set()), name
            assert set(d2[name].get("properties", {})) <= set(d3[name].get("properties", {}))
        # The new definitions' fields and `required` lists.
        assert set(d3["ClusterCell"]["properties"]) == {
            "severity",
            "cluster",
            "successes",
            "trials",
        }
        assert sorted(d3["ClusterCell"]["required"]) == sorted(d3["ClusterCell"]["properties"])
        assert set(d3["ClusterAdjacentPair"]["properties"]) == {
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
        assert sorted(d3["ClusterAdjacentPair"]["required"]) == sorted(
            d3["ClusterAdjacentPair"]["properties"]
        )
        assert d3["ClusterCell"]["additionalProperties"] is False
        assert d3["ClusterAdjacentPair"]["additionalProperties"] is False
        # The new Fit properties.
        assert d3["Fit"]["properties"]["dependence"]["enum"] == ["independent", "clustered"]
        assert d3["Fit"]["properties"]["cluster_cells"]["anyOf"] == [
            {"type": "null"},
            {"type": "array", "items": {"$ref": "#/$defs/ClusterCell"}},
        ]
        # The new Ratio property is a null-or-summary, like Threshold's.
        assert d3["Ratio"]["properties"]["bootstrap"]["anyOf"] == [
            {"type": "null"},
            {"$ref": "#/$defs/BootstrapSummary"},
        ]
        assert (
            d3["Ratio"]["properties"]["bootstrap"]["anyOf"]
            == d3["Threshold"]["properties"]["bootstrap"]["anyOf"]
        )

        edited = copy.deepcopy(v2)
        ed = edited["$defs"]
        fit = ed["Fit"]
        fit["properties"]["model"]["enum"].append("isotonic")
        fit["properties"]["link"]["enum"].append("none")
        ed["Threshold"]["properties"]["interval_method"]["enum"].extend(
            ["parametric_bootstrap", "cell_bootstrap", "cluster_bootstrap"]
        )
        ed["Threshold"]["properties"]["dependence"]["enum"].append("clustered")
        ed["Ratio"]["properties"]["method"]["enum"].append("cluster_bootstrap")
        ed["Ratio"]["properties"]["dependence"]["enum"].append("paired")
        monotonicity = ed["MonotonicityCheck"]["properties"]
        monotonicity["dependence"]["enum"].append("clustered")
        monotonicity["method"]["enum"].append("cluster_sign_test_one_sided_holm")
        monotonicity["pairs"]["items"]["oneOf"].append({"$ref": "#/$defs/ClusterAdjacentPair"})
        # The values of the new properties, the Fit conditionals and the new definitions are
        # taken from version 3; their behaviour is pinned by name in the validation tests.
        for key in ("cell_estimates", "dependence", "cluster_cells"):
            fit["properties"][key] = copy.deepcopy(d3["Fit"]["properties"][key])
        for name in ("Threshold", "Ratio"):
            ed[name]["properties"]["bootstrap"] = copy.deepcopy(d3[name]["properties"]["bootstrap"])
        fit["allOf"] = copy.deepcopy(d3["Fit"]["allOf"])
        for extra in set(d3) - set(d2):
            ed[extra] = copy.deepcopy(d3[extra])

        assert _digest(edited) == _digest(v3)

    def test_the_version_3_description_names_the_isotonic_model(self) -> None:
        assert "isotonic" in _schema("3")["description"]


class TestNewFieldShapes:
    def test_cell_estimates_is_null_or_an_array_of_numbers(self) -> None:
        payload = _json(_counts_fit())
        validator = _validator("Fit")
        validator.validate(payload)

        for bad in ("text", [1.0, "x"], {"a": 1.0}):
            broken = copy.deepcopy(payload)
            broken["cell_estimates"] = bad
            with pytest.raises(jsonschema.ValidationError):
                validator.validate(broken)

    def test_bootstrap_has_the_five_integer_fields_and_no_others(self) -> None:
        payload = _json(_counts_threshold())
        validator = _validator("Threshold")
        validator.validate(payload)

        assert set(payload["bootstrap"]) == {
            "n_resamples",
            "failed",
            "unreachable_low",
            "unreachable_high",
            "seed",
        }
        assert all(isinstance(v, int) for v in payload["bootstrap"].values())
        extra = copy.deepcopy(payload)
        extra["bootstrap"]["unexpected"] = 1
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(extra)
        missing = copy.deepcopy(payload)
        del missing["bootstrap"]["seed"]
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(missing)
        fractional = copy.deepcopy(payload)
        fractional["bootstrap"]["failed"] = 0.5
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(fractional)

    def test_bootstrap_is_null_on_a_threshold_that_ran_no_resampling(self) -> None:
        payload = _json(fake_threshold())

        assert payload["bootstrap"] is None
        _validator("Threshold").validate(payload)

    def test_every_bootstrap_interval_method_is_valid_and_an_unknown_one_is_not(self) -> None:
        validator = _validator("Threshold")
        for method in ("parametric_bootstrap", "cell_bootstrap"):
            validator.validate(_json(fake_threshold(interval_method=method)))
        broken = _json(fake_threshold())
        broken["interval_method"] = "jackknife"
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(broken)


class TestFitConditionals:
    def test_an_isotonic_fit_has_null_params_and_validates(self) -> None:
        payload = _json(_counts_fit())

        assert payload["model"] == "isotonic" and payload["link"] == "none"
        assert payload["status"] == "OK"
        assert payload["params"] is None and payload["covariance"] is None
        assert payload["diagnostics"] is None
        assert isinstance(payload["cell_estimates"], list)
        _validator("Fit").validate(payload)

    def test_an_isotonic_fit_without_cell_estimates_is_invalid(self) -> None:
        payload = _json(_counts_fit())
        payload["cell_estimates"] = None

        with pytest.raises(jsonschema.ValidationError):
            _validator("Fit").validate(payload)

    def test_a_parametric_fit_with_cell_estimates_is_invalid(self) -> None:
        payload = _json(fake_fit())
        payload["cell_estimates"] = [0.9, 0.5, 0.1]

        with pytest.raises(jsonschema.ValidationError):
            _validator("Fit").validate(payload)

    def test_a_gaussian_fit_with_cell_estimates_is_invalid(self) -> None:
        payload = _json(fake_fit(model="gaussian"))
        payload["cell_estimates"] = [9.0, 5.0, 1.0]

        with pytest.raises(jsonschema.ValidationError):
            _validator("Fit").validate(payload)

    @pytest.mark.parametrize("model", ["binomial", "gaussian"])
    def test_a_parametric_fit_has_null_cell_estimates_and_validates(self, model: str) -> None:
        payload = _json(fake_fit(model=model))

        assert payload["cell_estimates"] is None
        _validator("Fit").validate(payload)


class TestEveryIsotonicResultValidatesAndRoundTrips:
    @pytest.mark.parametrize(("name", "builder"), _ISOTONIC_BUILDERS, ids=_ISOTONIC_IDS)
    def test_a_scorecard_of_it_validates_against_version_3(
        self, name: str, builder: Callable[[], Any]
    ) -> None:
        card = Scorecard(results=(builder(),), provenance={})

        jsonschema.Draft202012Validator(_schema("3")).validate(_json(card))

    @pytest.mark.parametrize(("name", "builder"), _ISOTONIC_BUILDERS, ids=_ISOTONIC_IDS)
    def test_it_round_trips_through_json_text_unchanged(
        self, name: str, builder: Callable[[], Any]
    ) -> None:
        result = builder()

        reloaded = from_dict(_json(result))

        assert reloaded == result

    @pytest.mark.parametrize(("name", "builder"), _ISOTONIC_BUILDERS, ids=_ISOTONIC_IDS)
    def test_it_does_not_validate_against_version_2(
        self, name: str, builder: Callable[[], Any]
    ) -> None:
        card = Scorecard(results=(builder(),), provenance={})

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(_schema("2")).validate(_json(card))

    def test_an_isotonic_fit_round_trips_with_its_estimates_and_flags(self) -> None:
        fit = _counts_fit()

        reloaded = from_dict(_json(fit))

        assert isinstance(reloaded, Fit)
        assert reloaded.model == "isotonic" and reloaded.link == "none"
        assert reloaded.cell_estimates == fit.cell_estimates
        assert reloaded.log_likelihood == fit.log_likelihood
        assert (reloaded.params, reloaded.covariance, reloaded.residual) == (None,) * 3

    def test_an_isotonic_threshold_round_trips_with_method_definition_level_and_summary(
        self,
    ) -> None:
        t = _counts_threshold()
        assert t.status is Status.OK

        payload = _json(t)
        reloaded = from_dict(payload)

        assert isinstance(reloaded, Threshold)
        assert reloaded == t
        assert reloaded.interval_method == "cell_bootstrap"
        assert reloaded.definition == Definition.absolute(0.5)
        assert reloaded.level == _LEVEL
        assert isinstance(reloaded.bootstrap, BootstrapSummary)
        assert reloaded.bootstrap == t.bootstrap
        assert payload["bootstrap"]["n_resamples"] == 49 and payload["bootstrap"]["seed"] == 11
        assert reloaded.fit.cell_estimates == t.fit.cell_estimates

    def test_a_censored_isotonic_threshold_round_trips_with_a_null_bootstrap(self) -> None:
        obs = Observations.from_counts(
            _LOG,
            severity=[0.0, 1.0, 2.0, 4.0, 8.0],
            successes=[200, 200, 199, 199, 198],
            trials=[200] * 5,
            outcome="outcome",
            direction="decreasing",
        )
        t = threshold(
            fit_dose_response(obs, model="isotonic"),
            definition=Definition.absolute(0.5),
            interval_method="cell_bootstrap",
            level=_LEVEL,
            n_resamples=49,
            seed=3,
        )
        assert t.censoring is Censoring.RIGHT

        payload = _json(t)

        assert payload["bootstrap"] is None
        assert from_dict(payload) == t

    def test_a_parametric_bootstrap_fake_round_trips(self) -> None:
        t = fake_threshold(interval_method="parametric_bootstrap")

        assert from_dict(_json(t)) == t


class TestOlderVersionsCarryingVersion3Values:
    @pytest.mark.parametrize("version", ["1", "2"])
    @pytest.mark.parametrize(("name", "builder"), _ISOTONIC_BUILDERS, ids=_ISOTONIC_IDS)
    def test_an_isotonic_payload_tagged_with_an_older_version_is_rejected(
        self, name: str, builder: Callable[[], Any], version: str
    ) -> None:
        payload = _relabel(_json(builder()), version)

        with pytest.raises(ValueError):
            from_dict(payload)

    @pytest.mark.parametrize("version", ["1", "2"])
    @pytest.mark.parametrize("method", ["parametric_bootstrap", "cell_bootstrap"])
    def test_a_bootstrap_interval_method_on_an_older_version_is_rejected(
        self, method: str, version: str
    ) -> None:
        payload = _relabel(_json(fake_threshold(interval_method=method)), version)

        with pytest.raises(ValueError):
            from_dict(payload)

    @pytest.mark.parametrize("version", ["1", "2"])
    def test_a_bootstrap_summary_on_an_older_version_is_rejected(self, version: str) -> None:
        payload = _relabel(_json(fake_threshold()), version)
        payload["bootstrap"] = {
            "n_resamples": 49,
            "failed": 0,
            "unreachable_low": 0,
            "unreachable_high": 0,
            "seed": 1,
        }

        with pytest.raises(ValueError):
            from_dict(payload)

    @pytest.mark.parametrize("version", ["1", "2"])
    def test_cell_estimates_on_an_older_version_are_rejected(self, version: str) -> None:
        payload = _relabel(_json(fake_fit()), version)
        payload["cell_estimates"] = [0.9, 0.5, 0.1]

        with pytest.raises(ValueError):
            from_dict(payload)

    @pytest.mark.parametrize("version", ["1", "2"])
    def test_link_none_on_an_older_version_is_rejected(self, version: str) -> None:
        payload = _relabel(_json(fake_fit()), version)
        payload["link"] = "none"

        with pytest.raises(ValueError):
            from_dict(payload)

    def test_an_isotonic_result_nested_in_an_older_card_is_rejected(self) -> None:
        card = Scorecard(results=(_counts_threshold(),), provenance={})

        with pytest.raises(ValueError):
            from_dict(_relabel(_json(card), "2"))


_READ_BACK: list[tuple[str, Callable[[], Any]]] = [
    ("fake_fit", fake_fit),
    ("fake_threshold", fake_threshold),
    ("fake_ratio", fake_ratio),
    ("gaussian_fit", lambda: fake_fit(model="gaussian")),
    ("gaussian_threshold", lambda: fake_threshold(model="gaussian")),
    ("increasing_threshold", lambda: fake_threshold(direction="increasing")),
]


class TestCardsWrittenBeforeVersion3StillRead:
    @pytest.mark.parametrize("version", ["1", "2"])
    @pytest.mark.parametrize(("name", "builder"), _READ_BACK, ids=[n for n, _ in _READ_BACK])
    def test_a_card_without_the_new_keys_is_read_and_written_back_with_its_own_tag(
        self, name: str, builder: Callable[[], Any], version: str
    ) -> None:
        original = builder()
        if version == "1" and getattr(original, "direction", "decreasing") != "decreasing":
            pytest.skip("version 1 has no increasing results")
        if version == "1" and "gaussian" in name:
            pytest.skip("version 1 has no Gaussian results")
        old = _relabel(
            _without_v3_only_keys(_json(Scorecard(results=(original,), provenance={}))), version
        )

        card = from_dict(old)

        assert isinstance(card, Scorecard)
        (result,) = card.results
        rewritten = _json(card)
        assert set(_versions_in(rewritten)) == {version}
        assert rewritten == old
        assert result.schema_version == version
        assert card.schema_version == version
        assert from_dict(rewritten) == card

    @pytest.mark.parametrize("version", ["1", "2"])
    def test_an_older_card_with_explicit_null_new_keys_is_also_read(self, version: str) -> None:
        payload = _relabel(_json(fake_threshold()), version)

        assert payload["bootstrap"] is None and payload["fit"]["cell_estimates"] is None
        assert payload["fit"]["cluster_cells"] is None
        assert payload["fit"]["dependence"] == "independent"
        result = from_dict(payload)
        assert result.schema_version == version
        assert set(_versions_in(_json(result))) == {version}
        assert _json(result)["value"] == payload["value"]

    def test_a_version_2_card_with_a_real_parametric_threshold_is_read(self) -> None:
        obs = Observations.from_counts(
            _LOG,
            severity=[0.0, 1.0, 2.0, 4.0, 8.0],
            successes=[100, 90, 60, 20, 5],
            trials=[100] * 5,
            outcome="success",
            direction="decreasing",
        )
        t = threshold(
            fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0),
            definition=Definition.absolute(0.5),
            interval_method="profile",
            level=_LEVEL,
        )
        old = _relabel(_without_v3_only_keys(_json(t)), "2")

        reloaded = from_dict(old)

        assert isinstance(reloaded, Threshold)
        assert (reloaded.value, reloaded.lo, reloaded.hi) == (t.value, t.lo, t.hi)
        assert reloaded.bootstrap is None and reloaded.fit.cell_estimates is None
