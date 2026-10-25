"""Score-card schema ``"2"``: what it adds to ``"1"``, and what reading ``"1"`` still means.

Schema ``"2"`` differs from ``"1"`` in exactly three places, and nowhere else:

* ``Fit.direction`` accepts ``"decreasing"`` and ``"increasing"``;
* ``Threshold.direction`` accepts both;
* ``Definition.kind`` adds ``"baseline_multiple"``.

``scorecard-v1.json`` stays byte-for-byte what it was, so ``load_schema("1")`` still works, and
``from_dict`` reads both versions. A payload tagged ``"1"`` that carries ``"increasing"`` or
``"baseline_multiple"`` is rejected, because a ``"1"`` writer could never have produced it.
``to_dict`` always writes ``"2"``.

``test_schema_v1.py`` pins what stays of ``"1"``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from robustmargin import (
    Axis,
    Censoring,
    Definition,
    Fit,
    GridBreakPoint,
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
_AXIS = Axis(name="severity", unit="unit", scale="log")
_SCHEMA_DIR = Path(report_module.__file__).resolve().parent / "schema"

# The structural digest of scorecard-v1.json recorded in test_schema_v1.py. Version "1" must
# still be exactly that document.
_V1_STRUCTURAL_DIGEST = "6f7a70d8e037caefe6ae88b615443b180eab240971e7f8c8250568b232f86829"


def _strip(node: Any, *, sort_enums: bool, root: bool = True) -> Any:
    """The validation content of a schema: prose and ``$id`` are left out, and so is the order of
    every ``required`` list. ``sort_enums`` also leaves out the order of ``enum`` lists, which the
    digest recorded in ``test_schema_v1.py`` does not."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if root and key == "$id":
                continue
            if key in ("title", "description") and isinstance(value, str):
                continue
            if (key == "required" or (sort_enums and key == "enum")) and isinstance(value, list):
                out[key] = sorted(value)
                continue
            out[key] = _strip(value, sort_enums=sort_enums, root=False)
        return out
    if isinstance(node, list):
        return [_strip(item, sort_enums=sort_enums, root=False) for item in node]
    return node


def _digest(schema: dict[str, Any], *, sort_enums: bool = False) -> str:
    canonical = json.dumps(
        _strip(schema, sort_enums=sort_enums), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _schema(version: str) -> dict[str, Any]:
    schema: dict[str, Any] = load_schema(version)
    return schema


def _validator(version: str, type_name: str) -> jsonschema.protocols.Validator:
    schema = _schema(version)
    return jsonschema.Draft202012Validator(
        {"$defs": schema["$defs"], "$ref": f"#/$defs/{type_name}"}
    )


def _dump(obj: object) -> dict[str, Any]:
    payload: dict[str, Any] = to_dict(obj)
    return payload


def _relabel(obj: Any, version: str) -> Any:
    if isinstance(obj, dict):
        return {
            k: (version if k == "schema_version" else _relabel(v, version)) for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_relabel(item, version) for item in obj]
    return obj


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


def _increasing_fit() -> Fit:
    obs = Observations.from_counts(
        _AXIS,
        severity=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        successes=[20, 22, 35, 90, 220, 340, 380],
        trials=[400] * 7,
        outcome="outcome",
        direction="increasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower="estimate")
    assert fit.status is Status.OK
    return fit


def _increasing_threshold() -> Threshold:
    return threshold(
        _increasing_fit(),
        definition=Definition.baseline_multiple(2.0),
        interval_method="profile",
        level=_LEVEL,
    )


def _real_decreasing_threshold() -> Threshold:
    obs = Observations.from_counts(
        _AXIS,
        severity=[0.0, 1.0, 2.0, 4.0, 8.0],
        successes=[100, 90, 60, 20, 5],
        trials=[100] * 5,
        outcome="success",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
    return threshold(
        fit, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
    )


class TestVersionConstants:
    def test_the_current_schema_version_is_2(self) -> None:
        assert SCHEMA_VERSION == "2"

    def test_the_reader_supports_both_versions(self) -> None:
        assert report_module._SUPPORTED_SCHEMA_VERSIONS == frozenset({"1", "2"})

    def test_results_default_to_the_current_schema_version(self) -> None:
        assert fake_fit().schema_version == "2"
        assert fake_threshold().schema_version == "2"
        assert fake_ratio().schema_version == "2"

    def test_to_dict_writes_2_at_every_nesting_level(self) -> None:
        card = Scorecard(
            results=(fake_fit(), fake_threshold(), fake_ratio(), _increasing_threshold()),
            provenance={},
        )

        versions = _versions_in(_dump(card))

        assert versions
        assert set(versions) == {"2"}


class TestSchemaFiles:
    def test_both_packaged_schemas_exist_and_nothing_else(self) -> None:
        names = sorted(p.name for p in _SCHEMA_DIR.glob("scorecard-v*.json"))

        assert names == ["scorecard-v1.json", "scorecard-v2.json"]

    @pytest.mark.parametrize("version", ["1", "2"])
    def test_each_is_a_valid_json_schema(self, version: str) -> None:
        jsonschema.Draft202012Validator.check_schema(_schema(version))

    def test_load_schema_defaults_to_the_current_version(self) -> None:
        assert load_schema() == load_schema("2")
        assert load_schema("1") != load_schema("2")

    def test_the_two_documents_have_different_ids(self) -> None:
        assert _schema("1")["$id"] != _schema("2")["$id"]

    def test_version_1_is_unchanged(self) -> None:
        assert _digest(_schema("1")) == _V1_STRUCTURAL_DIGEST

    def test_version_1_still_admits_only_the_decreasing_binomial_vocabulary(self) -> None:
        v1 = _schema("1")["$defs"]

        assert v1["Fit"]["properties"]["direction"]["enum"] == ["decreasing"]
        assert v1["Threshold"]["properties"]["direction"]["enum"] == ["decreasing"]
        assert "baseline_multiple" not in v1["Definition"]["properties"]["kind"]["enum"]

    def test_version_2_differs_from_version_1_in_exactly_the_three_places(self) -> None:
        v2 = _schema("2")["$defs"]

        assert sorted(v2["Fit"]["properties"]["direction"]["enum"]) == [
            "decreasing",
            "increasing",
        ]
        assert sorted(v2["Threshold"]["properties"]["direction"]["enum"]) == [
            "decreasing",
            "increasing",
        ]
        assert sorted(v2["Definition"]["properties"]["kind"]["enum"]) == [
            "absolute",
            "baseline_fraction",
            "baseline_multiple",
            "relative",
        ]

        # Apply those three edits to version 1 and nothing else: the validation content must then
        # be identical to version 2's.
        edited = copy.deepcopy(_schema("1"))
        edited["$defs"]["Fit"]["properties"]["direction"]["enum"] = ["decreasing", "increasing"]
        edited["$defs"]["Threshold"]["properties"]["direction"]["enum"] = [
            "decreasing",
            "increasing",
        ]
        edited["$defs"]["Definition"]["properties"]["kind"]["enum"].append("baseline_multiple")
        assert _digest(edited, sort_enums=True) == _digest(_schema("2"), sort_enums=True)

    def test_the_version_2_descriptions_no_longer_say_increasing_is_unsupported(self) -> None:
        v2 = _schema("2")
        texts = [
            v2["$defs"]["Fit"]["description"],
            v2["$defs"]["Threshold"]["description"],
            v2["description"],
        ]

        assert not any("not yet supported" in text for text in texts)


_RESULT_BUILDERS: list[tuple[str, Callable[[], Any]]] = [
    ("fake_fit", fake_fit),
    ("fake_threshold", fake_threshold),
    ("fake_ratio", fake_ratio),
    ("increasing_fit", _increasing_fit),
    ("increasing_threshold", _increasing_threshold),
    ("decreasing_threshold", _real_decreasing_threshold),
    ("grid", lambda: GridBreakPoint(0.1, 0.2, Censoring.NONE)),
]


class TestEveryToDictOutputValidatesAgainstVersion2:
    @pytest.mark.parametrize(
        ("name", "builder"), _RESULT_BUILDERS, ids=[n for n, _ in _RESULT_BUILDERS]
    )
    def test_a_scorecard_of_it_validates(self, name: str, builder: Callable[[], Any]) -> None:
        card = Scorecard(results=(builder(),), provenance={})

        payload = json.loads(json.dumps(_dump(card), allow_nan=False))

        jsonschema.Draft202012Validator(_schema("2")).validate(payload)

    @pytest.mark.parametrize(
        "status", [s for s in Status if s not in (Status.UNREACHABLE, Status.FAILS_AT_BASELINE)]
    )
    def test_a_fit_of_every_status_validates(self, status: Status) -> None:
        _validator("2", "Fit").validate(_dump(fake_fit(status=status)))

    def test_the_increasing_threshold_does_not_validate_against_version_1(self) -> None:
        payload = _dump(_increasing_threshold())

        with pytest.raises(jsonschema.ValidationError):
            _validator("1", "Threshold").validate(payload)

    def test_a_decreasing_result_relabelled_1_still_validates_against_version_1(self) -> None:
        payload = _relabel(_dump(_real_decreasing_threshold()), "1")

        _validator("1", "Threshold").validate(payload)

    def test_a_baseline_multiple_definition_is_valid_in_2_and_not_in_1(self) -> None:
        payload = _dump(_real_decreasing_threshold())
        payload["definition"] = {"kind": "baseline_multiple", "value": 2.0}

        _validator("2", "Threshold").validate(payload)
        with pytest.raises(jsonschema.ValidationError):
            _validator("1", "Threshold").validate(payload)


class TestIncreasingResultsRoundTrip:
    def test_an_increasing_fit_round_trips_through_json_text(self) -> None:
        fit = _increasing_fit()

        reloaded = from_dict(json.loads(json.dumps(_dump(fit), allow_nan=False)))

        assert reloaded == fit
        assert isinstance(reloaded, Fit) and reloaded.direction == "increasing"

    def test_an_increasing_threshold_round_trips_with_definition_direction_and_flags(self) -> None:
        t = _increasing_threshold()
        assert t.definition == Definition.baseline_multiple(2.0)

        payload = json.loads(json.dumps(_dump(t), allow_nan=False))
        reloaded = from_dict(payload)

        assert reloaded == t
        assert isinstance(reloaded, Threshold)
        assert reloaded.direction == "increasing"
        assert reloaded.fit.direction == "increasing"
        assert reloaded.definition == Definition.baseline_multiple(2.0)
        assert reloaded.status is t.status and reloaded.censoring is t.censoring
        assert payload["definition"] == {"kind": "baseline_multiple", "value": 2.0}
        assert payload["direction"] == "increasing"

    def test_an_increasing_scorecard_round_trips(self) -> None:
        card = Scorecard(results=(_increasing_fit(), _increasing_threshold()), provenance={})

        assert from_dict(json.loads(json.dumps(_dump(card)))) == card

    def test_the_fake_increasing_results_round_trip(self) -> None:
        for built in (fake_fit(direction="increasing"), fake_threshold(direction="increasing")):
            assert from_dict(json.loads(json.dumps(_dump(built)))) == built


class TestVersionOnePayloads:
    def test_a_version_1_decreasing_scorecard_is_still_read(self) -> None:
        t = _real_decreasing_threshold()
        payload = _relabel(_dump(Scorecard(results=(t,), provenance={})), "1")

        card = from_dict(payload)

        assert isinstance(card, Scorecard)
        (result,) = card.results
        assert isinstance(result, Threshold)
        assert (result.value, result.lo, result.hi) == (t.value, t.lo, t.hi)
        assert result.status is t.status and result.definition == t.definition
        assert result.direction == "decreasing"

    @pytest.mark.parametrize(
        "builder",
        [
            lambda: fake_fit(direction="increasing"),
            lambda: fake_threshold(direction="increasing"),
            _increasing_fit,
            _increasing_threshold,
        ],
        ids=["fake_fit", "fake_threshold", "fit", "threshold"],
    )
    def test_a_version_1_payload_carrying_an_increasing_direction_is_rejected(
        self, builder: Callable[[], Any]
    ) -> None:
        payload = _relabel(_dump(builder()), "1")

        with pytest.raises(ValueError):
            from_dict(payload)

    def test_a_version_1_card_carrying_an_increasing_result_is_rejected(self) -> None:
        payload = _relabel(_dump(Scorecard(results=(_increasing_threshold(),), provenance={})), "1")

        with pytest.raises(ValueError):
            from_dict(payload)

    def test_a_version_1_payload_carrying_baseline_multiple_is_rejected(self) -> None:
        payload = _relabel(_dump(_real_decreasing_threshold()), "1")
        payload["definition"] = {"kind": "baseline_multiple", "value": 2.0}

        with pytest.raises(ValueError):
            from_dict(payload)

    def test_a_version_2_payload_with_the_same_content_is_read(self) -> None:
        t = _increasing_threshold()

        assert from_dict(_dump(t)) == t

    @pytest.mark.parametrize("version", ["0", "3", "4", ""])
    def test_other_versions_are_still_rejected_naming_the_supported_ones(
        self, version: str
    ) -> None:
        payload = _relabel(_dump(fake_threshold()), version)

        with pytest.raises(ValueError) as excinfo:
            from_dict(payload)

        message = str(excinfo.value)
        assert "unknown schema_version" in message
        assert "'1'" in message and "'2'" in message
