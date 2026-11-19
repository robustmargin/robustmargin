"""A card read from an older release keeps its schema version.

``from_dict`` returns the results of a ``"1"`` or ``"2"`` card tagged as read, so re-serialising
them writes the same tag and the same keys: the card stays readable by the release that wrote it.
Only a result constructed now defaults to ``"3"``. A ``"3"`` card reads and writes ``"3"`` as
before.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import jsonschema
import pytest

from robustmargin import Scorecard
from robustmargin.report import from_dict, load_schema, to_dict
from robustmargin.testing import fake_fit, fake_ratio, fake_threshold


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


_BUILDERS: list[tuple[str, Callable[[], Any]]] = [
    ("fake_fit", fake_fit),
    ("fake_threshold", fake_threshold),
    ("fake_ratio", fake_ratio),
    ("gaussian_fit", lambda: fake_fit(model="gaussian")),
    ("gaussian_threshold", lambda: fake_threshold(model="gaussian")),
    ("increasing_threshold", lambda: fake_threshold(direction="increasing")),
]


def _old_payload(builder: Callable[[], Any], name: str, version: str) -> dict[str, Any]:
    original = builder()
    if version == "1" and getattr(original, "direction", "decreasing") != "decreasing":
        pytest.skip("version 1 has no increasing results")
    if version == "1" and "gaussian" in name:
        pytest.skip("version 1 has no Gaussian results")
    return _relabel(_without_v3_only_keys(_json(original)), version)  # type: ignore[no-any-return]


@pytest.mark.parametrize("version", ["1", "2"])
@pytest.mark.parametrize(("name", "builder"), _BUILDERS, ids=[n for n, _ in _BUILDERS])
class TestAnOlderResultKeepsItsTag:
    def test_the_result_read_back_carries_the_tag_it_was_written_with(
        self, name: str, builder: Callable[[], Any], version: str
    ) -> None:
        result = from_dict(_old_payload(builder, name, version))

        assert result.schema_version == version

    def test_every_nested_object_carries_the_tag_too(
        self, name: str, builder: Callable[[], Any], version: str
    ) -> None:
        result = from_dict(_old_payload(builder, name, version))

        assert set(_versions_in(_json(result))) == {version}

    def test_writing_it_back_reproduces_the_payload_that_was_read(
        self, name: str, builder: Callable[[], Any], version: str
    ) -> None:
        old = _old_payload(builder, name, version)

        assert _json(from_dict(old)) == old

    def test_it_round_trips_through_to_dict_and_from_dict(
        self, name: str, builder: Callable[[], Any], version: str
    ) -> None:
        result = from_dict(_old_payload(builder, name, version))

        assert from_dict(_json(result)) == result

    def test_the_rewritten_payload_validates_against_the_schema_of_its_own_version(
        self, name: str, builder: Callable[[], Any], version: str
    ) -> None:
        result = from_dict(_old_payload(builder, name, version))
        schema = load_schema(version)
        type_name = type(result).__name__

        jsonschema.Draft202012Validator(
            {"$defs": schema["$defs"], "$ref": f"#/$defs/{type_name}"}
        ).validate(_json(result))


@pytest.mark.parametrize("version", ["1", "2"])
class TestAnOlderCardKeepsItsTag:
    def test_a_card_and_every_result_in_it_keep_the_tag(self, version: str) -> None:
        old = _relabel(
            _without_v3_only_keys(
                _json(Scorecard(results=(fake_fit(), fake_threshold()), provenance={"run": "a"}))
            ),
            version,
        )

        card = from_dict(old)

        assert isinstance(card, Scorecard) and card.schema_version == version
        assert {r.schema_version for r in card.results} == {version}
        assert _json(card) == old
        assert from_dict(_json(card)) == card

    def test_explicit_null_new_keys_on_an_older_payload_are_read_and_the_tag_kept(
        self, version: str
    ) -> None:
        payload = _relabel(_json(fake_threshold()), version)
        assert payload["bootstrap"] is None and payload["fit"]["cell_estimates"] is None

        result = from_dict(payload)

        assert result.schema_version == version
        assert set(_versions_in(_json(result))) == {version}


class TestNewResultsAndCurrentCards:
    def test_a_result_constructed_now_defaults_to_3(self) -> None:
        assert fake_fit().schema_version == "3"
        assert fake_threshold().schema_version == "3"
        assert set(_versions_in(_json(fake_threshold()))) == {"3"}

    @pytest.mark.parametrize(("name", "builder"), _BUILDERS, ids=[n for n, _ in _BUILDERS])
    def test_a_current_payload_reads_and_writes_3(
        self, name: str, builder: Callable[[], Any]
    ) -> None:
        payload = _json(builder())

        result = from_dict(payload)

        assert result.schema_version == "3"
        assert _json(result) == payload
