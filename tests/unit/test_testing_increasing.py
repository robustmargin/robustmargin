"""``fake_fit`` and ``fake_threshold`` gain a keyword-only ``direction`` that defaults to
``"decreasing"``, so every existing call is unchanged. ``"increasing"`` builds a schema-valid
increasing-outcome result.
"""

from __future__ import annotations

import inspect
import json

import jsonschema
import pytest

from robustmargin import Censoring, Status
from robustmargin.report import from_dict, load_schema, to_dict
from robustmargin.testing import fake_fit, fake_threshold

_FIT_STATUSES = [s for s in Status if s not in (Status.UNREACHABLE, Status.FAILS_AT_BASELINE)]
_THRESHOLD_COMBINATIONS = [
    *((Status.OK, c) for c in Censoring),
    *(
        (s, c)
        for s in (Status.SEPARATION, Status.NOT_CONVERGED)
        for c in (Censoring.NONE, Censoring.RIGHT, Censoring.LEFT)
    ),
    *(
        (s, Censoring.NONE)
        for s in (Status.UNREACHABLE, Status.FAILS_AT_BASELINE, Status.CONTROL_INCOMPATIBLE)
    ),
]


def _validator(type_name: str) -> jsonschema.protocols.Validator:
    schema = load_schema("2")
    return jsonschema.Draft202012Validator(
        {"$defs": schema["$defs"], "$ref": f"#/$defs/{type_name}"}
    )


class TestSignature:
    @pytest.mark.parametrize("function", [fake_fit, fake_threshold])
    def test_direction_is_keyword_only_and_defaults_to_decreasing(self, function: object) -> None:
        parameter = inspect.signature(function).parameters["direction"]  # type: ignore[arg-type]

        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default == "decreasing"

    def test_the_default_call_is_unchanged(self) -> None:
        assert fake_fit() == fake_fit(direction="decreasing")
        assert fake_threshold() == fake_threshold(direction="decreasing")
        assert fake_fit().direction == "decreasing"
        assert fake_threshold().direction == "decreasing"

    @pytest.mark.parametrize("function", [fake_fit, fake_threshold])
    def test_an_unknown_direction_raises_value_error(self, function: object) -> None:
        with pytest.raises(ValueError):
            function(direction="sideways")  # type: ignore[operator]


class TestFakeIncreasingFit:
    @pytest.mark.parametrize("status", _FIT_STATUSES)
    def test_every_status_builds_an_increasing_schema_valid_fit(self, status: Status) -> None:
        fit = fake_fit(status=status, direction="increasing")

        assert fit.direction == "increasing"
        assert fit.status is status
        _validator("Fit").validate(to_dict(fit))

    @pytest.mark.parametrize("status", _FIT_STATUSES)
    def test_it_round_trips_through_json_text(self, status: Status) -> None:
        fit = fake_fit(status=status, direction="increasing")

        assert from_dict(json.loads(json.dumps(to_dict(fit)))) == fit

    def test_an_ok_increasing_fit_has_a_valid_parameter_set(self) -> None:
        fit = fake_fit(direction="increasing")

        assert fit.params is not None
        assert fit.params["lower"].value < fit.params["upper"].value
        assert fit.monotonicity is not None
        assert fit.monotonicity.direction == "increasing"


class TestFakeIncreasingThreshold:
    @pytest.mark.parametrize(("status", "censoring"), _THRESHOLD_COMBINATIONS)
    def test_every_allowed_combination_is_an_increasing_schema_valid_threshold(
        self, status: Status, censoring: Censoring
    ) -> None:
        t = fake_threshold(status=status, censoring=censoring, direction="increasing")

        assert t.direction == "increasing"
        assert t.fit.direction == "increasing"
        assert (t.status, t.censoring) == (status, censoring)
        _validator("Threshold").validate(to_dict(t))

    @pytest.mark.parametrize(("status", "censoring"), _THRESHOLD_COMBINATIONS)
    def test_it_round_trips_through_json_text(self, status: Status, censoring: Censoring) -> None:
        t = fake_threshold(status=status, censoring=censoring, direction="increasing")

        assert from_dict(json.loads(json.dumps(to_dict(t)))) == t

    def test_a_forbidden_combination_still_raises_for_either_direction(self) -> None:
        for direction in ("decreasing", "increasing"):
            with pytest.raises(ValueError):
                fake_threshold(
                    status=Status.UNREACHABLE, censoring=Censoring.RIGHT, direction=direction
                )
