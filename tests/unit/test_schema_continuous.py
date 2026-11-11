"""Score-card schema ``"2"`` with continuous results: validation, round trip, and version rules.

Schema ``"2"`` carries Gaussian fits as well as binomial ones. ``Fit.model`` accepts
``"gaussian"``; for that model ``cells`` are ``ContinuousCell`` (``severity``, ``n``, ``mean``,
``sd``), ``residual`` is present, and ``upper`` and ``lower`` are any finite response values, where
for the binomial model they are probabilities in ``[0, 1]``. A payload tagged ``"1"`` that carries
Gaussian content is rejected, since a ``"1"`` writer could never have produced it.
"""

from __future__ import annotations

import copy
import functools
import json
import math
from typing import Any

import jsonschema
import numpy as np
import pytest

from robustmargin import (
    Axis,
    ContinuousObservations,
    Definition,
    Fit,
    Observations,
    Ratio,
    Scorecard,
    Status,
    Threshold,
    fit_dose_response,
    ratio_interval,
    threshold,
)
from robustmargin.report import from_dict, load_schema, to_dict
from robustmargin.testing import fake_fit, fake_threshold

_AXIS = Axis(name="severity", unit="unit", scale="log")
_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]


def _schema(version: str) -> dict[str, Any]:
    schema: dict[str, Any] = load_schema(version)
    return schema


def _validator(version: str, type_name: str) -> jsonschema.protocols.Validator:
    return jsonschema.Draft202012Validator(
        {"$defs": _schema(version)["$defs"], "$ref": f"#/$defs/{type_name}"}
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


def _gaussian_fit(*, direction: str = "decreasing", seed: int = 5, **asymptotes: Any) -> Fit:
    rng = np.random.default_rng(seed)
    severity = [x for x in _LEVELS for _ in range(4)]
    x = np.asarray(severity)
    with np.errstate(divide="ignore"):
        z = (np.log(x) - math.log(0.8)) / 0.5
    y = 10.0 - 9.0 * np.where(x == 0.0, 0.0, 1.0 / (1.0 + np.exp(-z)))
    y = y + rng.normal(0.0, 0.3, size=x.size)
    if direction == "increasing":
        y = 11.0 - y
    obs = ContinuousObservations.from_observations(
        _AXIS, severity=severity, value=y.tolist(), outcome="response", direction=direction
    )
    fit = fit_dose_response(
        obs,
        model="gaussian",
        link="logit",
        **{"upper": "estimate", "lower": "estimate", **asymptotes},
    )
    assert fit.status is Status.OK
    return fit


def _failed_fit() -> Fit:
    obs = ContinuousObservations.from_observations(
        _AXIS,
        severity=[0.5, 1.0, 2.0, 4.0],
        value=[9.0, 7.0, 3.0, 1.5],
        outcome="response",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="gaussian", link="logit", upper="estimate", lower="estimate")
    assert fit.status is Status.NOT_CONVERGED
    return fit


def _threshold(fit: Fit, definition: Definition, method: str) -> Threshold:
    return threshold(fit, definition=definition, interval_method=method, level=0.95)


def _binomial_fit() -> Fit:
    obs = Observations.from_counts(
        _AXIS,
        severity=[0.0, 1.0, 2.0, 4.0, 8.0],
        successes=[100, 90, 60, 20, 5],
        trials=[100] * 5,
        outcome="success",
        direction="decreasing",
    )
    return fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)


@functools.cache
def _results() -> dict[str, object]:
    fit = _gaussian_fit()
    inc = _gaussian_fit(direction="increasing")
    fixed = _gaussian_fit(upper=10.0, lower=1.0)
    t_profile = _threshold(fit, Definition.relative(0.5), "profile")
    t_delta = _threshold(fit, Definition.absolute(5.5), "delta")
    t_inc = _threshold(inc, Definition.baseline_multiple(2.0), "delta")
    t_other = _threshold(_gaussian_fit(seed=9), Definition.absolute(5.5), "delta")
    t_unreachable = _threshold(fit, Definition.absolute(500.0), "profile")
    failed = _failed_fit()
    ratio = ratio_interval(t_delta, t_other, method="fieller", dependence="independent")
    results: dict[str, object] = {
        "fit": fit,
        "increasing fit": inc,
        "fixed-asymptote fit": fixed,
        "failed fit": failed,
        "profile threshold": t_profile,
        "delta threshold": t_delta,
        "increasing threshold": t_inc,
        "unreachable threshold": t_unreachable,
        "failed threshold": _threshold(failed, Definition.relative(0.5), "profile"),
        "ratio": ratio,
        "fake fit": fake_fit(model="gaussian"),
        "fake threshold": fake_threshold(model="gaussian"),
    }
    results["card"] = Scorecard(results=tuple(results.values()), provenance={})  # type: ignore[arg-type]
    return results


_IDS = [
    "fit",
    "increasing fit",
    "fixed-asymptote fit",
    "failed fit",
    "profile threshold",
    "delta threshold",
    "increasing threshold",
    "unreachable threshold",
    "failed threshold",
    "ratio",
    "fake fit",
    "fake threshold",
    "card",
]


def _type_name(obj: object) -> str:
    return type(obj).__name__


class TestSchemaDefinitions:
    @pytest.mark.parametrize("name", ["ContinuousCell", "Residual", "ContinuousDiagnostics"])
    def test_the_new_definitions_exist_in_version_2_only(self, name: str) -> None:
        assert name in _schema("2")["$defs"]
        assert name not in _schema("1")["$defs"]

    def test_the_model_enum_gains_gaussian_in_version_2_and_not_in_1(self) -> None:
        payload = _dump(_gaussian_fit())

        _validator("2", "Fit").validate(payload)
        with pytest.raises(jsonschema.ValidationError):
            _validator("1", "Fit").validate(payload)

    def test_the_schema_is_still_a_valid_draft_2020_12_document(self) -> None:
        jsonschema.Draft202012Validator.check_schema(_schema("2"))


class TestEveryGaussianResultValidates:
    @pytest.mark.parametrize("name", _IDS)
    def test_it_validates_against_version_2(self, name: str) -> None:
        result = _results()[name]
        payload = json.loads(json.dumps(_dump(result), allow_nan=False))

        if isinstance(result, Scorecard):
            jsonschema.Draft202012Validator(_schema("2")).validate(payload)
        else:
            _validator("2", _type_name(result)).validate(payload)

    @pytest.mark.parametrize("name", _IDS)
    def test_it_round_trips_through_json_text(self, name: str) -> None:
        result = _results()[name]

        assert from_dict(json.loads(json.dumps(_dump(result), allow_nan=False))) == result

    @pytest.mark.parametrize("name", _IDS)
    def test_every_nested_schema_version_is_2(self, name: str) -> None:
        def versions(node: Any) -> list[Any]:
            if isinstance(node, dict):
                found = [v for k, v in node.items() if k == "schema_version"]
                return found + [
                    x for k, v in node.items() if k != "schema_version" for x in versions(v)
                ]
            if isinstance(node, list):
                return [x for item in node for x in versions(item)]
            return []

        assert set(versions(_dump(_results()[name]))) == {"2"}

    def test_a_gaussian_threshold_serialises_what_makes_it_reproducible(self) -> None:
        t = _results()["profile threshold"]
        assert isinstance(t, Threshold)

        payload = _dump(t)

        assert payload["interval_method"] == "profile"
        assert payload["definition"] == {"kind": "relative", "value": 0.5}
        assert payload["level"] == 0.95
        assert payload["fit"]["model"] == "gaussian"
        assert set(payload["fit"]["residual"]) >= {"sigma", "df", "rss"}
        assert set(payload["fit"]["cells"][0]) >= {"severity", "n", "mean", "sd"}

    def test_the_ratio_keeps_its_method_and_level(self) -> None:
        ratio = _results()["ratio"]
        assert isinstance(ratio, Ratio)

        back = from_dict(_dump(ratio))

        assert isinstance(back, Ratio)
        assert (back.method, back.level, back.dependence) == ("fieller", 0.95, "independent")


class TestTheModelDecidesWhatShapeIsValid:
    def test_a_gaussian_asymptote_outside_the_unit_interval_is_valid(self) -> None:
        payload = _dump(_gaussian_fit())
        assert payload["params"]["upper"]["value"] > 1.0

        _validator("2", "Fit").validate(payload)

    def test_a_binomial_asymptote_outside_the_unit_interval_is_still_invalid(self) -> None:
        payload = _dump(_binomial_fit())
        payload["params"]["upper"]["value"] = 10.0

        with pytest.raises(jsonschema.ValidationError):
            _validator("2", "Fit").validate(payload)

    def test_a_gaussian_fit_with_binomial_cells_is_invalid(self) -> None:
        payload = _dump(_gaussian_fit())
        payload["cells"] = _dump(_binomial_fit())["cells"]

        with pytest.raises(jsonschema.ValidationError):
            _validator("2", "Fit").validate(payload)

    def test_a_binomial_fit_with_continuous_cells_is_invalid(self) -> None:
        payload = _dump(_binomial_fit())
        payload["cells"] = _dump(_gaussian_fit())["cells"]

        with pytest.raises(jsonschema.ValidationError):
            _validator("2", "Fit").validate(payload)

    def test_a_gaussian_fit_without_a_residual_is_invalid(self) -> None:
        payload = _dump(_gaussian_fit())
        payload["residual"] = None

        with pytest.raises(jsonschema.ValidationError):
            _validator("2", "Fit").validate(payload)

    def test_a_binomial_fit_carries_no_residual(self) -> None:
        payload = _dump(_binomial_fit())

        assert payload.get("residual") is None
        _validator("2", "Fit").validate(payload)

    def test_a_cell_with_a_negative_sd_or_no_observations_is_invalid(self) -> None:
        payload = _dump(_gaussian_fit())
        bad_sd = copy.deepcopy(payload)
        bad_sd["cells"][1]["sd"] = -1.0
        bad_n = copy.deepcopy(payload)
        bad_n["cells"][1]["n"] = 0

        for bad in (bad_sd, bad_n):
            with pytest.raises(jsonschema.ValidationError):
                _validator("2", "Fit").validate(bad)


class TestVersionOneRejectsContinuousContent:
    @pytest.mark.parametrize(
        "name", ["fit", "failed fit", "fake fit", "profile threshold", "fake threshold", "card"]
    )
    def test_a_payload_relabelled_1_is_rejected(self, name: str) -> None:
        payload = _relabel(_dump(_results()[name]), "1")

        with pytest.raises(ValueError):
            from_dict(payload)

    def test_a_payload_tagged_2_with_the_same_content_is_read(self) -> None:
        fit = _results()["fit"]

        assert from_dict(_dump(fit)) == fit

    def test_a_binomial_payload_relabelled_1_is_still_read(self) -> None:
        fit = _binomial_fit()

        assert from_dict(_relabel(_dump(fit), "1")).model == "binomial"  # type: ignore[attr-defined]

    @pytest.mark.parametrize("residual", [None, {"sigma": 1.0, "df": 3, "rss": 3.0}])
    def test_a_binomial_fit_tagged_1_with_a_residual_key_is_rejected_even_when_null(
        self, residual: object
    ) -> None:
        payload = _relabel(_dump(_binomial_fit()), "1")
        payload["residual"] = residual

        with pytest.raises(ValueError, match="residual"):
            from_dict(payload)

    def test_a_card_mixing_one_continuous_result_among_binomial_ones_is_rejected_as_a_whole(
        self,
    ) -> None:
        card = Scorecard(results=(_binomial_fit(), _gaussian_fit()), provenance={})

        with pytest.raises(ValueError):
            from_dict(_relabel(_dump(card), "1"))
