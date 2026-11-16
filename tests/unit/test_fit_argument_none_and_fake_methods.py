"""Two small contracts kept across the isotonic release.

* **An explicit ``None`` for ``link``, ``upper`` or ``lower``** on a binomial or Gaussian fit
  raises what the previous release raised: ``ValueError`` for ``link`` (not one of the links), and
  ``TypeError`` for ``upper`` and ``lower`` (not a number). A missing argument is a ``TypeError``
  that names it. The isotonic model's relaxed signature does not change either.
* **``fake_threshold(interval_method=)``** builds an ``OK`` threshold with ``"profile"`` or
  ``"delta"`` as well as the two bootstrap methods; only a bootstrap method carries a
  ``BootstrapSummary``.
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema
import pytest

from robustmargin import (
    Axis,
    Censoring,
    ContinuousObservations,
    Observations,
    Status,
    fit_dose_response,
)
from robustmargin.report import from_dict, load_schema, to_dict
from robustmargin.testing import fake_threshold

_LOG = Axis(name="severity", unit="unit", scale="log")


def _counts() -> Observations:
    return Observations.from_counts(
        _LOG,
        severity=[0.0, 1.0, 2.0, 4.0],
        successes=[20, 15, 8, 2],
        trials=[20] * 4,
        outcome="outcome",
        direction="decreasing",
    )


def _continuous() -> ContinuousObservations:
    return ContinuousObservations.from_observations(
        _LOG,
        severity=[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0],
        value=[10.0, 9.5, 8.0, 8.4, 4.0, 4.5, 1.0, 1.3],
        outcome="response",
        direction="decreasing",
    )


_FITS = {
    "binomial": (_counts, {"link": "probit", "upper": 1.0, "lower": 0.0}),
    "gaussian": (_continuous, {"link": "logit", "upper": 10.0, "lower": 1.0}),
}


@pytest.mark.parametrize("model", sorted(_FITS))
class TestExplicitNoneKeepsItsExceptionType:
    def test_the_baseline_arguments_fit(self, model: str) -> None:
        build, kwargs = _FITS[model]

        assert fit_dose_response(build(), model=model, **kwargs).status in (  # type: ignore[call-overload]
            Status.OK,
            Status.NOT_CONVERGED,
            Status.SEPARATION,
        )

    def test_link_none_raises_value_error(self, model: str) -> None:
        build, kwargs = _FITS[model]

        with pytest.raises(ValueError):
            fit_dose_response(build(), model=model, **{**kwargs, "link": None})  # type: ignore[call-overload]

    @pytest.mark.parametrize("name", ["upper", "lower"])
    def test_an_asymptote_none_raises_type_error(self, model: str, name: str) -> None:
        build, kwargs = _FITS[model]

        with pytest.raises(TypeError):
            fit_dose_response(build(), model=model, **{**kwargs, name: None})  # type: ignore[call-overload]

    @pytest.mark.parametrize("name", ["link", "upper", "lower"])
    def test_a_missing_argument_is_a_type_error_that_names_it(self, model: str, name: str) -> None:
        build, kwargs = _FITS[model]
        given = {k: v for k, v in kwargs.items() if k != name}

        with pytest.raises(TypeError, match=name):
            fit_dose_response(build(), model=model, **given)  # type: ignore[call-overload]


def _json(obj: object) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(json.dumps(to_dict(obj), allow_nan=False))
    return payload


class TestFakeThresholdIntervalMethod:
    @pytest.mark.parametrize("model", ["binomial", "gaussian"])
    @pytest.mark.parametrize("method", ["profile", "delta"])
    def test_a_model_interval_method_gives_an_ok_threshold_with_no_bootstrap_summary(
        self, model: str, method: str
    ) -> None:
        t = fake_threshold(model=model, interval_method=method)

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.interval_method == method and t.bootstrap is None
        assert t.lo is not None and t.value is not None and t.hi is not None
        assert t.lo <= t.value <= t.hi

    @pytest.mark.parametrize("method", ["parametric_bootstrap", "cell_bootstrap"])
    def test_a_bootstrap_method_still_carries_a_summary(self, method: str) -> None:
        t = fake_threshold(interval_method=method)

        assert t.interval_method == method and t.bootstrap is not None

    def test_the_default_is_unchanged(self) -> None:
        assert fake_threshold(interval_method=None) == fake_threshold()

    @pytest.mark.parametrize("method", ["profile", "delta"])
    def test_the_result_round_trips_and_validates_against_the_schema(self, method: str) -> None:
        t = fake_threshold(interval_method=method)
        schema = load_schema()

        jsonschema.Draft202012Validator(
            {"$defs": schema["$defs"], "$ref": "#/$defs/Threshold"}
        ).validate(_json(t))
        assert from_dict(_json(t)) == t

    def test_an_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError):
            fake_threshold(interval_method="percentile_bootstrap")
