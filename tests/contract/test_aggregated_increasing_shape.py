"""Contract test: an aggregated, independent consumer of an **increasing** outcome.

The consumer holds per-level counts of an outcome that rises with severity (events out of trials),
fits them with ``direction="increasing"``, solves a threshold with ``absolute`` or
``baseline_multiple``, reads ``value``/``lo``/``hi``/``censoring``/``status`` and the original-scale
``baseline``, combines two independent thresholds with a ratio, and stores everything through
``to_dict``/``from_dict`` as JSON text. Only the public API surface is used -- no internal
robustmargin module is imported.

A failing test in this file means a breaking change to the public API that a consumer depends
on. **Do not edit this test to make it pass.** Stop and ask; that failure is the signal that a
breaking change needs an explicit decision.

Names are imported inside each test, as in ``test_aggregated_independent_shape.py``: removing or
renaming one public name then fails only the tests that use it, not the collection of the module.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from robustmargin import Fit, Threshold

# Per-level counts of the rising outcome: 5% at the control, up to 95% at the highest level.
_SEVERITY = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
_EVENTS = [20, 22, 35, 90, 220, 340, 380]
_TRIALS = [400] * 7


def _fit(
    *, upper: float | Literal["estimate"] = 1.0, lower: float | Literal["estimate"] = "estimate"
) -> Fit:
    from robustmargin import Axis, Observations, fit_dose_response

    axis = Axis(name="axis_1", unit="m", scale="log")
    obs = Observations.from_counts(
        axis,
        severity=_SEVERITY,
        successes=_EVENTS,
        trials=_TRIALS,
        outcome="events",
        direction="increasing",
    )
    return fit_dose_response(obs, model="binomial", link="probit", upper=upper, lower=lower)


def test_increasing_aggregated_counts_fit_without_refusal_and_keep_their_direction() -> None:
    from robustmargin import Status

    fit = _fit()

    assert fit.status is Status.OK
    assert fit.direction == "increasing"
    assert fit.params is not None
    assert fit.params["lower"].fixed is False and fit.params["upper"].fixed is True
    assert 0.0 < fit.params["lower"].value < fit.params["upper"].value <= 1.0
    # The original counts, never complemented.
    assert [(c.severity, c.successes, c.trials) for c in fit.cells] == list(
        zip(_SEVERITY, _EVENTS, _TRIALS, strict=True)
    )


def test_absolute_threshold_reads_a_bracketed_value_on_the_events_scale() -> None:
    from robustmargin import Censoring, Definition, Status, threshold

    fit = _fit()

    t = threshold(fit, definition=Definition.absolute(0.5), interval_method="profile", level=0.95)

    assert t.status is Status.OK and t.censoring is Censoring.NONE
    assert t.direction == "increasing"
    assert t.lo is not None and t.value is not None and t.hi is not None
    assert t.lo <= t.value <= t.hi
    assert t.fit is fit
    # The fitted curve rises through the target there.
    assert abs(float(fit.predict(t.value).estimate) - 0.5) < 1e-9


def test_baseline_multiple_is_the_definition_for_a_rate_that_rises_from_a_baseline() -> None:
    from robustmargin import Definition, Status, threshold

    fit = _fit()
    definition = Definition.baseline_multiple(2.0)

    t = threshold(fit, definition=definition, interval_method="profile", level=0.95)

    assert definition.kind == "baseline_multiple" and definition.value == 2.0
    assert t.status is Status.OK
    assert t.definition == definition
    assert t.value is not None and fit.params is not None
    assert abs(float(fit.predict(t.value).estimate) - 2.0 * fit.params["lower"].value) < 1e-6


def test_baseline_multiple_over_a_baseline_of_zero_raises_and_points_at_estimating_lower() -> None:
    import pytest

    from robustmargin import (
        Axis,
        Definition,
        Observations,
        Status,
        fit_dose_response,
        threshold,
    )

    obs = Observations.from_counts(
        Axis(name="axis_1", unit="m", scale="log"),
        severity=[0.0, 1.0, 2.0, 4.0, 8.0],
        successes=[0, 10, 40, 80, 95],
        trials=[100] * 5,
        outcome="events",
        direction="increasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
    assert fit.status is Status.OK

    with pytest.raises(ValueError, match='lower="estimate"'):
        threshold(
            fit,
            definition=Definition.baseline_multiple(2.0),
            interval_method="profile",
            level=0.95,
        )


def test_each_definition_refuses_the_direction_it_does_not_apply_to() -> None:
    import pytest

    from robustmargin import Axis, Definition, Observations, fit_dose_response, threshold

    fit = _fit()
    with pytest.raises(ValueError):
        threshold(
            fit,
            definition=Definition.baseline_fraction(0.5),
            interval_method="profile",
            level=0.95,
        )

    decreasing = fit_dose_response(
        Observations.from_counts(
            Axis(name="axis_1", unit="m", scale="log"),
            severity=_SEVERITY,
            successes=[n - k for k, n in zip(_EVENTS, _TRIALS, strict=True)],
            trials=_TRIALS,
            outcome="success",
            direction="decreasing",
        ),
        model="binomial",
        link="probit",
        upper="estimate",
        lower=0.0,
    )
    with pytest.raises(ValueError):
        threshold(
            decreasing,
            definition=Definition.baseline_multiple(2.0),
            interval_method="profile",
            level=0.95,
        )


def test_no_event_anywhere_is_right_censored_with_no_point_estimate() -> None:
    """Rising outcome never observed: the criterion still holds at the highest level tested. The
    result is a labelled bound, never the highest severity as a value."""
    from robustmargin import Axis, Censoring, Definition, Observations, fit_dose_response, threshold

    obs = Observations.from_counts(
        Axis(name="axis_2", unit="m/s", scale="linear"),
        severity=[0.0, 2.0, 5.0, 10.0],
        successes=[0, 0, 0, 0],
        trials=[100, 500, 500, 500],
        outcome="events",
        direction="increasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

    t = threshold(fit, definition=Definition.absolute(0.5), interval_method="profile", level=0.95)

    assert t.censoring is Censoring.RIGHT
    assert t.value is None
    assert t.hi is None
    assert t.lo == 10.0


def test_a_control_already_over_the_gate_fails_at_baseline_with_the_controls_own_rate() -> None:
    from robustmargin import Axis, Definition, Observations, Status, fit_dose_response, threshold

    obs = Observations.from_counts(
        Axis(name="axis_1", unit="m", scale="log"),
        severity=[0.0, 0.01, 0.05, 0.1],
        successes=[102, 300, 300, 300],
        trials=[200, 300, 300, 300],
        outcome="events",
        direction="increasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower="estimate")

    t = threshold(fit, definition=Definition.absolute(0.05), interval_method="profile", level=0.95)

    assert t.status is Status.FAILS_AT_BASELINE
    assert t.value is None and t.lo is None and t.hi is None
    assert t.baseline is not None
    assert (t.baseline.successes, t.baseline.trials) == (102, 200)
    assert t.baseline.lo <= t.baseline.rate <= t.baseline.hi


def test_threshold_serialises_to_json_text_and_reads_back_equal_under_schema_2() -> None:
    from robustmargin import Definition, Threshold, threshold
    from robustmargin.report import from_dict, to_dict

    t = threshold(
        _fit(),
        definition=Definition.baseline_multiple(2.0),
        interval_method="profile",
        level=0.95,
    )

    payload = json.loads(json.dumps(to_dict(t), allow_nan=False))
    reloaded = from_dict(payload)

    assert payload["schema_version"] == "2"
    assert payload["direction"] == "increasing"
    assert payload["definition"] == {"kind": "baseline_multiple", "value": 2.0}
    assert isinstance(reloaded, Threshold)
    assert reloaded == t


def test_scorecard_of_increasing_results_round_trips_and_validates_against_the_schema() -> None:
    import jsonschema

    from robustmargin import Definition, Scorecard, threshold
    from robustmargin.report import from_dict, load_schema, to_dict

    fit = _fit()
    card = Scorecard(
        results=(
            fit,
            threshold(
                fit, definition=Definition.absolute(0.5), interval_method="profile", level=0.95
            ),
        ),
        provenance={"run": "consumer-shaped"},
    )

    payload = json.loads(json.dumps(to_dict(card)))

    jsonschema.Draft202012Validator(load_schema()).validate(payload)
    assert from_dict(payload) == card


def test_two_independent_increasing_thresholds_form_a_ratio() -> None:
    from robustmargin import (
        Axis,
        Definition,
        IntervalShape,
        Observations,
        Status,
        fit_dose_response,
        ratio_interval,
        threshold,
    )

    axis = Axis(name="axis_1", unit="m", scale="log")

    def one(severity: list[float], events: list[int]) -> Threshold:
        obs = Observations.from_counts(
            axis,
            severity=severity,
            successes=events,
            trials=[100] * len(events),
            outcome="events",
            direction="increasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        return threshold(
            fit, definition=Definition.absolute(0.5), interval_method="profile", level=0.95
        )

    t_a = one([1.0, 2.0, 4.0, 8.0, 16.0], [5, 20, 50, 80, 95])
    t_b = one([1.5, 3.0, 6.0, 12.0, 24.0], [4, 18, 52, 82, 96])
    assert t_a.status is Status.OK and t_b.status is Status.OK

    r = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

    assert r.status is Status.OK
    assert isinstance(r.shape, IntervalShape)
    assert t_a.value is not None and t_b.value is not None
    assert r.estimate is not None
    assert abs(r.estimate - t_a.value / t_b.value) < 1e-9 * r.estimate
