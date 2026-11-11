"""Contract test: a consumer of a **continuous** response on a log axis with a zero control.

The consumer holds one real-valued response per run (a score that falls, or an error that rises,
as severity grows), fits it with ``model="gaussian"``, solves thresholds with ``relative``,
``absolute`` and (rising series) ``baseline_multiple``, with the profile and the delta interval,
combines two series with a ratio, and stores everything through ``to_dict``/``from_dict`` as JSON
text. Only the public API surface is used -- no internal robustmargin module is imported.

A failing test in this file means a breaking change to the public API that a consumer depends
on. **Do not edit this test to make it pass.** Stop and ask; that failure is the signal that a
breaking change needs an explicit decision.

Names are imported inside each test, so removing or renaming one public name fails only the tests
that use it.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robustmargin import ContinuousObservations, Fit, Threshold

_SEVERITY_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
_REPLICATES = 4


def _decreasing_values(seed: int, midpoint: float) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    severity = [x for x in _SEVERITY_LEVELS for _ in range(_REPLICATES)]
    x = np.asarray(severity)
    with np.errstate(divide="ignore"):
        z = (np.log(x) - math.log(midpoint)) / 0.5
    share = np.where(x == 0.0, 0.0, 1.0 / (1.0 + np.exp(-z)))
    return severity, (10.0 - 9.0 * share + rng.normal(0.0, 0.3, size=x.size)).tolist()


def _observations(
    *, direction: str = "decreasing", seed: int = 11, midpoint: float = 0.8
) -> ContinuousObservations:
    from robustmargin import Axis, ContinuousObservations

    severity, value = _decreasing_values(seed, midpoint)
    if direction == "increasing":
        value = [11.0 - v for v in value]  # rises from about 1 to about 10
    return ContinuousObservations.from_observations(
        Axis(name="axis_1", unit="m", scale="log"),
        severity=severity,
        value=value,
        outcome="response",
        direction=direction,
    )


def _fit(**kwargs: object) -> Fit:
    from robustmargin import fit_dose_response

    obs = _observations(**kwargs)  # type: ignore[arg-type]
    return fit_dose_response(
        obs, model="gaussian", link="logit", upper="estimate", lower="estimate"
    )


def test_continuous_values_fit_without_refusal_and_keep_their_direction() -> None:
    from robustmargin import Status

    for direction in ("decreasing", "increasing"):
        fit = _fit(direction=direction)

        assert fit.status is Status.OK
        assert fit.model == "gaussian" and fit.direction == direction
        assert fit.params is not None and fit.residual is not None
        assert fit.residual.df > 0 and fit.residual.sigma > 0.0
        assert fit.params["lower"].value < fit.params["upper"].value
        assert fit.cells and all(c.n == _REPLICATES for c in fit.cells)  # type: ignore[union-attr]


def test_aggregated_summaries_are_the_same_series_as_the_raw_values() -> None:
    from robustmargin import Axis, ContinuousObservations, Status, fit_dose_response

    raw = _observations()
    cells = raw.cells()
    summaries = ContinuousObservations.from_summaries(
        Axis(name="axis_1", unit="m", scale="log"),
        severity=[c.severity for c in cells],
        mean=[c.mean for c in cells],
        sd=[c.sd for c in cells],
        n=[c.n for c in cells],
        outcome="response",
        direction="decreasing",
    )

    a = fit_dose_response(raw, model="gaussian", link="logit", upper="estimate", lower="estimate")
    b = fit_dose_response(
        summaries, model="gaussian", link="logit", upper="estimate", lower="estimate"
    )

    assert a.status is Status.OK and b.status is Status.OK
    assert a.params is not None and b.params is not None
    assert abs(a.params["mu"].value - b.params["mu"].value) < 1e-8


def test_a_decreasing_series_gives_a_bracketed_profile_and_delta_threshold() -> None:
    from robustmargin import Censoring, Definition, Status, threshold

    fit = _fit()

    for method in ("profile", "delta"):
        t = threshold(fit, definition=Definition.relative(0.5), interval_method=method, level=0.95)

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.interval_method == method and t.direction == "decreasing"
        assert t.lo is not None and t.value is not None and t.hi is not None
        assert 0.0 < t.lo < t.value < t.hi
        assert t.fit is fit
        # Midpoint of a 0.8-centred curve, read in the response's own units.
        assert abs(t.value - 0.8) < 0.15


def test_an_absolute_target_is_in_response_units() -> None:
    from robustmargin import Definition, Status, threshold

    fit = _fit()

    t = threshold(fit, definition=Definition.absolute(5.5), interval_method="profile", level=0.95)

    assert t.status is Status.OK and t.value is not None
    assert abs(float(fit.predict(t.value).estimate) - 5.5) < 1e-8


def test_a_rising_series_gives_a_baseline_multiple_threshold_on_the_response_scale() -> None:
    from robustmargin import Definition, Status, threshold

    fit = _fit(direction="increasing")

    t = threshold(
        fit, definition=Definition.baseline_multiple(3.0), interval_method="delta", level=0.95
    )

    assert t.status is Status.OK and t.direction == "increasing"
    assert t.lo is not None and t.value is not None and t.hi is not None
    assert t.lo <= t.value <= t.hi
    assert fit.params is not None
    assert abs(float(fit.predict(t.value).estimate) - 3.0 * fit.params["lower"].value) < 1e-8


def test_an_unreachable_level_is_flagged_with_no_number() -> None:
    from robustmargin import Definition, Status, threshold

    t = threshold(
        _fit(), definition=Definition.absolute(500.0), interval_method="profile", level=0.95
    )

    assert t.status is Status.UNREACHABLE
    assert (t.value, t.lo, t.hi) == (None, None, None)


def test_a_failed_continuous_fit_is_flagged_and_carries_no_number() -> None:
    from robustmargin import (
        Axis,
        ContinuousObservations,
        Definition,
        Status,
        fit_dose_response,
        threshold,
    )

    obs = ContinuousObservations.from_observations(
        Axis(name="axis_1", unit="m", scale="log"),
        severity=[0.5, 1.0, 2.0, 4.0],
        value=[9.0, 7.0, 3.0, 1.5],
        outcome="response",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="gaussian", link="logit", upper="estimate", lower="estimate")

    assert fit.status is Status.NOT_CONVERGED and fit.params is None
    t = threshold(fit, definition=Definition.relative(0.5), interval_method="profile", level=0.95)
    assert t.status is Status.NOT_CONVERGED
    assert (t.value, t.lo, t.hi, t.grid) == (None, None, None, None)


def test_two_independent_series_combine_in_a_ratio() -> None:
    from robustmargin import Definition, IntervalShape, Status, ratio_interval, threshold

    def solve(fit: Fit) -> Threshold:
        return threshold(
            fit, definition=Definition.relative(0.5), interval_method="profile", level=0.95
        )

    t_a = solve(_fit(seed=1, midpoint=0.8))
    t_b = solve(_fit(seed=2, midpoint=0.4))
    assert t_a.status is Status.OK and t_b.status is Status.OK

    for method in ("fieller", "log_delta"):
        r = ratio_interval(t_a, t_b, method=method, dependence="independent")

        assert r.status is Status.OK and r.shape is IntervalShape.BOUNDED
        assert t_a.value is not None and t_b.value is not None
        assert r.estimate is not None and r.lo is not None and r.hi is not None
        assert abs(r.estimate - t_a.value / t_b.value) < 1e-9 * r.estimate
        assert r.lo < r.estimate < r.hi
        assert abs(r.estimate - 2.0) < 0.6


def test_every_continuous_result_round_trips_through_json_text() -> None:
    from robustmargin import Definition, Scorecard, ratio_interval, threshold
    from robustmargin.report import from_dict, to_dict

    fit = _fit()
    t_a = threshold(fit, definition=Definition.relative(0.5), interval_method="profile", level=0.95)
    t_b = threshold(
        _fit(seed=2, midpoint=0.4),
        definition=Definition.relative(0.5),
        interval_method="delta",
        level=0.95,
    )
    ratio = ratio_interval(t_a, t_b, method="log_delta", dependence="independent")
    card = Scorecard(results=(fit, t_a, t_b, ratio), provenance={"run": "contract"})

    for result in (fit, t_a, t_b, ratio, card):
        payload = json.loads(json.dumps(to_dict(result), allow_nan=False))
        assert from_dict(payload) == result

    payload = to_dict(t_a)
    assert payload["interval_method"] == "profile"
    assert payload["definition"] == {"kind": "relative", "value": 0.5}
    assert payload["level"] == 0.95
    assert payload["schema_version"] == "2"
    assert payload["fit"]["model"] == "gaussian"  # type: ignore[index]
    assert payload["fit"]["residual"]["df"] == fit.residual.df  # type: ignore[index, union-attr]
