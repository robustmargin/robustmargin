"""Contract test: a consumer of the **isotonic** (monotone, nonparametric) fit.

The consumer holds either success counts or real-valued responses at a few held severities, wants
a threshold that assumes only monotonicity, and stores everything through ``to_dict``/``from_dict``
as JSON text. It fits with ``model="isotonic"`` (no ``link``, ``upper`` or ``lower``), reads
``Fit.cell_estimates``, asks for a ``cell_bootstrap`` threshold with a seed, and uses the fakes
from ``robustmargin.testing`` in its own tests. Only the public API surface is used.

A failing test in this file means a breaking change to the public API that a consumer depends
on. **Do not edit this test to make it pass.** Stop and ask; that failure is the signal that a
breaking change needs an explicit decision.

Names are imported inside each test, so removing or renaming one public name fails only the tests
that use it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robustmargin import ContinuousObservations, Fit, Observations, Threshold


def _counts() -> Observations:
    from robustmargin import Axis, Observations

    return Observations.from_counts(
        Axis(name="axis_1", unit="m", scale="log"),
        severity=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
        successes=[100, 98, 90, 60, 20, 5],
        trials=[100] * 6,
        outcome="success",
        direction="decreasing",
    )


def _continuous() -> ContinuousObservations:
    from robustmargin import Axis, ContinuousObservations

    return ContinuousObservations.from_summaries(
        Axis(name="axis_1", unit="m", scale="log"),
        severity=[0.0, 0.5, 1.0, 2.0, 4.0],
        mean=[10.0, 8.0, 5.0, 2.0, 1.0],
        sd=[0.5] * 5,
        n=[4] * 5,
        outcome="response",
        direction="decreasing",
    )


def _solve(fit: Fit, target: float, *, seed: int = 5) -> Threshold:
    from robustmargin import Definition, threshold

    return threshold(
        fit,
        definition=Definition.absolute(target),
        interval_method="cell_bootstrap",
        level=0.95,
        n_resamples=49,
        seed=seed,
    )


def test_both_observation_types_fit_without_a_link_or_asymptotes() -> None:
    from robustmargin import Status, fit_dose_response

    for obs in (_counts(), _continuous()):
        fit = fit_dose_response(obs, model="isotonic")

        assert fit.status is Status.OK
        assert fit.model == "isotonic" and fit.link == "none"
        assert fit.params is None and fit.covariance is None
        assert fit.cell_estimates is not None
        assert len(fit.cell_estimates) == len(fit.cells)


def test_cell_estimates_is_a_public_fit_field_and_none_on_a_parametric_fit() -> None:
    import dataclasses

    from robustmargin import Fit, fit_dose_response

    assert "cell_estimates" in {f.name for f in dataclasses.fields(Fit)}
    parametric = fit_dose_response(_counts(), model="binomial", link="probit", upper=1.0, lower=0.0)
    assert parametric.cell_estimates is None


def test_an_isotonic_overload_rejects_the_parametric_arguments() -> None:
    import pytest

    from robustmargin import fit_dose_response

    with pytest.raises(ValueError):
        fit_dose_response(_counts(), model="isotonic", link="probit")  # type: ignore[call-overload]


def test_a_parametric_model_without_its_link_raises_type_error() -> None:
    import pytest

    from robustmargin import fit_dose_response

    with pytest.raises(TypeError):
        fit_dose_response(_counts(), model="binomial", upper=1.0, lower=0.0)  # type: ignore[call-overload]


def test_an_isotonic_threshold_has_a_value_a_bootstrap_summary_and_the_requested_method() -> None:
    from robustmargin import BootstrapSummary, Censoring, Status, fit_dose_response

    fit = fit_dose_response(_counts(), model="isotonic")

    t = _solve(fit, 0.5)

    assert t.status is Status.OK and t.censoring is Censoring.NONE
    assert t.interval_method == "cell_bootstrap"
    assert t.value is not None and 2.0 < t.value < 4.0
    assert isinstance(t.bootstrap, BootstrapSummary)
    assert (t.bootstrap.n_resamples, t.bootstrap.seed) == (49, 5)
    assert t.fit is fit


def test_the_same_seed_gives_an_identical_threshold() -> None:
    from robustmargin import fit_dose_response

    fit = fit_dose_response(_continuous(), model="isotonic")

    assert _solve(fit, 6.5, seed=9) == _solve(fit, 6.5, seed=9)


def test_a_parametric_interval_method_on_an_isotonic_fit_is_refused() -> None:
    import pytest

    from robustmargin import Definition, fit_dose_response, threshold

    fit = fit_dose_response(_counts(), model="isotonic")

    for method in ("profile", "delta", "parametric_bootstrap"):
        with pytest.raises(ValueError):
            threshold(fit, definition=Definition.absolute(0.5), interval_method=method, level=0.95)


def test_an_unreachable_target_is_flagged_with_no_number() -> None:
    from robustmargin import Status, fit_dose_response

    fit = fit_dose_response(_continuous(), model="isotonic")

    t = _solve(fit, 50.0)

    assert t.status is Status.UNREACHABLE
    assert (t.value, t.lo, t.hi) == (None, None, None)
    assert t.interval_method == "exact_bound" and t.bootstrap is None


def test_a_right_censored_result_has_no_point_estimate_and_a_labelled_bound() -> None:
    from robustmargin import Axis, Censoring, Observations, fit_dose_response

    obs = Observations.from_counts(
        Axis(name="axis_1", unit="m", scale="log"),
        severity=[0.0, 1.0, 2.0, 4.0, 8.0],
        successes=[200, 200, 199, 199, 198],
        trials=[200] * 5,
        outcome="success",
        direction="decreasing",
    )

    t = _solve(fit_dose_response(obs, model="isotonic"), 0.5)

    assert t.censoring is Censoring.RIGHT
    assert t.value is None and t.lo == 8.0 and t.hi is None


def test_an_isotonic_threshold_cannot_enter_a_ratio() -> None:
    import pytest

    from robustmargin import fit_dose_response, ratio_interval

    t = _solve(fit_dose_response(_counts(), model="isotonic"), 0.5)

    with pytest.raises(ValueError):
        ratio_interval(t, t, method="log_delta", dependence="independent")


def test_the_isotonic_fakes_are_schema_valid_and_carry_what_a_consumer_renders() -> None:
    import jsonschema

    from robustmargin import BootstrapSummary, Scorecard, Status
    from robustmargin.report import load_schema, to_dict
    from robustmargin.testing import fake_fit, fake_threshold

    fit = fake_fit(model="isotonic")
    t = fake_threshold(model="isotonic")

    assert fit.model == "isotonic" and fit.status is Status.OK
    assert fit.cell_estimates is not None and fit.params is None
    assert t.status is Status.OK and t.interval_method == "cell_bootstrap"
    assert isinstance(t.bootstrap, BootstrapSummary) and t.fit.model == "isotonic"
    card = Scorecard(results=(fit, t), provenance={})
    jsonschema.Draft202012Validator(load_schema()).validate(
        json.loads(json.dumps(to_dict(card), allow_nan=False))
    )


def test_a_failed_isotonic_fake_is_not_offered() -> None:
    import pytest

    from robustmargin import Status
    from robustmargin.testing import fake_fit

    with pytest.raises(ValueError):
        fake_fit(model="isotonic", status=Status.NOT_CONVERGED)


def test_the_bootstrap_fake_kwarg_gives_each_bootstrap_method_and_a_summary() -> None:
    from robustmargin import BootstrapSummary
    from robustmargin.testing import fake_threshold

    for method in ("parametric_bootstrap", "cell_bootstrap"):
        t = fake_threshold(interval_method=method)

        assert t.interval_method == method
        assert isinstance(t.bootstrap, BootstrapSummary)
    assert fake_threshold(interval_method=None) == fake_threshold()
    assert fake_threshold().bootstrap is None


def test_every_isotonic_result_round_trips_through_json_text() -> None:
    from robustmargin import Scorecard, fit_dose_response
    from robustmargin.report import from_dict, to_dict

    results = []
    for obs, target in ((_counts(), 0.5), (_continuous(), 6.5)):
        fit = fit_dose_response(obs, model="isotonic")
        results.extend([fit, _solve(fit, target)])
    card = Scorecard(results=tuple(results), provenance={"run": "contract"})

    for result in (*results, card):
        payload = json.loads(json.dumps(to_dict(result), allow_nan=False))
        assert from_dict(payload) == result

    payload = to_dict(results[1])
    assert payload["schema_version"] == "3"
    assert payload["interval_method"] == "cell_bootstrap"
    assert payload["fit"]["model"] == "isotonic"  # type: ignore[index]
    assert payload["fit"]["link"] == "none"  # type: ignore[index]
    assert payload["bootstrap"]["seed"] == 5  # type: ignore[index]
