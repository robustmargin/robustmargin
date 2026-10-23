"""Edges of a caller-fixed ``lower`` on an increasing fit, and the definition/direction pairing
that a ``Threshold`` itself enforces.

* A tiny positive fixed ``lower`` is not zero, and the complementary parametrisation cannot
  represent it: ``1 - 1e-17`` rounds to exactly 1.0, so it is refused at the call.
* For a fixed ``lower``, the curve at zero severity on a log axis is exactly that ``lower``.
* ``Threshold`` refuses ``baseline_multiple`` with a decreasing direction and ``baseline_fraction``
  with an increasing one, whether it is built directly, by ``dataclasses.replace`` or read back
  from a payload.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from robustmargin import Axis, Definition, Observations, Status, Threshold, fit_dose_response
from robustmargin.report import from_dict, to_dict
from robustmargin.testing import fake_threshold

_AXIS = Axis(name="severity", unit="unit", scale="log")
_SEVERITY = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
_EVENTS = [20, 22, 35, 90, 220, 340, 380]
_TRIALS = [400] * 7


def _fit(severity: list[float], events: list[int], trials: list[int], *, lower: float):
    obs = Observations.from_counts(
        _AXIS,
        severity=severity,
        successes=events,
        trials=trials,
        outcome="outcome",
        direction="increasing",
    )
    return fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=lower)


class TestAFixedLowerTheComplementCannotRepresent:
    """The fit runs on the complementary outcome, whose upper asymptote is ``1 - lower``. A
    ``lower`` that is not zero but is so small that ``1 - lower`` rounds to exactly 1.0 cannot be
    told apart from zero there, so it is refused at the call, alongside ``upper <= lower``."""

    @pytest.mark.parametrize("lower", [1e-17, 1e-20, 5e-17])
    def test_it_raises_at_entry(self, lower: float) -> None:
        with pytest.raises(ValueError, match="lower"):
            _fit(_SEVERITY, _EVENTS, _TRIALS, lower=lower)

    def test_it_raises_whether_or_not_the_control_has_events(self) -> None:
        with pytest.raises(ValueError, match="lower"):
            _fit(_SEVERITY[1:], [0, *_EVENTS[2:]], _TRIALS[1:], lower=1e-17)

    def test_a_lower_the_complement_can_represent_is_accepted_and_kept(self) -> None:
        fit = _fit(_SEVERITY, _EVENTS, _TRIALS, lower=1e-9)

        assert fit.status is not Status.CONTROL_INCOMPATIBLE
        if fit.status is Status.OK:
            assert fit.params is not None
            assert fit.params["lower"].fixed is True

    def test_an_exact_zero_lower_still_is_control_incompatible_over_a_control_with_events(
        self,
    ) -> None:
        fit = _fit(_SEVERITY, _EVENTS, _TRIALS, lower=0.0)

        assert fit.status is Status.CONTROL_INCOMPATIBLE

    def test_an_exact_zero_lower_is_accepted_over_a_control_without_events(self) -> None:
        fit = _fit([0.0, 1.0, 2.0, 4.0, 8.0], [0, 10, 40, 80, 95], [100] * 5, lower=0.0)

        assert fit.status is Status.OK


class TestCurveAtZeroSeverityIsTheFixedLower:
    @pytest.mark.parametrize("lower", [0.1, 0.05, 0.02])
    def test_predict_zero_returns_exactly_the_fixed_lower(self, lower: float) -> None:
        fit = _fit(_SEVERITY, _EVENTS, _TRIALS, lower=lower)
        assert fit.status is Status.OK and fit.params is not None

        prediction = fit.predict(0.0)

        assert fit.params["lower"].value == lower
        assert float(prediction.estimate) == fit.params["lower"].value
        assert float(prediction.lo) <= float(prediction.estimate) <= float(prediction.hi)

    def test_predict_zero_and_a_vanishing_severity_agree(self) -> None:
        fit = _fit(_SEVERITY, _EVENTS, _TRIALS, lower=0.1)

        at_zero = float(fit.predict(0.0).estimate)
        near_zero = float(fit.predict(1e-12).estimate)

        assert near_zero == pytest.approx(at_zero, abs=1e-9)


def _payload(t: Threshold) -> dict[str, Any]:
    payload: dict[str, Any] = to_dict(t)
    return payload


class TestThresholdEnforcesTheDefinitionDirectionPairing:
    def test_baseline_multiple_with_a_decreasing_direction_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="baseline_multiple"):
            dataclasses.replace(fake_threshold(), definition=Definition.baseline_multiple(2.0))

    def test_baseline_fraction_with_an_increasing_direction_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="baseline_fraction"):
            dataclasses.replace(
                fake_threshold(direction="increasing"),
                definition=Definition.baseline_fraction(0.5),
            )

    def test_the_matching_pairings_are_accepted(self) -> None:
        increasing = dataclasses.replace(
            fake_threshold(direction="increasing"), definition=Definition.baseline_multiple(2.0)
        )
        decreasing = dataclasses.replace(
            fake_threshold(), definition=Definition.baseline_fraction(0.5)
        )

        assert increasing.definition.kind == "baseline_multiple"
        assert decreasing.definition.kind == "baseline_fraction"

    @pytest.mark.parametrize("direction", ["decreasing", "increasing"])
    def test_absolute_and_relative_apply_in_both_directions(self, direction: str) -> None:
        for definition in (Definition.absolute(0.5), Definition.relative(0.5)):
            t = dataclasses.replace(fake_threshold(direction=direction), definition=definition)
            assert t.definition == definition

    def test_from_dict_rejects_baseline_multiple_on_a_decreasing_payload(self) -> None:
        payload = _payload(fake_threshold())
        payload["definition"] = {"kind": "baseline_multiple", "value": 2.0}

        with pytest.raises(ValueError, match="baseline_multiple"):
            from_dict(payload)

    def test_from_dict_rejects_baseline_fraction_on_an_increasing_payload(self) -> None:
        payload = _payload(fake_threshold(direction="increasing"))
        payload["definition"] = {"kind": "baseline_fraction", "value": 0.5}

        with pytest.raises(ValueError, match="baseline_fraction"):
            from_dict(payload)


_LINEAR = Axis(name="severity", unit="unit", scale="linear")


class TestThresholdRefusesBaselineMultipleOnALinearAxis:
    """On a linear axis the lower asymptote is not the performance at zero severity, so a
    ``Threshold`` cannot carry ``baseline_multiple`` there, however it is built."""

    @staticmethod
    def _linear_increasing() -> Threshold:
        base = fake_threshold(direction="increasing")
        fit = dataclasses.replace(base.fit, axis=_LINEAR)
        return dataclasses.replace(base, axis=_LINEAR, fit=fit)

    def test_direct_construction_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="linear"):
            dataclasses.replace(
                self._linear_increasing(), definition=Definition.baseline_multiple(2.0)
            )

    def test_the_same_definition_on_a_log_axis_is_accepted(self) -> None:
        t = dataclasses.replace(
            fake_threshold(direction="increasing"), definition=Definition.baseline_multiple(2.0)
        )

        assert t.axis.scale == "log"

    def test_other_definitions_are_accepted_on_a_linear_axis(self) -> None:
        t = dataclasses.replace(self._linear_increasing(), definition=Definition.relative(0.5))

        assert t.axis.scale == "linear"

    def test_from_dict_rejects_it(self) -> None:
        payload = _payload(fake_threshold(direction="increasing"))
        payload["axis"]["scale"] = "linear"
        payload["fit"]["axis"]["scale"] = "linear"
        payload["definition"] = {"kind": "baseline_multiple", "value": 2.0}

        with pytest.raises(ValueError, match="linear"):
            from_dict(payload)
