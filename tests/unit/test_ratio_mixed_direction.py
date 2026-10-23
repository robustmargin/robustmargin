"""``ratio_interval`` has no direction guard: an increasing-outcome threshold and a
decreasing-outcome threshold on the same axis, with the same definition, form a valid ratio.

The ratio of severities does not depend on which way each outcome runs, so the pinned behaviour is
that a mixed pair gives exactly what the same pair gives once the increasing threshold is replaced
by its mirrored decreasing view (``absolute(0.5)`` mirrors to ``absolute(0.5)``).
"""

from __future__ import annotations

import pytest

from robustmargin import (
    Axis,
    Definition,
    Observations,
    Status,
    Threshold,
    fit_dose_response,
    ratio_interval,
    threshold,
)

_LEVEL = 0.95
_AXIS = Axis(name="severity", unit="unit", scale="log")
_SEVERITY_A = [1.0, 2.0, 4.0, 8.0, 16.0]
_EVENTS_A = [5, 20, 50, 80, 95]
_SEVERITY_B = [1.5, 3.0, 6.0, 12.0, 24.0]
_SUCCESSES_B = [95, 80, 50, 20, 5]


def _threshold(severity: list[float], counts: list[int], direction: str) -> Threshold:
    obs = Observations.from_counts(
        _AXIS,
        severity=severity,
        successes=counts,
        trials=[100] * len(counts),
        outcome="outcome",
        direction=direction,
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
    assert fit.status is Status.OK
    t = threshold(fit, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL)
    assert t.status is Status.OK
    return t


@pytest.mark.parametrize("method", ["fieller", "log_delta"])
class TestMixedDirectionRatio:
    def test_the_ratio_is_valid_and_is_the_ratio_of_the_two_values(self, method: str) -> None:
        t_inc = _threshold(_SEVERITY_A, _EVENTS_A, "increasing")
        t_dec = _threshold(_SEVERITY_B, _SUCCESSES_B, "decreasing")
        assert t_inc.direction == "increasing" and t_dec.direction == "decreasing"

        r = ratio_interval(t_inc, t_dec, method=method, dependence="independent", level=_LEVEL)

        assert r.status is Status.OK
        assert t_inc.value is not None and t_dec.value is not None
        assert r.estimate == pytest.approx(t_inc.value / t_dec.value, rel=1e-12)
        assert r.lo is not None and r.hi is not None and r.estimate is not None
        assert r.lo <= r.estimate <= r.hi

    def test_it_equals_the_ratio_with_the_increasing_side_replaced_by_its_mirror(
        self, method: str
    ) -> None:
        t_inc = _threshold(_SEVERITY_A, _EVENTS_A, "increasing")
        t_mirror = _threshold(_SEVERITY_A, [100 - k for k in _EVENTS_A], "decreasing")
        t_dec = _threshold(_SEVERITY_B, _SUCCESSES_B, "decreasing")

        mixed = ratio_interval(t_inc, t_dec, method=method, dependence="independent", level=_LEVEL)
        reference = ratio_interval(
            t_mirror, t_dec, method=method, dependence="independent", level=_LEVEL
        )

        assert mixed.status is reference.status
        assert mixed.shape is reference.shape
        assert mixed.censoring is reference.censoring
        assert mixed.estimate == pytest.approx(reference.estimate, rel=1e-9)
        assert mixed.lo == pytest.approx(reference.lo, rel=1e-9)
        assert mixed.hi == pytest.approx(reference.hi, rel=1e-9)

    def test_order_does_not_matter_for_validity(self, method: str) -> None:
        t_inc = _threshold(_SEVERITY_A, _EVENTS_A, "increasing")
        t_dec = _threshold(_SEVERITY_B, _SUCCESSES_B, "decreasing")

        r = ratio_interval(t_dec, t_inc, method=method, dependence="independent", level=_LEVEL)

        assert r.status is Status.OK
        assert t_inc.value is not None and t_dec.value is not None
        assert r.estimate == pytest.approx(t_dec.value / t_inc.value, rel=1e-12)
