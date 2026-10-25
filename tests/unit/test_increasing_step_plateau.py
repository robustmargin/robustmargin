"""The step-plateau rule under the direction mirror.

An increasing fit is the decreasing fit of the complement counts with swapped asymptote
arguments, so the step rule (a fit whose best log-likelihood does not exceed the supremum of a
step between adjacent tested levels is ``NOT_CONVERGED``) must give the same status, and the same
reason, on both sides. Each design is one whose supremum equals the saturated likelihood, so
``NOT_CONVERGED`` is a step plateau and not an optimizer stop. Most are the pinned designs of
``tests/unit/test_models_step_plateau.py``, which checks this; the zero-control design justifies
it in place.
"""

from __future__ import annotations

import pytest

from robustmargin import Axis, Fit, Observations, Status, fit_dose_response

_STEP_REASON = "a step between adjacent tested levels"
_LOG = Axis(name="severity", unit="unit", scale="log")
_LINEAR = Axis(name="severity", unit="unit", scale="linear")
_SEVERITY_7 = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def _flip(v: float | str) -> float | str:
    return v if isinstance(v, str) else 1.0 - v


def _pair(
    axis: Axis,
    severity: list[float],
    decreasing_successes: list[int],
    trials: list[int],
    link: str,
    upper: float | str,
    lower: float | str,
) -> tuple[Fit, Fit]:
    """Fit the increasing data and its complement (mirrored arguments).

    ``decreasing_successes`` are the counts of the decreasing design; the increasing data are
    their complements, so the increasing fit is the one the mirror reduces to it.
    """
    inc_counts = [n - k for k, n in zip(decreasing_successes, trials, strict=True)]
    inc = Observations.from_counts(
        axis,
        severity=severity,
        successes=inc_counts,
        trials=trials,
        outcome="outcome",
        direction="increasing",
    )
    dec = Observations.from_counts(
        axis,
        severity=severity,
        successes=decreasing_successes,
        trials=trials,
        outcome="outcome",
        direction="decreasing",
    )
    # The increasing fit's upper/lower are the decreasing fit's flipped lower/upper.
    inc_fit = fit_dose_response(
        inc, model="binomial", link=link, upper=_flip(lower), lower=_flip(upper)
    )
    dec_fit = fit_dose_response(dec, model="binomial", link=link, upper=upper, lower=lower)
    return inc_fit, dec_fit


def _has_step(fit: Fit) -> bool:
    return any(_STEP_REASON in w for w in fit.warnings)


def _assert_both_step_plateaus(inc: Fit, dec: Fit) -> None:
    for fit in (inc, dec):
        assert fit.status is Status.NOT_CONVERGED
        assert fit.params is None
        assert fit.log_likelihood is None
        assert _has_step(fit), fit.warnings


@pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
def test_exact_step_with_both_asymptotes_estimated_is_a_step_plateau_in_both_directions(
    link: str,
) -> None:
    inc, dec = _pair(
        _LOG, _SEVERITY_7, [190, 190, 190, 190, 10, 10, 10], [200] * 7, link,
        "estimate", "estimate",
    )  # fmt: skip

    _assert_both_step_plateaus(inc, dec)


@pytest.mark.parametrize("link", ["probit", "logit"])
def test_step_with_an_intermediate_level_is_a_step_plateau_in_both_directions(link: str) -> None:
    inc, dec = _pair(
        _LOG, _SEVERITY_7, [190, 190, 190, 100, 10, 10, 10], [200] * 7, link,
        "estimate", "estimate",
    )  # fmt: skip

    _assert_both_step_plateaus(inc, dec)


def test_step_with_fixed_asymptotes_other_than_one_and_zero_is_a_plateau_in_both_directions() -> (
    None
):
    inc, dec = _pair(_LOG, [1.0, 2.0, 3.0, 4.0], [90, 90, 30, 30], [100] * 4, "cloglog", 0.9, 0.3)

    _assert_both_step_plateaus(inc, dec)


@pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
def test_step_with_one_estimated_and_one_fixed_asymptote_agrees_in_both_directions(
    link: str,
) -> None:
    inc, dec = _pair(
        _LOG, _SEVERITY_7, [190, 190, 190, 190, 10, 10, 10], [200] * 7, link, 0.95, "estimate"
    )

    assert inc.status is dec.status
    assert _has_step(inc) == _has_step(dec)


def test_step_with_a_zero_severity_control_on_a_log_axis_is_a_plateau_in_both_directions() -> None:
    # In the decreasing frame the observed rates are 0.9 at x in {0, 1, 2} and 0.3 at x in {3, 4},
    # equal to the fixed asymptotes, so a step between levels 2 and 3 attains the saturated
    # likelihood. Any finite slope puts P(1), P(2) strictly below 0.9 and P(3), P(4) strictly
    # above 0.3, so the supremum is approached as s -> 0 but never reached.
    inc, dec = _pair(
        _LOG, [0.0, 1.0, 2.0, 3.0, 4.0], [90, 90, 90, 30, 30], [100] * 5, "cloglog", 0.9, 0.3
    )

    assert inc.status is dec.status
    assert _has_step(inc) == _has_step(dec)
    _assert_both_step_plateaus(inc, dec)


@pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
def test_interior_design_is_ok_in_both_directions_with_the_same_curve(link: str) -> None:
    inc, dec = _pair(
        _LOG,
        [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        [380, 378, 365, 310, 180, 60, 20],
        [400] * 7,
        link,
        "estimate",
        "estimate",
    )

    assert inc.status is Status.OK and dec.status is Status.OK
    assert not _has_step(inc) and not _has_step(dec)
    assert inc.params is not None and dec.params is not None
    for name in ("mu", "s"):
        assert inc.params[name].value == pytest.approx(dec.params[name].value, rel=1e-9)
    assert inc.log_likelihood == pytest.approx(dec.log_likelihood, rel=1e-9)


def test_step_reason_on_an_increasing_fit_prints_no_logit_scale_parameter() -> None:
    inc, _ = _pair(
        _LOG, _SEVERITY_7, [190, 190, 190, 190, 10, 10, 10], [200] * 7, "logit",
        "estimate", "estimate",
    )  # fmt: skip

    assert not any("logit-scale" in w for w in inc.warnings), inc.warnings
