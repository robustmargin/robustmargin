"""Contract test: a consumer that asks ``threshold()`` for a bootstrap interval.

The consumer fits counts or a continuous response, asks for ``"parametric_bootstrap"`` or
``"cell_bootstrap"`` with a replicate count and a seed, reads the summary the result carries
(``Threshold.bootstrap``), and relies on the argument checks. Only the public API surface is used:
no internal robustmargin module is imported.

A failing test in this file means a breaking change to the public API that a consumer depends
on. **Do not edit this test to make it pass.** Stop and ask; that failure is the signal that a
breaking change needs an explicit decision.

Names are imported inside each test, so removing or renaming one public name fails only the tests
that use it.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.stats import norm

if TYPE_CHECKING:
    from robustmargin import Fit

_LEVELS = [0.0, *np.logspace(-1.0, 1.0, 10).tolist()]
_REPLICATES = 29


def _counts_fit(*, counts: tuple[list[int], list[int]] | None = None) -> Fit:
    from robustmargin import Axis, Observations, fit_dose_response

    if counts is None:
        rng = np.random.default_rng(20261114)
        x = np.asarray(_LEVELS)
        with np.errstate(divide="ignore"):
            z = (np.log(x) - math.log(1.5)) / 0.5
        p = np.where(x == 0.0, 1.0, 1.0 - norm.cdf(z))
        trials = [400] * len(_LEVELS)
        successes = rng.binomial(400, p).tolist()
    else:
        successes, trials = counts
    obs = Observations.from_counts(
        Axis(name="axis_1", unit="m", scale="log"),
        severity=_LEVELS[: len(trials)],
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )
    return fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)


def _gaussian_fit() -> Fit:
    from robustmargin import Axis, ContinuousObservations, fit_dose_response

    rng = np.random.default_rng(20261115)
    levels = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
    severity = [x for x in levels for _ in range(4)]
    x = np.asarray(severity)
    with np.errstate(divide="ignore"):
        z = (np.log(x) - math.log(0.8)) / 0.5
    share = np.where(x == 0.0, 0.0, 1.0 / (1.0 + np.exp(-z)))
    value = 10.0 - 9.0 * share + rng.normal(0.0, 0.3, size=x.size)
    obs = ContinuousObservations.from_observations(
        Axis(name="axis_1", unit="m", scale="log"),
        severity=severity,
        value=value.tolist(),
        outcome="response",
        direction="decreasing",
    )
    return fit_dose_response(obs, model="gaussian", link="logit", upper=10.0, lower=1.0)


def _isotonic_fit() -> Fit:
    from robustmargin import Axis, Observations, fit_dose_response

    obs = Observations.from_counts(
        Axis(name="axis_1", unit="m", scale="log"),
        severity=[0.0, 0.1, 0.2, 0.4, 0.8, 1.6],
        successes=[200, 190, 170, 110, 40, 10],
        trials=[200] * 6,
        outcome="success",
        direction="decreasing",
    )
    return fit_dose_response(obs, model="isotonic")


def _solve(fit: Fit, method: str, **kwargs: object):  # type: ignore[no-untyped-def]
    from robustmargin import Definition, threshold

    return threshold(
        fit,
        definition=Definition.absolute(0.5),
        interval_method=method,
        level=0.95,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------------------------
# BootstrapSummary
# ---------------------------------------------------------------------------------------------


def test_bootstrap_summary_is_exported_with_its_five_fields() -> None:
    from robustmargin import BootstrapSummary

    s = BootstrapSummary(n_resamples=999, failed=3, unreachable_low=2, unreachable_high=5, seed=7)

    assert (s.n_resamples, s.failed, s.unreachable_low, s.unreachable_high, s.seed) == (
        999,
        3,
        2,
        5,
        7,
    )
    assert [f.name for f in dataclasses.fields(BootstrapSummary)] == [
        "n_resamples",
        "failed",
        "unreachable_low",
        "unreachable_high",
        "seed",
    ]


def test_bootstrap_summary_is_frozen() -> None:
    from robustmargin import BootstrapSummary

    s = BootstrapSummary(n_resamples=10, failed=0, unreachable_low=0, unreachable_high=0, seed=1)

    with pytest.raises(dataclasses.FrozenInstanceError):
        s.failed = 1  # type: ignore[misc]


def test_bootstrap_summary_compares_by_value() -> None:
    from robustmargin import BootstrapSummary

    a = BootstrapSummary(n_resamples=10, failed=1, unreachable_low=0, unreachable_high=2, seed=1)
    b = BootstrapSummary(n_resamples=10, failed=1, unreachable_low=0, unreachable_high=2, seed=1)

    assert a == b


def test_bootstrap_summary_accepts_a_zero_seed_and_counts_that_fill_the_resamples() -> None:
    from robustmargin import BootstrapSummary

    s = BootstrapSummary(n_resamples=10, failed=4, unreachable_low=3, unreachable_high=3, seed=0)

    assert s.seed == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_resamples": 0},
        {"n_resamples": -1},
        {"failed": -1},
        {"unreachable_low": -1},
        {"unreachable_high": -1},
        {"seed": -1},
        {"failed": 11},
        {"failed": 6, "unreachable_low": 3, "unreachable_high": 2},
        {"unreachable_low": 11},
        {"unreachable_high": 11},
    ],
)
def test_bootstrap_summary_rejects_out_of_range_counts(overrides: dict[str, int]) -> None:
    from robustmargin import BootstrapSummary

    fields = {
        "n_resamples": 10,
        "failed": 0,
        "unreachable_low": 0,
        "unreachable_high": 0,
        "seed": 1,
    }

    with pytest.raises(ValueError):
        BootstrapSummary(**{**fields, **overrides})


@pytest.mark.parametrize("name", ["n_resamples", "failed", "unreachable_low", "unreachable_high"])
@pytest.mark.parametrize("bad", [True, 1.0, 2.5, "3", None])
def test_bootstrap_summary_rejects_a_bool_or_non_integer_count(name: str, bad: object) -> None:
    from robustmargin import BootstrapSummary

    fields: dict[str, object] = {
        "n_resamples": 10,
        "failed": 0,
        "unreachable_low": 0,
        "unreachable_high": 0,
        "seed": 1,
    }
    fields[name] = bad

    with pytest.raises(ValueError):
        BootstrapSummary(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [True, 1.0, "3", None])
def test_bootstrap_summary_rejects_a_bool_or_non_integer_seed(bad: object) -> None:
    from robustmargin import BootstrapSummary

    with pytest.raises(ValueError):
        BootstrapSummary(
            n_resamples=10,
            failed=0,
            unreachable_low=0,
            unreachable_high=0,
            seed=bad,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------------------------
# Threshold.bootstrap is set exactly when a bootstrap ran
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["parametric_bootstrap", "cell_bootstrap"])
def test_a_bootstrap_threshold_carries_the_summary_and_the_requested_method(method: str) -> None:
    from robustmargin import BootstrapSummary, Censoring, Status

    t = _solve(_counts_fit(), method, n_resamples=_REPLICATES, seed=11)

    assert t.status is Status.OK and t.censoring is Censoring.NONE
    assert t.interval_method == method
    assert isinstance(t.bootstrap, BootstrapSummary)
    assert t.bootstrap.n_resamples == _REPLICATES and t.bootstrap.seed == 11
    assert t.lo is not None and t.value is not None and t.hi is not None
    assert t.lo <= t.value <= t.hi


def test_a_gaussian_bootstrap_threshold_carries_the_summary() -> None:
    from robustmargin import BootstrapSummary, Status

    t = _solve(_gaussian_fit(), "parametric_bootstrap", n_resamples=_REPLICATES, seed=3)

    assert t.status is Status.OK and t.interval_method == "parametric_bootstrap"
    assert isinstance(t.bootstrap, BootstrapSummary) and t.bootstrap.n_resamples == _REPLICATES


@pytest.mark.parametrize("method", ["profile", "delta"])
def test_a_model_interval_threshold_has_no_bootstrap_summary(method: str) -> None:
    t = _solve(_counts_fit(), method)

    assert t.bootstrap is None


def test_the_bootstrap_field_defaults_to_none_and_is_last() -> None:
    from robustmargin import Threshold

    fields = dataclasses.fields(Threshold)

    assert fields[-1].name == "bootstrap"
    assert fields[-1].default is None


def test_a_right_censored_result_has_no_bootstrap_summary() -> None:
    from robustmargin import Censoring

    fit = _counts_fit(counts=([200, 200, 200, 200, 200], [200] * 5))  # separated: no failures

    t = _solve(fit, "parametric_bootstrap", n_resamples=_REPLICATES, seed=1)

    assert t.censoring is Censoring.RIGHT and t.value is None
    assert t.interval_method == "exact_bound"
    assert t.bootstrap is None


def test_a_failed_fit_result_has_no_bootstrap_summary() -> None:
    from robustmargin import Status

    fit = _counts_fit(counts=([20, 0, 0, 0, 0], [200] * 5))

    t = _solve(fit, "parametric_bootstrap", n_resamples=_REPLICATES, seed=1)

    assert t.status is Status.CONTROL_INCOMPATIBLE
    assert t.value is None
    assert t.bootstrap is None


def test_an_unreachable_result_has_no_bootstrap_summary() -> None:
    from robustmargin import Definition, Status, threshold

    t = threshold(
        _counts_fit(),
        definition=Definition.absolute(1.5),
        interval_method="parametric_bootstrap",
        level=0.95,
        n_resamples=_REPLICATES,
        seed=1,
    )

    assert t.status is Status.UNREACHABLE and t.value is None
    assert t.bootstrap is None


def test_the_same_seed_gives_an_equal_result_and_the_recorded_seed_replays_it() -> None:
    fit = _counts_fit()

    first = _solve(fit, "parametric_bootstrap", n_resamples=_REPLICATES, seed=5)
    second = _solve(fit, "parametric_bootstrap", n_resamples=_REPLICATES, seed=5)
    unseeded = _solve(fit, "parametric_bootstrap", n_resamples=_REPLICATES)

    assert first == second
    assert unseeded.bootstrap is not None and isinstance(unseeded.bootstrap.seed, int)
    replay = _solve(
        fit, "parametric_bootstrap", n_resamples=_REPLICATES, seed=unseeded.bootstrap.seed
    )
    assert replay == unseeded


# ---------------------------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.0, 19.5, "19", [19]])
def test_n_resamples_must_be_a_positive_non_bool_int(bad: object) -> None:
    with pytest.raises(ValueError):
        _solve(_counts_fit(), "parametric_bootstrap", n_resamples=bad, seed=1)


@pytest.mark.parametrize("bad", [-1, True, False, 1.0, 2.5, "7", [7]])
def test_seed_must_be_a_non_negative_non_bool_int(bad: object) -> None:
    with pytest.raises(ValueError):
        _solve(_counts_fit(), "parametric_bootstrap", n_resamples=_REPLICATES, seed=bad)


def test_a_zero_seed_is_accepted() -> None:
    t = _solve(_counts_fit(), "parametric_bootstrap", n_resamples=_REPLICATES, seed=0)

    assert t.bootstrap is not None and t.bootstrap.seed == 0


@pytest.mark.parametrize("method", ["profile", "delta"])
def test_n_resamples_with_a_model_interval_raises(method: str) -> None:
    with pytest.raises(ValueError):
        _solve(_counts_fit(), method, n_resamples=_REPLICATES)


@pytest.mark.parametrize("method", ["profile", "delta"])
def test_seed_with_a_model_interval_raises(method: str) -> None:
    with pytest.raises(ValueError):
        _solve(_counts_fit(), method, seed=1)


def test_n_resamples_and_seed_default_to_none_for_a_model_interval() -> None:
    t = _solve(_counts_fit(), "profile", n_resamples=None, seed=None)

    assert t.interval_method == "profile"


def test_cell_bootstrap_on_a_gaussian_fit_raises() -> None:
    with pytest.raises(ValueError):
        _solve(_gaussian_fit(), "cell_bootstrap", n_resamples=_REPLICATES, seed=1)


@pytest.mark.parametrize("method", ["profile", "delta", "parametric_bootstrap"])
def test_an_isotonic_fit_accepts_only_cell_bootstrap(method: str) -> None:
    fit = _isotonic_fit()

    with pytest.raises(ValueError):
        _solve(fit, method)


def test_an_isotonic_fit_accepts_cell_bootstrap() -> None:
    t = _solve(_isotonic_fit(), "cell_bootstrap", n_resamples=_REPLICATES, seed=1)

    assert t.interval_method == "cell_bootstrap" and t.bootstrap is not None


def test_an_unknown_interval_method_still_raises() -> None:
    with pytest.raises(ValueError):
        _solve(_counts_fit(), "percentile_bootstrap", n_resamples=_REPLICATES, seed=1)


# ---------------------------------------------------------------------------------------------
# Threshold invariants widened for the bootstrap methods
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["parametric_bootstrap", "cell_bootstrap"])
def test_an_ok_bounded_threshold_may_carry_a_bootstrap_method(method: str) -> None:
    from robustmargin import Censoring
    from robustmargin.testing import fake_threshold

    base = fake_threshold()
    assert base.censoring is Censoring.NONE

    t = dataclasses.replace(base, interval_method=method)

    assert t.interval_method == method


@pytest.mark.parametrize("method", ["parametric_bootstrap", "cell_bootstrap"])
@pytest.mark.parametrize("censoring_name", ["OPEN_UPPER", "OPEN_LOWER"])
def test_an_open_interval_may_carry_a_bootstrap_method(method: str, censoring_name: str) -> None:
    from robustmargin import Censoring
    from robustmargin.testing import fake_threshold

    base = fake_threshold(censoring=Censoring[censoring_name])

    t = dataclasses.replace(base, interval_method=method)

    assert t.interval_method == method and t.censoring is Censoring[censoring_name]


@pytest.mark.parametrize("censoring_name", ["OPEN_UPPER", "OPEN_LOWER"])
def test_an_open_interval_still_refuses_delta(censoring_name: str) -> None:
    from robustmargin import Censoring
    from robustmargin.testing import fake_threshold

    base = fake_threshold(censoring=Censoring[censoring_name])

    with pytest.raises(ValueError):
        dataclasses.replace(base, interval_method="delta")


@pytest.mark.parametrize("method", ["parametric_bootstrap", "cell_bootstrap"])
def test_a_censored_exact_bound_result_refuses_a_bootstrap_method(method: str) -> None:
    from robustmargin import Censoring
    from robustmargin.testing import fake_threshold

    base = fake_threshold(censoring=Censoring.RIGHT)

    with pytest.raises(ValueError):
        dataclasses.replace(base, interval_method=method)


def test_an_unknown_interval_method_is_refused_by_the_threshold() -> None:
    from robustmargin.testing import fake_threshold

    with pytest.raises(ValueError):
        dataclasses.replace(fake_threshold(), interval_method="percentile_bootstrap")
